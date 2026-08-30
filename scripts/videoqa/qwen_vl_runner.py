"""
Qwen2-VL / Qwen2.5-VL inference wrapper for multi-image video QA.

Supports both:
  - Open-ended QA  (MSR-VTT style: short free-form answer)
  - MCQ            (Video-MME style: answer A / B / C / D)

Supported model IDs (pass via --model_id):
  Qwen/Qwen2-VL-7B-Instruct        (~16 GB)
  Qwen/Qwen2-VL-2B-Instruct        (~5 GB, good for DGX Spark eval)
  Qwen/Qwen2-VL-72B-Instruct       (~160 GB, multi-GPU)
  Qwen/Qwen2.5-VL-7B-Instruct      (newer arch)
  Qwen/Qwen2.5-VL-72B-Instruct     (best quality)

Note on NVIDIA GB10 (DGX Spark, unified memory):
  device_map="auto" triggers the OOM killer on GB10 because the allocator
  treats unified memory like a discrete GPU. We load directly onto cuda:0
  instead, which correctly uses the unified memory pool.

Memory-saving defaults (tuned for DGX Spark / GB10 to avoid OOM):
  - max_image_pixels = 448×28×28 (~350 K px) per image
  - default batch size = 4 (faster; use 2 or 1 if OOM on GB10)
  - default max_new_tokens: open=16, MCQ=8 (answers are short)
  - use_flash_attn=True by default → flash_attention_2 when available (faster); falls back to SDPA
  - torch.cuda.empty_cache() called after each sub-batch
  - pre_downsample_frames=True: load and resize frames to max_image_pixels before
    the processor, so we never hold full-resolution frames in memory.

CRITICAL: images passed to processor() must be a list-of-lists aligned with texts:
  images[i] = list of images for sample i; len(images) == len(texts).
  Flattening all images into one list causes vision/text misalignment → tensor
  shape explosion → GPU OOM / Xid.

Frame count (K): Eval scripts use scored_k in {2, 4} and uniform_k in {8}. Each
  (frame_paths, question) or (frame_paths, question, options) must have
  len(frame_paths) == K for that config. The runner checks this and raises if
  process_vision_info returns a different number of images than in the message.
  Set QWEN_VL_DEBUG=1 to log frames_per_sample per batch.
"""

from __future__ import annotations
import gc
import math
import os
import re
import torch
from pathlib import Path
from typing import Optional

# Set QWEN_VL_DEBUG=1 to log per-batch frame counts (ensure K=2,4,8 is preserved).
_DEBUG = os.environ.get("QWEN_VL_DEBUG", "").strip() in ("1", "true", "yes")

# Optional PIL for pre-downsampling frames
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    Image = None  # type: ignore


def _is_qwen25(model_id: str) -> bool:
    """Return True for Qwen2.5-VL model IDs."""
    return "Qwen2.5-VL" in model_id or "qwen2.5-vl" in model_id.lower()


def _is_gb10() -> bool:
    """Detect NVIDIA GB10 (DGX Spark unified-memory GPU)."""
    try:
        name = torch.cuda.get_device_name(0).lower()
        return "gb10" in name or "gb200" in name or "grace" in name
    except Exception:
        return False


def _count_images_in_messages(msgs: list[dict]) -> int:
    """Count content items that are images (must match len of image_inputs from process_vision_info)."""
    n = 0
    for msg in msgs:
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            # process_vision_info appends to image_inputs only for "image" or "image_url"
            if "image" in item or "image_url" in item or item.get("type", "text") in ("image", "image_url"):
                n += 1
    return n


def _load_and_resize_frame(path: str | Path, max_pixels: int, factor: int = 28) -> "Image.Image":
    """Load image from path and resize to fit within max_pixels (dims divisible by factor)."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("PIL is required for pre_downsample_frames; install Pillow.")
    im = Image.open(path).convert("RGB")
    w, h = im.size  # PIL: (width, height)
    if h < factor or w < factor:
        return im
    # Keep aspect ratio, total pixels <= max_pixels, both dims divisible by factor.
    if w * h > max_pixels:
        beta = math.sqrt((w * h) / max_pixels)
        w_bar = max(factor, int(w / beta) // factor * factor)
        h_bar = max(factor, int(h / beta) // factor * factor)
        if w_bar * h_bar > max_pixels:
            w_bar = max(factor, (int(w / beta) // factor - 1) * factor)
            h_bar = max(factor, (int(h / beta) // factor - 1) * factor)
    else:
        w_bar = (w // factor) * factor or factor
        h_bar = (h // factor) * factor or factor
    return im.resize((w_bar, h_bar), Image.BICUBIC)


class QwenVLRunner:
    """Wraps Qwen2-VL / Qwen2.5-VL for multi-frame video question answering."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        # 336×28×28 (~263 K) per image for faster inference; use 448*28*28 for higher quality.
        max_image_pixels: int = 336 * 28 * 28,
        # True → flash_attention_2 when available (faster); falls back to SDPA on error.
        use_flash_attn: bool = True,
        # True → load in 8-bit (bitsandbytes) for lower memory and faster inference; falls back to dtype if unavailable.
        load_in_8bit: bool = True,
        # True: load frames and resize to max_image_pixels before the processor (saves memory).
        pre_downsample_frames: bool = True,
    ):
        from transformers import AutoProcessor
        if _is_qwen25(model_id):
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        else:
            from transformers import Qwen2VLForConditionalGeneration as ModelCls

        # On GB10 (DGX Spark), device_map="auto" misreads unified memory and
        # triggers the OOM killer. Load directly onto cuda:0 instead.
        gb10 = _is_gb10()
        if gb10:
            print("[INFO] NVIDIA GB10 detected — loading model directly to cuda:0 (unified memory).")

        load_kwargs: dict = {"torch_dtype": dtype}
        if gb10:
            load_kwargs["device_map"] = {"": "cuda:0"}
        else:
            load_kwargs["device_map"] = "auto"

        if load_in_8bit:
            try:
                load_8bit_kw = {k: v for k, v in load_kwargs.items() if k != "torch_dtype"}
                load_8bit_kw["load_in_8bit"] = True
                self.model = ModelCls.from_pretrained(
                    model_id,
                    attn_implementation="sdpa",
                    **load_8bit_kw,
                )
                print("[INFO] Model loaded in 8-bit (bitsandbytes)")
            except Exception as e:
                print(f"[WARN] load_in_8bit failed ({e}); falling back to {dtype}.")
                if use_flash_attn:
                    try:
                        self.model = ModelCls.from_pretrained(
                            model_id,
                            attn_implementation="flash_attention_2",
                            **load_kwargs,
                        )
                        print("[INFO] Attention: flash_attention_2")
                    except Exception as e2:
                        print(f"[WARN] flash_attention_2 unavailable ({e2}); using SDPA.")
                        self.model = ModelCls.from_pretrained(
                            model_id,
                            attn_implementation="sdpa",
                            **load_kwargs,
                        )
                else:
                    self.model = ModelCls.from_pretrained(
                        model_id,
                        attn_implementation="sdpa",
                        **load_kwargs,
                    )
        elif use_flash_attn:
            try:
                self.model = ModelCls.from_pretrained(
                    model_id,
                    attn_implementation="flash_attention_2",
                    **load_kwargs,
                )
                print("[INFO] Attention: flash_attention_2")
            except Exception as e:
                print(f"[WARN] flash_attention_2 unavailable ({e}); falling back to SDPA.")
                self.model = ModelCls.from_pretrained(
                    model_id,
                    attn_implementation="sdpa",
                    **load_kwargs,
                )
            print("[INFO] Attention: SDPA (fallback)")
        else:
            self.model = ModelCls.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                **load_kwargs,
            )
            print("[INFO] Attention: SDPA")

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            max_pixels=max_image_pixels,
        )
        self.device = next(self.model.parameters()).device
        self.max_image_pixels = max_image_pixels
        self.pre_downsample_frames = pre_downsample_frames and _PIL_AVAILABLE
        if pre_downsample_frames and not _PIL_AVAILABLE:
            print("[WARN] pre_downsample_frames=True but PIL not available; disabling.")
        print(f"[INFO] Model loaded on {self.device} | dtype={dtype} | max_px/img={max_image_pixels:,} | pre_downsample={self.pre_downsample_frames}")

    # ------------------------------------------------------------------
    def answer_open(
        self,
        image_paths: list[str | Path],
        question: str,
        max_new_tokens: int = 12,
    ) -> str:
        """Open-ended QA (MSR-VTT). Returns a short free-form answer."""
        return self.answer_open_batch([(image_paths, question)], max_new_tokens)[0]

    def answer_open_batch(
        self,
        batch: list[tuple[list[str | Path], str]],
        max_new_tokens: int = 12,
        max_batch_size: int = 4,
    ) -> list[str]:
        """Batched open-ended QA. batch = [(frame_paths, question), ...]. Each frame_paths must have length K (2, 4, or 8)."""
        results = []
        for i in range(0, len(batch), max_batch_size):
            chunk = batch[i : i + max_batch_size]
            frame_counts = [len(paths) for (paths, _) in chunk]
            if _DEBUG and i == 0:
                print(f"[QWEN_VL_DEBUG] answer_open_batch chunk: batch_size={len(chunk)}, frames_per_sample={frame_counts}")
            for paths, _ in chunk:
                if len(paths) == 0:
                    raise ValueError("Each sample must have at least 1 frame (K>=1). Got 0 frames.")
            prompts = [
                (
                    "You are watching a video. The frames below are sampled from the video in temporal order.\n"
                    f"Question: {question}\n"
                    "Give a short, direct answer (a single word or brief phrase)."
                )
                for _, question in chunk
            ]
            messages_list = [
                [{"role": "user", "content":
                    [{"type": "image", "image": str(p)} for p in paths] +
                    [{"type": "text", "text": prompt}]
                }]
                for (paths, _), prompt in zip(chunk, prompts)
            ]
            results.extend(self._run_batch(messages_list, max_new_tokens))
            gc.collect()
            torch.cuda.empty_cache()
        return results

    # ------------------------------------------------------------------
    def answer_mcq(
        self,
        image_paths: list[str | Path],
        question: str,
        options: list[str],
        max_new_tokens: int = 6,
    ) -> str:
        """MCQ (Video-MME). Returns predicted letter A/B/C/D."""
        return self.answer_mcq_batch([(image_paths, question, options)], max_new_tokens)[0]

    def answer_mcq_batch(
        self,
        batch: list[tuple[list[str | Path], str, list[str]]],
        max_new_tokens: int = 6,
        max_batch_size: int = 4,
    ) -> list[str]:
        """Batched MCQ. batch = [(frame_paths, question, options), ...]. Each frame_paths must have length K (2, 4, or 8)."""
        results = []
        for i in range(0, len(batch), max_batch_size):
            chunk = batch[i : i + max_batch_size]
            frame_counts = [len(paths) for (paths, _, _) in chunk]
            if _DEBUG and i == 0:
                print(f"[QWEN_VL_DEBUG] answer_mcq_batch chunk: batch_size={len(chunk)}, frames_per_sample={frame_counts}")
            for paths, _, _ in chunk:
                if len(paths) == 0:
                    raise ValueError("Each sample must have at least 1 frame (K>=1). Got 0 frames.")
            prompts = [
                (
                    "You are watching a video. The frames below are sampled from the video in temporal order.\n"
                    f"Question: {question}\n"
                    f"Options:\n{chr(10).join(options)}\n"
                    "Reply with only the letter of the correct option (A, B, C, or D)."
                )
                for _, question, options in chunk
            ]
            messages_list = [
                [{"role": "user", "content":
                    [{"type": "image", "image": str(p)} for p in paths] +
                    [{"type": "text", "text": prompt}]
                }]
                for (paths, _, _), prompt in zip(chunk, prompts)
            ]
            raw = self._run_batch(messages_list, max_new_tokens)
            results.extend(self._extract_letter(r) for r in raw)
            gc.collect()
            torch.cuda.empty_cache()
        return results

    def _messages_with_downsampled_images(
        self, messages_list: list[list[dict]]
    ) -> list[list[dict]]:
        """Return a copy of messages_list with path-only images replaced by resized PIL Images."""
        import copy
        out: list[list[dict]] = []
        for msgs in messages_list:
            new_msgs = copy.deepcopy(msgs)
            for msg in new_msgs:
                content = msg.get("content") or []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img = item.get("image")
                        if isinstance(img, (str, Path)) and not str(img).startswith(("http://", "https://", "data:")):
                            item["image"] = _load_and_resize_frame(
                                img, self.max_image_pixels
                            )
            out.append(new_msgs)
        return out

    # ------------------------------------------------------------------
    def _run_batch(
        self,
        messages_list: list[list[dict]],
        max_new_tokens: int,
    ) -> list[str]:
        """Core batched inference. Each element of messages_list is one conversation."""
        from qwen_vl_utils import process_vision_info

        # Optionally replace path-only images with pre-resized PIL images.
        if self.pre_downsample_frames:
            messages_list = self._messages_with_downsampled_images(messages_list)

        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]

        # Keep images as list-of-lists: images[i] = images for sample i.
        # len(images) == len(texts). Do NOT flatten (causes vision/text misalignment → OOM).
        image_inputs_per_sample: list[list] = []
        for idx, msgs in enumerate(messages_list):
            expected_images = _count_images_in_messages(msgs)
            img_inputs, _ = process_vision_info(msgs)
            got = len(img_inputs) if img_inputs else 0
            if got != expected_images:
                raise ValueError(
                    f"Frame count mismatch for sample {idx}: message has {expected_images} image(s), "
                    f"process_vision_info returned {got}. Ensure each sample has exactly K frames (2, 4, or 8)."
                )
            image_inputs_per_sample.append(img_inputs if img_inputs else [])

        if len(image_inputs_per_sample) != len(texts):
            raise ValueError(
                f"Image/text alignment broken: {len(image_inputs_per_sample)} image lists vs {len(texts)} texts. "
                "This would cause OOM or wrong predictions."
            )

        if _DEBUG:
            frame_counts = [len(imgs) for imgs in image_inputs_per_sample]
            print(f"[QWEN_VL_DEBUG] _run_batch: batch_size={len(texts)}, frames_per_sample={frame_counts}")

        inputs = self.processor(
            text=texts,
            images=image_inputs_per_sample,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        # With left-padding, generated tokens start at position input_ids.shape[1]
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return [
            self.processor.decode(g, skip_special_tokens=True).strip()
            for g in generated
        ]

    # kept for backward compatibility
    def _run(self, image_paths, prompt, max_new_tokens):
        msgs = [[{"role": "user", "content":
            [{"type": "image", "image": str(p)} for p in image_paths] +
            [{"type": "text", "text": prompt}]
        }]]
        return self._run_batch(msgs, max_new_tokens)[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_letter(text: str) -> str:
        """Extract the first A/B/C/D letter from model output."""
        m = re.search(r"\b([A-D])\b", text.upper())
        return m.group(1) if m else text.strip()[:1].upper()
