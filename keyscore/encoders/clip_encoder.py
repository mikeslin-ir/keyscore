import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import argparse
import torch
import torch.nn.functional as F
import cv2
from transformers import CLIPModel, CLIPProcessor

from torch.utils.data import Dataset, DataLoader
import contextlib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # set before any HF/transformers imports

class ClipEncoder:
    """
    Minimal CLIP helper:
      - encode(): optionally encode frames (folder) and/or text, save results
      - compute_probabilities(): load saved video/text embeddings and output probabilities
    Uses cv2 for image IO.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        model_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,  # NEW
        prefetch_factor: int = 2,  # NEW
        amp: bool = True,  # NEW (autocast on CUDA)
        use_fast_processor: bool = True  # NEW (silence HF warning & speed)
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.amp = amp and self.device.startswith("cuda")
        self.model_id = model_id
        self.model, self.processor = self._load_model_and_processor(use_fast_processor)

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True  # speed on Ampere+

    def _load_model_and_processor(self, use_fast: bool):
        # Explicit float32: other models (e.g. Qwen2-VL) loaded with bfloat16
        # temporarily change torch's global default dtype via accelerate, which
        # would cause CLIP to silently load in bfloat16 without this override.
        model = CLIPModel.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        ).to(self.device).eval()
        processor = CLIPProcessor.from_pretrained(self.model_id, use_fast=use_fast)
        return model, processor

    # ------------ DataLoader-based fast path ------------
    class _FrameDataset(Dataset):
        def __init__(self, paths: List[Path]):
            self.paths = paths

        def __len__(self): return len(self.paths)

        def __getitem__(self, idx: int):
            p = self.paths[idx]
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"Failed to read image: {p}")
            # BGR -> RGB (np.uint8)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # ---------- Files & IO ----------

    @staticmethod
    def _ensure_dir(p: Union[str, Path]) -> Path:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _list_images(self, folder: Union[str, Path]) -> List[Path]:
        folder = Path(folder)
        paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in self.IMG_EXTS]
        if not paths:
            raise FileNotFoundError(f"No images found in: {folder}")
        return [p.resolve() for p in paths]

    @staticmethod
    def _load_images_cv2(paths: List[Path]) -> List:
        """Load with cv2 (BGR) → RGB numpy arrays."""
        imgs_rgb = []
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"Failed to read image: {p}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs_rgb.append(im)
        return imgs_rgb

    @staticmethod
    def _save_tensor(t: torch.Tensor, path: Union[str, Path]):
        torch.save(t.cpu(), Path(path))

    # ---------- Embedding ----------

    # @torch.no_grad()
    # def embed_frames(self, image_paths: List[Path]) -> torch.Tensor:
    #     """
    #     Per-frame embeddings: [N, D].
    #     """
    #     feats_all = []
    #     for i in range(0, len(image_paths), self.batch_size):
    #         batch_paths = image_paths[i : i + self.batch_size]
    #         batch_imgs = self._load_images_cv2(batch_paths)
    #         inputs = self.processor(images=batch_imgs, return_tensors="pt")
    #         pixel_values = inputs["pixel_values"].to(self.device)
    #         feats = self.model.get_image_features(pixel_values=pixel_values)
    #         feats_all.append(feats.float().cpu())
    #     return torch.cat(feats_all, dim=0)

    @torch.inference_mode()
    def embed_frames(self, image_paths: List[Path]) -> torch.Tensor:
        """
        Per-frame embeddings [N, D] with:
          - parallel cv2 decode via DataLoader (num_workers)
          - CLIPProcessor in collate_fn
          - AMP on CUDA (autocast)
        """
        ds = self._FrameDataset(image_paths)

        # --- inside embed_frames(...) replace collate_fn with ---
        def collate_fn(batch_rgb_np):
            # Try image_processor first; fallback to legacy feature_extractor
            iproc = getattr(self.processor, "image_processor", None)
            if iproc is None:
                iproc = getattr(self.processor, "feature_extractor", None)
                if iproc is None:
                    raise RuntimeError("No image processor/feature extractor found on processor.")
            out = iproc(images=batch_rgb_np, return_tensors="pt")
            return out["pixel_values"]  # [B,3,H,W]

        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.startswith("cuda"),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn,
        )

        feats_all = []
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext()
        with amp_ctx:
            for pixel_values in dl:
                pixel_values = pixel_values.to(self.device, non_blocking=True)
                feats = self.model.get_image_features(pixel_values=pixel_values)
                # transformers >=5.x may return a model-output object instead of a
                # tensor; extract the projected image embeddings if so.
                if not isinstance(feats, torch.Tensor):
                    image_embeds = getattr(feats, "image_embeds", None)
                    if image_embeds is not None:
                        feats = image_embeds
                    elif getattr(feats, "pooler_output", None) is not None:
                        # pooler_output is already the projected CLIP image
                        # embedding (matches text_projection dim).
                        feats = feats.pooler_output
                    else:
                        feats = feats[0]
                # keep outputs on CPU as float32 for saving/compat
                feats_all.append(feats.detach().to("cpu", copy=True).float())
        return torch.cat(feats_all, dim=0)

    @torch.inference_mode()
    def embed_text(self, text):
        texts = [text] if isinstance(text, str) else text
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Two-layer guard against bfloat16 overflow in CLIP's causal-mask creation:
        #
        # 1. Other models (e.g. Qwen2-VL) loaded with torch_dtype=bfloat16 call
        #    torch.set_default_dtype(bfloat16) internally via accelerate and do
        #    not always restore it.  CLIP's _make_causal_mask calls torch.full()
        #    with torch.finfo(hidden_states.dtype).min; when the global default
        #    dtype is bfloat16, that value overflows → RuntimeError.
        # 2. We therefore (a) temporarily reset the global default dtype to
        #    float32, (b) cast the text sub-modules to float32, and (c) disable
        #    CUDA autocast, so the entire forward runs cleanly in float32.
        saved_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.autocast("cuda", enabled=False):
                text_model_fp32 = self.model.text_model.float()
                text_proj_fp32  = self.model.text_projection.float()
                text_out = text_model_fp32(**inputs)
                pooled   = text_out[1]                  # [B, D_text]
                feats    = text_proj_fp32(pooled).cpu() # [B, D_clip]
        finally:
            torch.set_default_dtype(saved_default_dtype)
        return feats[0] if isinstance(text, str) else feats

    @staticmethod
    def mean_pool(embs: torch.Tensor) -> torch.Tensor:
        """Mean pool frames: [N, D] -> [D]."""
        return embs.mean(dim=0)

    # ---------- Public API ----------

    def encode(
        self,
        frame_folder_path: Optional[Union[str, Path]] = None,
        text: Optional[Union[str, List[str]]] = None,
        output_dir: Union[str, Path] = "outs",
        save_paths_json: bool = True,
    ):
        """
        Encode frames and/or text. Each is optional.
        Saves:
          - If frames provided:
              frame_embeddings.pt  (tensor [N,D])
              video_embedding.pt   (tensor [D], mean-pooled)
              frame_paths.json     (optional)
          - If text provided:
              text_embedding.pt    (tensor [D] or [T,D])
        """
        out = self._ensure_dir(output_dir)

        # Frames → embeddings (+ mean-pooled video embedding)
        if frame_folder_path is not None:
            frame_paths = self._list_images(frame_folder_path)
            frame_embs = self.embed_frames(frame_paths)
            self._save_tensor(frame_embs, out / "frame_embeddings.pt")
            video_emb = self.mean_pool(frame_embs)
            self._save_tensor(video_emb, out / "video_embedding.pt")
            if save_paths_json:
                with open(out / "frame_paths.json", "w") as f:
                    json.dump([str(p) for p in frame_paths], f, indent=2)

        # Text → embeddings
        if text is not None:
            text_emb = self.embed_text(text)
            self._save_tensor(text_emb, out / "text_embedding.pt")

    # ---------- Similarity / Probabilities ----------

    @staticmethod
    def _load_embeddings_maybe_expand(path: Union[str, Path]) -> torch.Tensor:
        """
        Loads a tensor from .pt.
          Accepts [D], [T,D], or [N,D].
          Returns 2D tensor [n, D].
        """
        t = torch.load(Path(path), map_location="cpu")
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t.float()

    @staticmethod
    def _softmax_probs(sim: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Convert similarity matrix to probabilities via softmax along `dim`.
        """
        return F.softmax(sim, dim=dim)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    @classmethod
    def cosine_similarity(
            cls,
            video_emb_path: str,
            text_emb_path: str,
            normalize: bool = True,
            mean_pool_frames: bool = True,
    ) -> float:
        """
        Single scalar similarity between one video (or frame set) and one text.
        If video_emb_path points to frame_embeddings.pt and mean_pool_frames=True,
        mean-pool frames first.
        Returns a float in [-1, 1] (cosine similarity).
        """
        V = cls._load_embeddings_maybe_expand(video_emb_path)  # [V,D] or [N,D]
        T = cls._load_embeddings_maybe_expand(text_emb_path)  # [T,D]

        # expect one text
        if T.shape[0] != 1:
            raise ValueError("cosine_similarity expects exactly one text embedding.")

        # If per-frame, mean-pool to a single video vector
        if V.shape[0] > 1 and Path(video_emb_path).name == "frame_embeddings.pt" and mean_pool_frames:
            V = V.mean(dim=0, keepdim=True)

        if V.shape[0] != 1:
            raise ValueError("cosine_similarity expects exactly one video embedding (after pooling).")

        v = V[0]
        t = T[0]
        if normalize:
            v = F.normalize(v, dim=-1)
            t = F.normalize(t, dim=-1)
        sim = torch.dot(v, t).item()
        return sim

    @classmethod
    def frame_attention_probs(
            cls,
            frame_emb_path: str,
            text_emb_path: str,
            temperature: float = 1.0,
            normalize: bool = True,
    ) -> torch.Tensor:
        """
        Probability distribution over frames for a single text.
        Good for verifying that some frames align more than others.
        Returns [N] that sums to 1.
        """
        Fm = cls._load_embeddings_maybe_expand(frame_emb_path)  # [N,D]
        T = cls._load_embeddings_maybe_expand(text_emb_path)  # [1,D] expected
        if T.shape[0] != 1:
            raise ValueError("frame_attention_probs expects exactly one text embedding.")
        if normalize:
            Fm = F.normalize(Fm, dim=-1)
            T = F.normalize(T, dim=-1)
        logits = (Fm @ T[0]) / max(temperature, 1e-8)  # [N]
        return torch.softmax(logits, dim=0)

    @classmethod
    def compute_probabilities(
            cls,
            video_emb_path: str,
            text_emb_path: str,
            temperature: float = 1.0,
            normalize: bool = True,
            softmax_over: str = "video",
    ) -> torch.Tensor:
        """
        Original method. Note: if the softmax axis has length 1, the result is trivially 1.0.
        """
        V = cls._load_embeddings_maybe_expand(video_emb_path)  # [V,D] or [N,D]
        T = cls._load_embeddings_maybe_expand(text_emb_path)  # [T,D]

        # If V is per-frame [N,D], treat as single video by mean-pooling.
        if V.shape[0] > 1 and Path(video_emb_path).name == "frame_embeddings.pt":
            V = V.mean(dim=0, keepdim=True)  # [1,D]

        if normalize:
            V = F.normalize(V, dim=-1)
            T = F.normalize(T, dim=-1)

        sim = (V @ T.t()) / max(temperature, 1e-8)  # [V,T]

        if softmax_over.lower() == "video":
            # need V >= 2 to get meaningful probabilities
            if V.shape[0] == 1:
                raise ValueError("Trivial softmax: only one video candidate. Use cosine_similarity() instead.")
            probs = torch.softmax(sim, dim=0).t()  # [T,V]
        elif softmax_over.lower() == "text":
            # need T >= 2 to get meaningful probabilities
            if T.shape[0] == 1:
                raise ValueError("Trivial softmax: only one text candidate. Use cosine_similarity() instead.")
            probs = torch.softmax(sim, dim=1)  # [V,T]
        else:
            raise ValueError("softmax_over must be 'video' or 'text'")

        return probs

    def save_all_frames(
        self,
        video_path: Union[str, Path],
        save_dir: Union[str, Path],
        prefix: str = "frame",
        ext: str = ".jpg",
        zero_pad: int = 6,
        save_index_json: bool = True,
    ) -> List[Path]:
        """
        Save ALL frames from `video_path` into `save_dir`.
        Returns a list of saved file paths.

        Args:
          video_path: path to the video file.
          save_dir:   folder to write frames (created if missing).
          prefix:     filename prefix (e.g., 'frame' -> frame_000001.jpg).
          ext:        image extension, e.g., '.jpg', '.png'.
          zero_pad:   zero-padding length in filenames.
          save_index_json: also dump frame_paths.json in the folder.
        """
        save_dir = self._ensure_dir(save_dir)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        saved: List[Path] = []
        idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                idx += 1
                fname = f"{prefix}_{idx:0{zero_pad}d}{ext}"
                fpath = (save_dir / fname).resolve()
                # frame_bgr is already BGR; cv2.imwrite expects BGR
                if not cv2.imwrite(str(fpath), frame_bgr):
                    raise RuntimeError(f"cv2.imwrite failed for {fpath}")
                saved.append(fpath)
        finally:
            cap.release()

        if save_index_json:
            with open(save_dir / "frame_paths.json", "w") as f:
                json.dump([str(p) for p in saved], f, indent=2)

        return saved


# ---------------- CLI (optional) ----------------

def main():
    parser = argparse.ArgumentParser(description="Encode frames and/or text with CLIP; compute similarities.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # encode
    p_enc = subparsers.add_parser("encode", help="Encode frames and/or text, save to output folder.")
    p_enc.add_argument("--input-frames", type=str, default=None, help="Folder of sampled frames (optional).")
    p_enc.add_argument("--text", type=str, default=None, help="Text or path to a .txt file (optional).")
    p_enc.add_argument("--output-dir", type=str, required=True, help="Where to save embeddings.")
    p_enc.add_argument("--model-id", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    p_enc.add_argument("--batch-size", type=int, default=64)
    p_enc.add_argument("--device", type=str, default=None)

    # probs
    p_prob = subparsers.add_parser("probs", help="Compute probabilities from saved embeddings.")
    p_prob.add_argument("--video-emb", type=str, required=True, help="Path to video_embedding.pt or frame_embeddings.pt")
    p_prob.add_argument("--text-emb", type=str, required=True, help="Path to text_embedding.pt")
    p_prob.add_argument("--temperature", type=float, default=1.0)
    p_prob.add_argument("--normalize", action="store_true", help="Use cosine-like similarity.")
    p_prob.add_argument("--softmax-over", type=str, default="video", choices=["video", "text"])

    args = parser.parse_args()

    if args.cmd == "encode":
        # Optional: support reading text from file path
        text_value: Optional[Union[str, List[str]]] = args.text
        if text_value and os.path.isfile(text_value) and text_value.lower().endswith(".txt"):
            with open(text_value, "r", encoding="utf-8") as f:
                text_value = f.read().strip()

        enc = ClipEncoder(
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
        enc.encode(
            frame_folder_path=args.input_frames,
            text=text_value,
            output_dir=args.output_dir,
        )

    elif args.cmd == "probs":
        probs = ClipEncoder.compute_probabilities(
            video_emb_path=args.video_emb,
            text_emb_path=args.text_emb,
            temperature=args.temperature,
            normalize=args.normalize,
            softmax_over=args.softmax_over,
        )
        # Print as a small JSON-ish blob
        tensor_list = probs.tolist()
        print(json.dumps({"probs": tensor_list}, indent=2))


if __name__ == "__main__":
    main()

# #1) Python snippet (programmatic)
# from clip_encoder import ClipEncoder  # save the class code as clip_encoder.py
# from pathlib import Path
# import torch
#
# # Inputs
# frames_dir = "/path/to/video_frames"       # folder of sampled frames (e.g., frame_0001.jpg, ...)
# text       = "A person riding a bicycle on a city street."
# out_dir    = "/tmp/clip_out"
#
# # 1) Encode frames and text, saving outputs to out_dir
# enc = ClipEncoder(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", batch_size=64)
# enc.encode(frame_folder_path=frames_dir, text=text, output_dir=out_dir)
#
# # 2) Compute similarity probabilities between the saved video/text embeddings
# video_emb_path = Path(out_dir) / "video_embedding.pt"   # or "frame_embeddings.pt"
# text_emb_path  = Path(out_dir) / "text_embedding.pt"
#
# probs = ClipEncoder.compute_probabilities(
#     video_emb_path=video_emb_path,
#     text_emb_path=text_emb_path,
#     temperature=1.0,
#     normalize=True,           # cosine-like similarity
#     softmax_over="video",     # P(video | text) since there’s one video embedding
# )
#
# print("Probabilities (P(video | text)):", probs.squeeze(0).tolist())
# print("Scalar similarity probability:", float(probs.squeeze()))  # when single text + single video
#
#
# #2) CLI usage
# # Encode frames + text and save outputs
# python clip_encoder.py encode \
#   --input-frames "/path/to/video_frames" \
#   --text "A person riding a bicycle on a city street." \
#   --output-dir "/tmp/clip_out"
#
# # Compute probabilities from saved embeddings
# python clip_encoder.py probs \
#   --video-emb "/tmp/clip_out/video_embedding.pt" \
#   --text-emb "/tmp/clip_out/text_embedding.pt" \
#   --normalize \
#   --softmax-over video
