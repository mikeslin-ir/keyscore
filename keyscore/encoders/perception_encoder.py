# pip install torch torchvision pillow opencv-python open_clip_torch
# optional: pip install faiss-cpu
# pip install torch torchvision pillow opencv-python open_clip_torch
# pip install jupyterlab ipykernel
# python -m ipykernel install --user --name=pe --display-name "Python (PE)"
import os
import glob
from typing import List, Union, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import open_clip


class PerceptionEncoder:
    """
    Wrapper around the Perception Encoder (PE Core bigG/14 @ 448) via OpenCLIP-remapped weights.

    - Images/frames -> image embeddings
    - Video -> per-frame embeddings + mean-pooled video embedding
    - Text -> text embeddings
    - Cosine similarities + probability lists for video <-> text
    - Save helpers for text_emb and video_emb

    Model card / how-to use (OpenCLIP remapped): timm/PE-Core-bigG-14-448
    """

    def __init__(
            self,
            model_id: str = "hf-hub:timm/PE-Core-bigG-14-448",
            device: Optional[str] = None,
            dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_id: OpenCLIP model identifier. Default loads PE-Core-bigG-14-448 from HF Hub.
            device: 'cuda'|'cpu'|'mps' or None to auto-select.
            dtype: torch dtype for inference (fp16 default). Use torch.float32 if on CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if device == "cpu":
            dtype = torch.float32  # better numerical stability on CPU

        self.device = device
        self.dtype = dtype

        # Create model + preprocess + tokenizer (OpenCLIP)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, device=self.device)
        self.model = self.model.to(self.device, dtype=self.dtype).eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)

        # For convenience
        self.context_length = getattr(self.model, "context_length", 77)

    # -------------------------
    # Encoding utilities
    # -------------------------
    @torch.inference_mode()
    def encode_images(
            self,
            image_paths: List[str],
            batch_size: int = 32,
            normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of image file paths into embeddings.

        Returns:
            Tensor [N, D] on CPU (L2-normalized if normalize=True)
        """
        embs = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(self.preprocess(img))
            pixel_batch = torch.stack(images, dim=0).to(self.device, dtype=self.dtype)

            feats = self.model.encode_image(pixel_batch, normalize=normalize)
            embs.append(feats.float().cpu())

        return torch.cat(embs, dim=0) if embs else torch.empty(0, 0)

    @torch.inference_mode()
    def encode_text(
            self,
            texts: Union[str, List[str]],
            normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode one or more text prompts.

        Returns:
            Tensor [T, D] on CPU (L2-normalized if normalize=True)
        """
        if isinstance(texts, str):
            texts = [texts]
        toks = self.tokenizer(texts, context_length=self.context_length).to(self.device)
        feats = self.model.encode_text(toks, normalize=normalize)
        return feats.float().cpu()

    # -------------------------
    # Video: frame sampling + encoding
    # -------------------------
    def _sample_video_frames(
            self,
            video_path: str,
            max_frames: int = 8,
    ) -> List[Image.Image]:
        """
        Evenly sample up to `max_frames` frames from a video and return PIL Images (RGB).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError("Could not determine frame count for video.")

        idxs = np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)

        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    @torch.inference_mode()
    def encode_video_frames(
            self,
            video_path: str,
            max_frames: int = 8,
            batch_size: int = 32,
            normalize: bool = True,
    ) -> torch.Tensor:
        """
        Sample frames from a video and encode them.

        Returns:
            frame_embs: Tensor [F, D] on CPU (L2-normalized if normalize=True)
        """
        frames = self._sample_video_frames(video_path, max_frames=max_frames)
        # print("number of frames: ", len(frames))
        embs = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i: i + batch_size]
            pixel_batch = torch.stack([self.preprocess(img) for img in batch], dim=0).to(self.device, dtype=self.dtype)
            feats = self.model.encode_image(pixel_batch, normalize=normalize)
            embs.append(feats.float().cpu())
        return torch.cat(embs, dim=0) if embs else torch.empty(0, 0)

    # -------------------------
    # Frame folder: encoding
    # -------------------------
    def encode_frame_folder(
            self,
            folder_path: str,
            image_glob: str = "*.*",
            batch_size: int = 32,
            normalize: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Encode all images in a folder (sorted by filename).

        Returns:
            (frame_embs [N, D], sorted_paths [N])
        """
        paths = sorted([p for p in glob.glob(os.path.join(folder_path, image_glob)) if _is_image_file(p)])
        print("number of sampled frames: ", len(paths))
        frame_embs = self.encode_images(paths, batch_size=batch_size, normalize=normalize)
        return frame_embs, paths

    # -------------------------
    # Pooling + similarity + probabilities
    # -------------------------
    @staticmethod
    def mean_pool_video(frame_embs: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool per-frame embeddings into a single video embedding, then L2-normalize.

        Args:
            frame_embs: [F, D], assumed already L2-normalized row-wise.
        Returns:
            video_emb: [1, D], L2-normalized
        """
        if frame_embs.numel() == 0:
            return torch.empty(0, 0)
        v = frame_embs.mean(dim=0, keepdim=True)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        return v

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity matrix between rows of a (N,D) and b (M,D).
        If rows aren't normalized, they will be normalized here.
        Returns: [N, M] torch.Tensor on CPU.
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Inputs must be 2D tensors [N,D] and [M,D].")
        a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return a @ b.T

    @staticmethod
    def probs_from_sim(sim: torch.Tensor, temperature: float = 0.07, dim: int = -1) -> torch.Tensor:
        """
        Convert similarity matrix to probabilities with a softmax over 'dim'.
        E.g., for Video→Text: dim=-1 over the text axis.
        """
        return F.softmax(sim / max(temperature, 1e-8), dim=dim)

    # -------------------------
    # Save helpers
    # -------------------------
    @staticmethod
    def _ensure_dir(path: str):
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_tensor(t: torch.Tensor, save_path: str):
        """
        Save a tensor as a .pt file. Creates parent dirs if needed.
        """
        parent = os.path.dirname(save_path)
        PerceptionEncoder._ensure_dir(parent)
        torch.save(t, save_path)

    # -------------------------
    # Convenience: compute & (optionally) save embeddings
    # -------------------------
    def compute_text_emb(
            self, texts: Union[str, List[str]], *, save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Returns:
            text_embs: [T, D]
        """
        text_embs = self.encode_text(texts)
        if save_path:
            self.save_tensor(text_embs, save_path)
        return text_embs

    def compute_video_emb(
            self,
            *,
            video_path: Optional[str] = None,
            video_frame_folder: Optional[str] = None,
            max_frames: int = 8,
            batch_size: int = 32,
            image_glob: str = "*.*",
            save_video_emb_path: Optional[str] = None,
            save_frames_emb_path: Optional[str] = None,
            return_frame_embs: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute mean-pooled video embedding from either a video file or a frame folder.

        Returns:
            - if return_frame_embs: (frame_embs [F,D], video_emb [1,D])
            - else: video_emb [1,D]
        """
        if (video_path is None) == (video_frame_folder is None):
            raise ValueError("Specify exactly one of video_path or video_frame_folder.")

        if video_path is not None:
            frame_embs = self.encode_video_frames(video_path, max_frames=max_frames, batch_size=batch_size)
        else:
            frame_embs, _ = self.encode_frame_folder(video_frame_folder, image_glob=image_glob, batch_size=batch_size)

        video_emb = self.mean_pool_video(frame_embs)

        if save_video_emb_path:
            self.save_tensor(video_emb, save_video_emb_path)

        if save_frames_emb_path:
            self.save_tensor(frame_embs, save_frames_emb_path)

        return (frame_embs, video_emb) if return_frame_embs else video_emb

    # -------------------------
    # New: text-video similarity (and probabilities)
    # -------------------------
    def text_video_similarity(
            self,
            *,
            text_embs: torch.Tensor,
            video_embs: torch.Tensor,
            temperature: float = 0.07,
            direction: str = "video_to_text",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine similarities and probability distributions.

        Args:
            text_embs:  [T, D]
            video_embs: [N, D] (use [1, D] for a single video)
            direction:  "video_to_text"  -> probs over texts for each video  (softmax on dim=-1)
                        "text_to_video"  -> probs over videos for each text  (softmax on dim=-1)
        Returns:
            sim:   [N, T] if direction == "video_to_text", else [T, N]
            probs: same shape as sim, softmaxed along the last dim
        """
        if direction not in {"video_to_text", "text_to_video"}:
            raise ValueError('direction must be "video_to_text" or "text_to_video"')

        if direction == "video_to_text":
            sim = self.cosine_similarity(video_embs, text_embs)  # [N, T]
            probs = self.probs_from_sim(sim, temperature=temperature, dim=-1)
        else:
            sim = self.cosine_similarity(text_embs, video_embs)  # [T, N]
            probs = self.probs_from_sim(sim, temperature=temperature, dim=-1)
        return sim, probs


def _is_image_file(p: str) -> bool:
    ext = os.path.splitext(p)[-1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]

# # Usage Example:
# from perception_encoder import PerceptionEncoder
#
# pe = PerceptionEncoder()
#
# # 1) Text → embeddings (and save)
# texts = [
#     "a person riding a bike",
#     "a dog running on the beach",
#     "a chef slicing vegetables in a kitchen",
# ]
# text_embs = pe.compute_text_emb(texts, save_path="outs/text_embs.pt")  # [T,D]
#
# # 2A) Video file → frame_embs + mean-pooled video_emb (and save)
# frame_embs, video_emb = pe.compute_video_emb(
#     video_path="/path/to/video.mp4",
#     max_frames=8,
#     save_path="outs/video_emb.pt",   # optional; saves [1,D]
#     return_frame_embs=True,
# )
#
# # 2B) (Alternative) Frame folder → frame_embs + video_emb
# # frame_embs, video_emb = pe.compute_video_emb(
# #     video_frame_folder="/path/to/frame_folder",
# #     save_path="outs/video_emb.pt",
# # )
#
# print("Per-frame embeddings:", tuple(frame_embs.shape))  # [F,D]
# print("Video embedding:", tuple(video_emb.shape))        # [1,D]
#
# # 3) Similarity & probabilities (Video→Text)
# sim_vt, probs_vt = pe.text_video_similarity(
#     text_embs=text_embs,
#     video_embs=video_emb,          # [1,D] or [N,D]
#     temperature=0.07,
#     direction="video_to_text",
# )
#
# print("\nVideo→Text probabilities:")
# for t, p in zip(texts, probs_vt[0].tolist()):
#     print(f"  {t}: {p:.4f}")
#
# # 4) (Optional) Text→Video probabilities across multiple videos
# # Suppose you have a batch of videos stacked as [N, D] in `video_bank`
# # sim_tv, probs_tv = pe.text_video_similarity(
# #     text_embs=text_embs,
# #     video_embs=video_bank,       # [N,D]
# #     temperature=0.07,
# #     direction="text_to_video",
# # )
