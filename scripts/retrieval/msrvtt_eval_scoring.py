# MSRVTT Scoring
import os
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
import json


test_list_path = f"{os.environ.get('KEYSCORE_DATA_ROOT', 'data')}/MSRVTT/structured-symlinks/test_list_miech.txt"
TOP_K = [1, 5, 10]

# ======= 0. Init ==========
res_dir = "_msrvtt"
sampled_types = ["uniform", "kanta", "stacfp"]

# embedding dirs
embedding_dirs = [f"{res_dir}/embeddings/{sampled_type}" for sampled_type in sampled_types]
[Path(embedding_dir).mkdir(parents=True, exist_ok=True) for embedding_dir in embedding_dirs]

# score dirs
score_dirs = [f"{res_dir}/scores/{sampled_type}" for sampled_type in sampled_types]
[Path(score_dir).mkdir(parents=True, exist_ok=True) for score_dir in score_dirs]

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

    
# Load Caption Annotations
def load_test_videos_ids(test_list_path:str):
    video_ids = []
    with open(test_list_path, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]
        print(f"Evaluating on {len(video_ids)} videos from test list")
    return video_ids


# ========= Compute Recall@K =========
def compute_recall(sim_matrix, gt, k_vals):
    recalls = {k: 0 for k in k_vals}
    for i, sims in enumerate(sim_matrix):
        sorted_indices = np.argsort(sims)[::-1]
        for k in k_vals:
            if gt[i] in sorted_indices[:k]:
                recalls[k] += 1
    for k in recalls:
        recalls[k] = 100.0 * recalls[k] / len(gt)
    return recalls

# ========= Evaluation =========
def evaluate(video_embs, text_embs, gt_text_to_video):
    res = {}
    # Normalize
    video_embs = video_embs / video_embs.norm(dim=1, keepdim=True)
    text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)

    # Text-to-Video
    sim_t2v = text_embs @ video_embs.T
    recall_t2v = compute_recall(sim_t2v.cpu().numpy(), gt_text_to_video, TOP_K)

    # Video-to-Text
    sim_v2t = video_embs @ text_embs.T
    recall_v2t = compute_recall(sim_v2t.cpu().numpy(), gt_text_to_video, TOP_K)

    print("\n--- Text-to-Video (1 caption per video) ---")
    res["recall_t2v"] = {}
    for k in TOP_K:
        print(f"R@{k}: {recall_t2v[k]:.2f}%")
        res["recall_t2v"][k] = recall_t2v[k]
        

    print("\n--- Video-to-Text (1 caption per video) ---")
    res["recall_v2t"] = {}
    for k in TOP_K:
        print(f"R@{k}: {recall_v2t[k]:.2f}%")
        res["recall_v2t"][k] = recall_v2t[k]

    return res

# ------- Run Evaluation ---- #
final_res = {"best": {"recall_t2v": {}, "recall_v2t": {}}}
for k in TOP_K:
    if k not in final_res["best"]["recall_t2v"]:
        final_res["best"]["recall_t2v"][k] = 0.0
    if k not in final_res["best"]["recall_t2v"]:
        final_res["best"]["recall_v2t"][k] = 0.0

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

print("thresholds: ", thresholds)


normalize = False  # Set True if scores not in [0,1]

# Load Testing Video IDs
test_video_ids = load_test_videos_ids(test_list_path)



final_res = {}
final_res = {"best": {"recall_t2v": {}, "recall_v2t": {}}}
for k in TOP_K:
    final_res["best"]["recall_t2v"][k] = 0.0
    final_res["best"]["recall_v2t"][k] = 0.0


for i in range(len(embedding_dirs)):

    embedding_dir = embedding_dirs[i]
    score_dir = score_dirs[i]

    print("score_dir: ", score_dir)
    
    final_res[embedding_dir] = {"best": {"recall_t2v": {}, "recall_v2t": {}}}
    for k in TOP_K:
        final_res[embedding_dir]["best"]["recall_t2v"][k] = 0.0
        final_res[embedding_dir]["best"]["recall_v2t"][k] = 0.0
        
    # Try Differnt score threshold
    for t in thresholds:
    
        video_embs = []
        text_embs = []
        gt_text_to_video = []

        Cnt = 0
        
        # Get PE Embeddings
        for idx, video_id in enumerate(tqdm(test_video_ids, desc=f"Load Encoding ({score_dir} | {t})")):
    
            # load video_emb, frame_emb, and text_emb
            frame_embedding_path = os.path.join(embedding_dir, video_id, "frame_embedding.pt")
            video_embedding_path = os.path.join(embedding_dir, video_id, "video_embedding.pt")
            text_embedding_path = os.path.join(embedding_dir, video_id, "text_embedding.pt")
            
            frame_emb = torch.load(frame_embedding_path)
            video_emb = torch.load(video_embedding_path)
            
            text_emb = torch.load(text_embedding_path)
            text_emb = text_emb.squeeze(0) if text_emb.ndim == 2 and text_emb.shape[0] == 1 else text_emb
        
            # --- Load and prepare scores ---
            score_path = os.path.join(score_dir, video_id)
            # scores.pt  sim_drop.pt  sim_semantic.pt  sim_temporal.pt
            s = torch.load(f"{score_path}/scores.pt")
            s = s.get("scores", s).squeeze() if isinstance(s, dict) else s.squeeze()
            if normalize: s = F.normalize(s - s.min(), dim=0, p=2)
    
            # --- Generate & Save Masks ---
            binary_mask = (s >= t)
            # --- Selected Frames and Generated New Frame Embs ---
            selected_frame_emb = frame_emb[binary_mask]

            # Count Frames
            N, D = selected_frame_emb.shape
            Cnt += N
    
            # --- Mean Pooling for Video Embd ---#
            video_emb = mean_pool_video(selected_frame_emb)
            video_emb = video_emb.squeeze(0)  # from [1, D] → [D]

            if video_emb.shape == torch.Size([0, 0]):
                continue
            
            video_embs.append(video_emb)
            text_embs.append(text_emb)
            gt_text_to_video.append(idx)  # ith caption maps to ith video
     
        if not video_embs:
            continue
            
        video_embs = torch.stack(video_embs)
        text_embs = torch.stack(text_embs)

        avgFrameCnt = Cnt/len(test_video_ids)
        print("avgFrameCnt: " , avgFrameCnt)
    
        res = evaluate(video_embs, text_embs, gt_text_to_video)
        res["avgFrameCnt"] = avgFrameCnt
        
        print("res: ", res)
    
        final_res[embedding_dir][t] = res
    
        for k in TOP_K:
            if res["recall_t2v"][k] > final_res[embedding_dir]["best"]["recall_t2v"][k]:
                final_res[embedding_dir]["best"]["recall_t2v"][k] = res["recall_t2v"][k]
                final_res[embedding_dir]["best"]["recall_t2v"]["avgFrameCnt"] = avgFrameCnt
       
            if res["recall_v2t"][k] > final_res[embedding_dir]["best"]["recall_v2t"][k]:
                final_res[embedding_dir]["best"]["recall_v2t"][k] = res["recall_v2t"][k]
                final_res[embedding_dir]["best"]["recall_v2t"]["avgFrameCnt"] = avgFrameCnt

        
    for k in TOP_K:
        if final_res[embedding_dir]["best"]["recall_t2v"][k] > final_res["best"]["recall_t2v"][k]:
            final_res["best"]["recall_t2v"][k] = final_res[embedding_dir]["best"]["recall_t2v"][k]

        if final_res[embedding_dir]["best"]["recall_v2t"][k] > final_res["best"]["recall_v2t"][k]:
            final_res["best"]["recall_v2t"][k] = final_res[embedding_dir]["best"]["recall_v2t"][k]

                
    import json

    with open("res_msrvtt_scoring_sim_temporal.json", "w") as f:
        json.dump(final_res, f, indent=4)