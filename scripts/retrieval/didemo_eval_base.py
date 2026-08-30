import os
import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import sys, os

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Frame Sampling
from keyscore.proposals.uniform import FrameSampler_Uniform
from keyscore.proposals.katna import FrameSampler_Kanta
from keyscore.proposals.stacfp import FrameSampler_STACFP

# Frame and Text Encoder
from keyscore.encoders.perception_encoder import PerceptionEncoder

# Frame Scoring
from keyscore.scoring.frame_scorer import FrameScorer

# ======= 0. Init ==========
res_dir = "_didemo"
sampled_types = ["uniform", "kanta", "stacfp"]

# sampled frame dirs
sampled_frames_dirs = [f"{res_dir}/sampled_frames/{sampled_type}" for sampled_type in sampled_types]
[Path(sampled_frames_dir).mkdir(parents=True, exist_ok=True) for sampled_frames_dir in sampled_frames_dirs]

# embedding dirs
embedding_dirs = [f"{res_dir}/embeddings/{sampled_type}" for sampled_type in sampled_types]
[Path(embedding_dir).mkdir(parents=True, exist_ok=True) for embedding_dir in embedding_dirs]

# score dirs
score_dirs = [f"{res_dir}/scores/{sampled_type}" for sampled_type in sampled_types]
[Path(score_dir).mkdir(parents=True, exist_ok=True) for score_dir in score_dirs]

DATA_ROOT = os.environ.get("KEYSCORE_DATA_ROOT", "data")
video_dir = f"{DATA_ROOT}/DiDeMo/video/test/"
TOP_K = [1, 5, 10]

# Load Caption Annotations and 
DIDEMO_ANNOTATION_PATH = f"{DATA_ROOT}/DiDeMo/didemo_test.json"
with open(DIDEMO_ANNOTATION_PATH, 'r') as f:
    annotations = json.load(f)
    # print(annotations)
    seen_video_ids = set()
    
    caption_dict = {}
    for ann in annotations:
        vid = ann['video'].split("/")[1]
        cap = ann['caption']
        caption_dict.setdefault(vid, []).append(cap)

test_video_ids = list(caption_dict.keys())

# # ======= 1. Frame Sampling ==========
# def process_all_videos(sampler, video_dir: str, video_ids: list, output_dir: str):

#     for video_id in tqdm(sorted(video_ids), desc="Frame Samplers ({output_dir}"):
#         video_path = Path(f"{video_dir}/{video_id}")
#         output_folder = Path(f"{output_dir}/{video_id}")

#         sampler.extract_frames(str(video_path), str(output_folder))
        
# for i in range(len(sampled_frames_dirs)):

#     if i == 1: # skip kanta for now
#         continue
    
#     if i == 0:
#         sampler = FrameSampler_Uniform(no_of_frames=8)
#     elif i == 1:
#         sampler = FrameSampler_Kanta(no_of_frames=8)
#     elif i == 2:
#         sampler = FrameSampler_STACFP(
#             frame_step=5,         # tune based on FPS; 10 is fine for 1–2 min clips
#             gamma_time=5,        # stronger temporal grouping
#             k_min=6,
#             k_max=30
#         )

#     process_all_videos(sampler,
#                        video_dir=video_dir,
#                        video_ids=test_video_ids,
#                        output_dir=sampled_frames_dirs[i])

# ======= 2. Frame and Text Embedding ==========
from pathlib import Path

def encode_sampled_frames(encoder,
                          sampled_frames_dir: str,
                          embedding_dir: str,
                          video_ids: list):

    # Get PE Embeddings
    for idx, video_id in enumerate(tqdm(test_video_ids, desc="PE Video Encoding")):

        # create folder to save embeddings
        out_dir = Path(f"{embedding_dir}/{video_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        video_embedding_path = os.path.join(out_dir, "video_embedding.pt")
        frame_embedding_path = os.path.join(out_dir, "frame_embedding.pt")
        text_embedding_path = os.path.join(out_dir, "text_embedding.pt")

        

        # Video Emb from Folder
        frame_embs, video_emb = encoder.compute_video_emb(
            video_frame_folder=f"{sampled_frames_dir}/{video_id}",
            return_frame_embs=True,
        )

        # Text Emb
        caption = caption_dict[video_id][0]
        encoder.compute_text_emb(caption, save_path=text_embedding_path)
        
        torch.save(frame_embs, frame_embedding_path)
        torch.save(video_emb, video_embedding_path)

# Use PE Encoder
pe = PerceptionEncoder()

for i in range(len(embedding_dirs)):

    if i == 1: # skip kanta for now
        continue
    

    print("embedding_dir: ", embedding_dirs[i])
    
    encode_sampled_frames(encoder=pe,
                          sampled_frames_dir=sampled_frames_dirs[i],
                          embedding_dir=embedding_dirs[i],
                          video_ids=test_video_ids)

# ======= 3. Frame Scoring ==========
def score_sampled_frames(frame_scorer,
                         embedding_dir: str,
                         score_dir: str,
                         video_ids: list,
                         caption_dict: dict):

    # Get PE Embeddings
    for idx, video_id in enumerate(tqdm(test_video_ids, desc=f"PE Video Scoring")):

        # create folder to save embeddings
        out_dir = Path(f"{score_dir}/{video_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        video_embedding_path = os.path.join(embedding_dir, video_id, "video_embedding.pt")
        frame_embedding_path = os.path.join(embedding_dir, video_id, "frame_embedding.pt")
        text_embedding_path = os.path.join(embedding_dir, video_id, "text_embedding.pt")
        frame_paths_path = os.path.join(embedding_dir, video_id, "frame_paths.json")

        frame_embeddings = torch.load(frame_embedding_path)
        text_embeddings=torch.load(text_embedding_path)

        
        if text_embeddings.dim() == 2 and text_embeddings.size(0) == 1:
            text_embeddings = text_embeddings.squeeze(0) 

        out_frame_scores = frame_scorer.compute(frame_embeddings, text_embeddings)
        torch.save(out_frame_scores["sim_semantic"], out_dir / "sim_semantic.pt")
        torch.save(out_frame_scores["sim_temporal"], out_dir / "sim_temporal.pt")
        torch.save(out_frame_scores["sim_drop"], out_dir / "sim_drop.pt")
        torch.save(out_frame_scores["scores"], out_dir / "scores.pt")

# Load Frame Scorer
from keyscore.scoring.frame_scorer import *
frame_scorer = FrameScorer(
    pooling=MeanPooling(),
    alpha=0.7,  # semantic
    beta=0.1,   # temporal
    gamma=0.2,  # drop
    normalize_each=True
)


for i in range(len(score_dirs)):

    if i == 1: # skip kanta for now
        continue
    

    print("score_dir: ", score_dirs[i])
    
    score_sampled_frames(frame_scorer=frame_scorer,
                         embedding_dir=embedding_dirs[i],
                         score_dir=score_dirs[i],
                         video_ids=test_video_ids,
                         caption_dict=caption_dict)


# ======= 4. Video-Text Retrieval ==========
# ========= Load 1 caption per video based on test_list =========
def load_embeddings(embedding_dir, test_video_ids):
    
    video_embs = []
    text_embs = []
    gt_text_to_video = []
    Cnt = 0

    for idx, video_id in enumerate(tqdm(test_video_ids, desc="Loading")):

        video_embedding_path = os.path.join(embedding_dir, video_id, "video_embedding.pt")
        text_embedding_path = os.path.join(embedding_dir, video_id, "text_embedding.pt")  # Only use cap0
        frame_embedding_path = os.path.join(embedding_dir, video_id, "frame_embedding.pt")

        if not os.path.exists(video_embedding_path) or not os.path.exists(text_embedding_path):
            print(f"Skipping missing: {video_id}")
            print("video_embedding_path: ", video_embedding_path)
            print("text_embedding_path: ",text_embedding_path)
            continue

        video_emb = torch.load(video_embedding_path)

        video_emb = torch.load(video_embedding_path).squeeze(0)

        if video_emb.shape == torch.Size([0, 0]):
                continue
        
        text_emb = torch.load(text_embedding_path).squeeze(0)

        # Count Frames
        frames_emb = torch.load(frame_embedding_path).squeeze(0)
        N, D = frames_emb.shape
        Cnt += N

        video_embs.append(video_emb)
        text_embs.append(text_emb)
        gt_text_to_video.append(idx)  # ith caption maps to ith video

    video_embs = torch.stack(video_embs)  # [N, D]
    text_embs = torch.stack(text_embs)    # [N, D]
    
    avgFrameCnt = Cnt/len(test_video_ids)
    print("avgFrameCnt: " , avgFrameCnt)
    
    return video_embs, text_embs, gt_text_to_video, avgFrameCnt

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

final_res = {}
# ------- Run Evaluation ---- #
final_res = {"best": {"recall_t2v": {}, "recall_v2t": {}}}
for k in TOP_K:
    final_res["best"]["recall_t2v"][k] = 0.0
    final_res["best"]["recall_v2t"][k] = 0.0


for i in range(len(embedding_dirs)):

    if i == 1: # skip kanta for now
        continue

    embedding_dir = embedding_dirs[i]
    print("embedding_dir: ", embedding_dir)
    final_res[embedding_dir] ={}
    
    video_embs, text_embs, gt_text_to_video, avgFrameCnt = load_embeddings(embedding_dir=embedding_dir,
                                                                           test_video_ids=test_video_ids)

    res = evaluate(video_embs, text_embs, gt_text_to_video)
    res["avgFrameCnt"] = avgFrameCnt
    final_res[embedding_dir] = res

    for k in TOP_K:
        if final_res["best"]["recall_t2v"][k] <= final_res[embedding_dir]["recall_t2v"][k]:
            final_res["best"]["recall_t2v"][k] = final_res[embedding_dir]["recall_t2v"][k]
        if final_res["best"]["recall_v2t"][k] <= final_res[embedding_dir]["recall_v2t"][k]:
            final_res["best"]["recall_v2t"][k] = final_res[embedding_dir]["recall_v2t"][k]
    
    import json

    with open("res_msrvtt.json", "w") as f:
        json.dump(final_res, f, indent=4)
    