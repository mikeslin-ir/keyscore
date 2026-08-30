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
# **** changed
res_dir = "_msvd"
save_rs = "res_msvd.json"

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

# **** changed
DATA_ROOT = os.environ.get("KEYSCORE_DATA_ROOT", "data")
test_list_path = f"{DATA_ROOT}/MSVD/annotaions/msvd_data/test_list.txt"
video_dir = f"{DATA_ROOT}/MSVD/YouTubeClips/"

TOP_K = [1, 5, 10]

# Load Testing ID list
def load_test_videos_ids(test_list_path:str):
    video_ids = []
    with open(test_list_path, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]
        print(f"Evaluating on {len(video_ids)} videos from test list")
    return video_ids

test_video_ids = load_test_videos_ids(test_list_path)

# # **** changed
# # Load Caption Annotations and 
MSVD_ANNOTATION_PATH = f"{DATA_ROOT}/MSVD/AllVideoDescriptions.txt"
caption_dict = {}
with open(MSVD_ANNOTATION_PATH, 'r') as f:
    for line in f:
        if line.strip():
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                vid, cap = parts[0], parts[1]
                if vid in test_video_ids:
                    caption_dict.setdefault(vid, []).append(cap)

with open("msvd_caption_dict.json", "w") as f:
    json.dump(caption_dict, f, indent=4)

# ======= 1. Frame Sampling ==========
def process_all_videos(sampler, video_dir: str, video_ids: list, output_dir: str):

    for video_id in tqdm(sorted(video_ids), desc="Frame Samplers ({output_dir}"):
        video_path = Path(f"{video_dir}/{video_id}.avi")
        output_folder = Path(f"{output_dir}/{video_id}")

        sampler.extract_frames(str(video_path), str(output_folder))
        
for i in range(len(sampled_frames_dirs)):

    if i == 0:
        sampler = FrameSampler_Uniform(no_of_frames=8)
    elif i == 1:
        sampler = FrameSampler_Kanta(no_of_frames=30)
    elif i == 2:
        sampler = FrameSampler_STACFP(
            frame_step=5,         # tune based on FPS; 10 is fine for 1–2 min clips
            gamma_time=5,        # stronger temporal grouping
            k_min=5,
            k_max=25
        )

    print("sampled_frames_dir: ", sampled_frames_dirs[i])
    process_all_videos(sampler,
                       video_dir=video_dir,
                       video_ids=test_video_ids,
                       output_dir=sampled_frames_dirs[i])

# ======= 2. Frame and Text Embedding ==========
from pathlib import Path

def encode_sampled_frames_with_all_captions(encoder,
                                            sampled_frames_dir: str,
                                            embedding_dir: str,
                                            video_ids: list,
                                            caption_dict: dict):
    for idx, video_id in enumerate(tqdm(video_ids, desc=f"PE Video+AllCaptions Encoding")):

        tqdm.write(f"Processing video_id: {video_id}")
    
        out_dir = Path(f"{embedding_dir}/{video_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        
        video_embedding_path = os.path.join(out_dir, "video_embedding.pt")
        frame_embedding_path = os.path.join(out_dir, "frame_embedding.pt")
        all_text_embeddings_path = os.path.join(out_dir, "text_embedding.pt")
        tqdm.write(f"Processing all_text_embeddings_path: {all_text_embeddings_path}")
        
        
        # Video Embedding
        frame_embs, video_emb = encoder.compute_video_emb(
            video_frame_folder=f"{sampled_frames_dir}/{video_id}",
            save_video_emb_path=video_embedding_path,
            save_frames_emb_path=frame_embedding_path,
            return_frame_embs=True,
        )
        
        
        # Compute all caption embeddings and stack
        all_texts = caption_dict[video_id] # All ~40 captions
        all_text_embs = encoder.compute_text_emb(all_texts)

        torch.save(all_text_embs, all_text_embeddings_path)

# Use PE Encoder
pe = PerceptionEncoder()

for i in range(len(embedding_dirs)):

    print("embedding_dir: ", embedding_dirs[i])
    
    encode_sampled_frames_with_all_captions(encoder=pe,
                                            sampled_frames_dir=sampled_frames_dirs[i],
                                            embedding_dir=embedding_dirs[i],
                                            video_ids=test_video_ids,
                                            caption_dict=caption_dict)

# ======= 3. Frame Scoring ==========
def score_sampled_frames(frame_scorer,
                         embedding_dir: str,
                         score_dir: str,
                         video_ids: list,
                         caption_dict: dict):

    # Get PE Embeddings
    for idx, video_id in enumerate(tqdm(video_ids, desc=f"PE Video Scoring")):

        # create folder to save embeddings
        out_dir = Path(f"{score_dir}/{video_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        video_embedding_path = os.path.join(embedding_dir, video_id, "video_embedding.pt")
        frame_embedding_path = os.path.join(embedding_dir, video_id, "frame_embedding.pt")
        text_embedding_path = os.path.join(embedding_dir, video_id, "text_embedding.pt")
        frame_paths_path = os.path.join(embedding_dir, video_id, "frame_paths.json")

        frame_embeddings = torch.load(frame_embedding_path)

        N, D = frame_embeddings.shape
        if N == 0:
            continue
            
        text_embeddings=torch.load(text_embedding_path)
        text_embeddings = text_embeddings.mean(dim=0) 

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


    print("score_dir: ", score_dirs[i])
    
    score_sampled_frames(frame_scorer=frame_scorer,
                         embedding_dir=embedding_dirs[i],
                         score_dir=score_dirs[i],
                         video_ids=test_video_ids,
                         caption_dict=caption_dict)


# ======= 4. Video-Text Retrieval ==========
# ========= Load 1 caption per video based on test_list =========
def load_embeddings_all_captions(embedding_dir, test_video_ids, caption_dict):
    V_list, T_list = [], []
    gt_t2v = []                 # len T; each entry in [0..V-1]
    v2t_text_indices = []       # len V; list of lists of caption indices for that video
    Cnt = 0
    text_cursor = 0

    # # debug
    # misssing = {}
    # missing_video_emb = []
    # missing_cnt = 0

    for video_id in tqdm(test_video_ids, desc="Loading AllCaptions (strict)"):
        v_path = Path(embedding_dir) / video_id / "video_embedding.pt"
        t_path = Path(embedding_dir) / video_id / "text_embedding.pt"
        f_path = Path(embedding_dir) / video_id / "frame_embedding.pt"
        if not (v_path.exists() and t_path.exists() and f_path.exists()):
            # # debug
            # print(f"Skipping missing: {video_id}")
            # misssing[video_id] = f"{v_path} ; {t_path} ; {f_path}"
            # missing_cnt += 1
            continue

        # video -> [D]
        v = torch.load(v_path)
        N, D = v.shape
        
        if N == 0:
            # # debug
            # print(video_id, v.shape, "N==0" )
            # missing_video_emb.append(video_id)  
            # missing_cnt += 1
            continue
            
        if v.dim() == 2 and v.size(0) == 1: v = v.squeeze(0)
        assert v.dim() == 1, f"video_emb must be [D], got {v.shape}"

        # text -> [C,D]
        T = torch.load(t_path)
        if T.dim() == 3 and T.size(1) == 1: T = T.squeeze(1)
        elif T.dim() == 1: T = T.unsqueeze(0)
        assert T.dim() == 2, f"text embs must be [C,D], got {T.shape}"

        # frames for avg
        F = torch.load(f_path)
        if F.dim() == 3 and F.size(0) == 1: F = F.squeeze(0)
        assert F.dim() == 2, f"frame embs must be [F,D], got {F.shape}"
        Cnt += F.shape[0]

        V_idx = len(V_list)
        V_list.append(v)
        T_list.append(T)  # [C,D]
        C = T.shape[0]
        gt_t2v.extend([V_idx] * C)
        v2t_text_indices.append(list(range(text_cursor, text_cursor + C)))
        text_cursor += C


    # # debug
    # print("misssing: ", misssing)
    # print("missing_video_emb: ", missing_video_emb)
    # print("missing_cnt: ",missing_cnt)

    video_embs = torch.stack(V_list, dim=0)      # [V,D]
    text_embs  = torch.cat(T_list, dim=0)        # [T,D]

    avgFrameCnt = Cnt / max(1, len(V_list))
    return video_embs, text_embs, gt_t2v, v2t_text_indices, avgFrameCnt


# ========= Compute Recall@K =========
def compute_recall_t2v(sim_t2v, gt_t2v, k_vals):
    # sim_t2v: [T,V]
    order = np.argsort(sim_t2v, axis=1)[:, ::-1]
    recalls = {k: 0 for k in k_vals}
    T = sim_t2v.shape[0]
    for t in range(T):
        for k in k_vals:
            if gt_t2v[t] in order[t, :k]:
                recalls[k] += 1
    for k in k_vals:
        recalls[k] = 100.0 * recalls[k] / T
    return recalls

def compute_recall_v2t_multi(sim_v2t, v2t_text_indices, k_vals):
    # sim_v2t: [V,T]
    order = np.argsort(sim_v2t, axis=1)[:, ::-1]
    recalls = {k: 0 for k in k_vals}
    V = sim_v2t.shape[0]
    for v in range(V):
        correct = set(v2t_text_indices[v])
        for k in k_vals:
            topk = set(order[v, :k])
            if correct & topk:
                recalls[k] += 1
    for k in k_vals:
        recalls[k] = 100.0 * recalls[k] / V
    return recalls


def evaluate_all_caps(video_embs, text_embs, gt_t2v, v2t_text_indices, TOP_K):
    video_embs = video_embs / video_embs.norm(dim=1, keepdim=True)
    text_embs  = text_embs  / text_embs.norm(dim=1, keepdim=True)

    sim_t2v = (text_embs @ video_embs.T).cpu().numpy()  # [T,V]
    sim_v2t = (video_embs @ text_embs.T).cpu().numpy()  # [V,T]

    r_t2v = compute_recall_t2v(sim_t2v, gt_t2v, TOP_K)
    r_v2t = compute_recall_v2t_multi(sim_v2t, v2t_text_indices, TOP_K)
    return {"recall_t2v": r_t2v, "recall_v2t": r_v2t}

final_res = {}
# ------- Run Evaluation ---- #
final_res = {"best": {"recall_t2v": {}, "recall_v2t": {}}}
for k in TOP_K:
    final_res["best"]["recall_t2v"][k] = 0.0
    final_res["best"]["recall_v2t"][k] = 0.0


for i in range(len(embedding_dirs)):

    embedding_dir = embedding_dirs[i]
    print("embedding_dir: ", embedding_dir)
    final_res[embedding_dir] ={}
    
    video_embs, text_embs, gt_t2v, v2t_text_indices, avgFrameCnt = load_embeddings_all_captions(embedding_dir=embedding_dir,
                                                                                        test_video_ids=test_video_ids,
                                                                                       caption_dict=caption_dict)


    res = evaluate_all_caps(video_embs, text_embs, gt_t2v, v2t_text_indices, TOP_K)

    print("res: ", res)

    res["avgFrameCnt"] = avgFrameCnt
    final_res[embedding_dir] = res

    for k in TOP_K:
        if final_res["best"]["recall_t2v"][k] <= final_res[embedding_dir]["recall_t2v"][k]:
            final_res["best"]["recall_t2v"][k] = final_res[embedding_dir]["recall_t2v"][k]
        if final_res["best"]["recall_v2t"][k] <= final_res[embedding_dir]["recall_v2t"][k]:
            final_res["best"]["recall_v2t"][k] = final_res[embedding_dir]["recall_v2t"][k]

    print("final_res: ", final_res)
    
    import json

    with open("res_msvd.json", "w") as f:
        json.dump(final_res, f, indent=4)
    