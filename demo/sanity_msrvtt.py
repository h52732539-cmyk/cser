"""Sanity check: reproduce pure cosine R@1 on MSR-VTT 1K from the cache."""
import sys, numpy as np, csv
from pathlib import Path

PR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PR))

cache = np.load(r"E:\Work\HKUST(2025)\video_query\video_retrieval_code_no_dataset\data\cache\msrvtt_cache.npz",
                 allow_pickle=True)
vids    = cache["video_ids"].astype(str).tolist()
vembs   = cache["video_embs"].astype("float32")   # (1000, 512)
protos  = cache["protos"].astype("float32")       # (1000, 6, 512)
pcounts = cache["proto_counts"].astype(np.int32)  # (1000,)
print(f"Videos: {len(vids)}, dim={vembs.shape[1]}")

# normalize
vembs  /= np.linalg.norm(vembs, axis=-1, keepdims=True) + 1e-9
for i in range(len(vids)):
    for k in range(pcounts[i]):
        n = np.linalg.norm(protos[i, k]) + 1e-9
        protos[i, k] /= n

# load queries
qs, gt = [], []
with open(r"E:\Work\HKUST(2025)\video_query\video_retrieval_code_no_dataset\data\msrvtt_test_1k.csv",
           "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qs.append(row["sentence"])
        gt.append(row["video_id"])

q_embs = np.load(r"E:\Work\HKUST(2025)\video_query\litevtr_multi_model_framework\BENCHMARK_MSRVTT_V2_limit100.text_embs.npy")
q_embs = q_embs.astype("float32")
q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9
print(f"Query embs: {q_embs.shape}")
qs, gt = qs[:q_embs.shape[0]], gt[:q_embs.shape[0]]

gt_idx = np.array([vids.index(g) for g in gt])

# Method 1: pure cosine against video_embs
sim = q_embs @ vembs.T                  # (Nq, 1000)
order = np.argsort(-sim, axis=1)
r = np.array([int(np.where(order[i] == gt_idx[i])[0][0]) for i in range(len(qs))])
print(f"Cosine (video_embs):     R@1={np.mean(r==0)*100:.1f}%  R@5={np.mean(r<5)*100:.1f}%  "
      f"R@10={np.mean(r<10)*100:.1f}%  MeanR={r.mean()+1:.1f}")

# Method 2: max over protos
vid_score = np.full((len(qs), len(vids)), -1e9, dtype=np.float32)
for j in range(len(vids)):
    K = int(pcounts[j])
    if K > 0:
        p = protos[j, :K]
        s = q_embs @ p.T   # (Nq, K)
        vid_score[:, j] = s.max(axis=1)
order = np.argsort(-vid_score, axis=1)
r = np.array([int(np.where(order[i] == gt_idx[i])[0][0]) for i in range(len(qs))])
print(f"MaxOverProtos:           R@1={np.mean(r==0)*100:.1f}%  R@5={np.mean(r<5)*100:.1f}%  "
      f"R@10={np.mean(r<10)*100:.1f}%  MeanR={r.mean()+1:.1f}")

# Method 3: NNN+QAMP exactly like CNPR gaps_poc.py
def score_qamp(q, p, pc, tau):
    if pc <= 0: return -np.inf
    sims = p[:pc] @ q
    z = sims / max(tau, 1e-6)
    z = z - z.max()
    w = np.exp(z); w /= w.sum() + 1e-12
    return float((w * sims).sum())

def rerank_nnn_qamp(base, qs, ps, pc, topm, alpha, tau):
    nq, nv = base.shape
    vid_mu = base.mean(0); vid_std = base.std(0) + 1e-8
    out = base.copy()
    for i in range(nq):
        row = base[i]
        cand = np.argpartition(-row, topm-1)[:topm]
        nnn = (row[cand] - vid_mu[cand]) / vid_std[cand]
        qa = np.array([score_qamp(qs[i], ps[j], int(pc[j]), tau) for j in cand])
        nnn_z = (nnn - nnn.mean()) / (nnn.std() + 1e-8)
        qa_z  = (qa - qa.mean())  / (qa.std() + 1e-8)
        fused = (1-alpha) * nnn_z + alpha * qa_z
        non_max = np.partition(-row, topm)[topm]
        fused = fused - fused.min() + non_max + 1e-6
        for ci, j in enumerate(cand):
            out[i, j] = fused[ci]
    return out

s = rerank_nnn_qamp(vid_score, q_embs, protos, pcounts, 50, 0.5, 0.02)
order = np.argsort(-s, axis=1)
r = np.array([int(np.where(order[i] == gt_idx[i])[0][0]) for i in range(len(qs))])
print(f"NNN+QAMP(0.5,0.02)[CNPR]:R@1={np.mean(r==0)*100:.1f}%  R@5={np.mean(r<5)*100:.1f}%  "
      f"R@10={np.mean(r<10)*100:.1f}%  MeanR={r.mean()+1:.1f}")
