import numpy as np, json, os
from scipy.optimize import lsq_linear

base = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
bs_npz = os.path.join(base, "bs_skin_Mark.npz")
cfg_path = os.path.join(base, "bs_skin_config_Mark.json")

bs = np.load(bs_npz, allow_pickle=True)
names = [n.decode() if isinstance(n, bytes) else n for n in bs["poseNames"]]
pose_names = names[1:]

cfg = json.load(open(cfg_path, encoding="utf-8"))["blendshape_params"]
active = cfg["bsSolveActivePoses"]; symmetry = cfg["bsSolveSymmetryPoses"]
L1 = cfg["strengthL1regularization"]; L2 = cfg["strengthL2regularization"]
temp_reg = cfg["strengthTemporalSmoothing"]; sym_reg = cfg["strengthSymmetry"]
template_bb = cfg["templateBBSize"]

fm = bs["frontalMask"].astype(np.int64)
mask_positions = np.concatenate([fm*3, fm*3+1, fm*3+2]); mask_positions.sort()

bs_neutral = bs["neutral"].astype(np.float64).reshape(-1)
bsn = bs_neutral.reshape(-1,3)
target_bb = float(np.linalg.norm(bsn.max(axis=0) - bsn.min(axis=0)))
sf = (target_bb/template_bb)**2

active_idx = [i for i in range(52) if active[i] != 0]
K = len(active_idx); P = len(mask_positions)
D = np.zeros((K,P), np.float64)
for j, ai in enumerate(active_idx):
    D[j] = bs[pose_names[ai]].astype(np.float64).reshape(-1)[mask_positions]

ATA = D @ D.T
AMat = ATA + L1*L1*(0.25*sf)*np.ones((K,K)) + L2*(10.0*sf)*np.eye(K) + temp_reg*(100.0*sf)*np.eye(K)
rows = []
for a in range(K):
    aid = symmetry[active_idx[a]]
    if aid >= 0:
        for b in range(K):
            if b != a and symmetry[active_idx[b]] == aid:
                row = np.zeros(K); row[a]=1; row[b]=-1
                rows.append(row); break
if rows:
    SM = np.array(rows); AMat += sym_reg*(10.0*sf)*(SM.T@SM)

def solve(td, prev):
    b = D @ td + temp_reg*sf*prev
    return lsq_linear(AMat, b, bounds=(0,1), tol=1e-10).x

name_to_active = {pose_names[ai]: j for j, ai in enumerate(active_idx)}

# 自检1：把 jawOpen 模板 delta 喂进去（无 temporal），应解出 jawOpen ≈ 1
print("=== 自检1: 单个 pose delta 还原 ===")
for test_name in ["jawOpen", "mouthClose", "mouthSmileLeft"]:
    if test_name not in name_to_active:
        print(f"  {test_name}: 不在活跃列表"); continue
    j = name_to_active[test_name]
    td = D[j].copy()  # 该 pose 的 delta
    x = solve(td, np.zeros(K))
    # 找最大权重的 pose
    top = np.argsort(x)[::-1][:3]
    print(f"  {test_name}: 最大权重 pose 及值:")
    for idx in top:
        print(f"      {pose_names[active_idx[idx]]} = {x[idx]:.4f}")

# 自检2：全零 delta → 应解出全 0
print("=== 自检2: 全零 delta ===")
x = solve(np.zeros(P), np.zeros(K))
print(f"  最大权重 = {x.max():.6f} (应接近 0)")

# 自检3：jawOpen delta 的 RMS 归一化后投影（简单投影 vs BVLS）
print("=== 自检3: 简单投影 vs BVLS (jawOpen) ===")
j = name_to_active["jawOpen"]
td = D[j].copy()
simple = D @ td / (np.diag(D @ D.T))
print(f"  简单投影 jawOpen = {simple[j]:.4f}")
x = solve(td, np.zeros(K))
print(f"  BVLS jawOpen = {x[j]:.4f}")
