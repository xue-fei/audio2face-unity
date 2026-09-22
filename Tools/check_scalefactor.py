import numpy as np, os

base = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
npz_path = os.path.join(base, "bs_skin_Mark.npz")

d = np.load(npz_path, allow_pickle=True)
neutral = d["neutral"].astype(np.float32)
names = [n.decode() if isinstance(n, bytes) else n for n in d["poseNames"]]

num_vertex = len(neutral) // 3
neu = neutral.reshape(-1, 3)
minXYZ = neu.min(axis=0)
maxXYZ = neu.max(axis=0)
targetBBSize = float(np.linalg.norm(maxXYZ - minXYZ))

templateBBSize = 53.45050282121686
scaleFactor = (targetBBSize / templateBBSize) ** 2

print(f"neutral 顶点数 = {num_vertex}")
print(f"neutral 包围盒 min = {minXYZ}")
print(f"neutral 包围盒 max = {maxXYZ}")
print(f"targetBBSize = {targetBBSize:.4f}")
print(f"templateBBSize = {templateBBSize:.4f}")
print(f"scaleFactor = {scaleFactor:.4f}")
print()

# 各模板 delta 的 RMS
print("模板 delta RMS (前若干 pose):")
for i, n in enumerate(names):
    if n == "neutral":
        continue
    if n not in d.files:
        continue
    delta = d[n].astype(np.float32)
    rms = float(np.sqrt(np.mean(delta ** 2)))
    print(f"  [{i}] {n:20s} RMS={rms:.5f}")
