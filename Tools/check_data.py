import numpy as np, os

base = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
bs_npz = os.path.join(base, "bs_skin_Mark.npz")
md_npz = os.path.join(base, "model_data_Mark.npz")

md = np.load(md_npz, allow_pickle=True)
bs = np.load(bs_npz, allow_pickle=True)

neutral_skin = md["neutral_skin"].astype(np.float32)
bs_neutral = bs["neutral"].astype(np.float32)

print("neutral_skin:", neutral_skin.shape, "dtype", neutral_skin.dtype)
print("bs_neutral:", bs_neutral.shape, "dtype", bs_neutral.dtype)

diff = np.abs(neutral_skin - bs_neutral)
print(f"\ndiff 统计: max={diff.max():.6f} mean={diff.mean():.6f} 中位数={np.median(diff):.6f}")
print(f"diff > 0.1 的顶点数: {(diff > 0.1).sum()} / {len(diff)}")
print(f"diff > 0.5 的顶点数: {(diff > 0.5).sum()} / {len(diff)}")

# 看前几个顶点的坐标对比
print("\n前 5 个顶点 (neutral_skin vs bs_neutral):")
for v in range(5):
    i3 = v * 3
    a = neutral_skin[i3:i3+3]
    b = bs_neutral[i3:i3+3]
    print(f"  v{v}: skin={a}  bs={b}  diff={np.abs(a-b).max():.4f}")

# frontalMask
fm = bs["frontalMask"]
print(f"\nfrontalMask: shape={fm.shape} dtype={fm.dtype}")
print("frontalMask 前 20:", fm[:20])
print("frontalMask max:", fm.max(), "min:", fm.min())

# bs 里有哪些 keys
print("\nbs_skin keys:", list(bs.keys()))
# pose 数组的 shape
print("\nneutral 长度 =", len(bs_neutral))
print("pose eyeBlinkLeft 长度 =", len(bs["eyeBlinkLeft"]))
