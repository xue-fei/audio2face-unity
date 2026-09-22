import numpy as np, os

base = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
bs_npz = os.path.join(base, "bs_skin_Mark.npz")
md_npz = os.path.join(base, "model_data_Mark.npz")

bs = np.load(bs_npz, allow_pickle=True)
md = np.load(md_npz, allow_pickle=True)

for name, d in [("bs", bs), ("md", md)]:
    print(f"=== {name} ===")
    for k in d.keys():
        arr = d[k]
        if isinstance(arr, np.ndarray):
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype} size={arr.size}")

print("\n=== 关键数值 ===")
print("bs neutral shape:", bs["neutral"].shape, "size:", bs["neutral"].size)
print("bs neutral 展平长度:", bs["neutral"].reshape(-1).shape[0])
print("md neutral_skin shape:", md["neutral_skin"].shape, "size:", md["neutral_skin"].size)
print("md neutral_skin 展平长度:", md["neutral_skin"].reshape(-1).shape[0])
print("frontalMask shape:", bs["frontalMask"].shape, "max:", bs["frontalMask"].max(), "min:", bs["frontalMask"].min())
