import json, numpy as np, os

base = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
npz_path = os.path.join(base, "bs_skin_Mark.npz")
cfg_path = os.path.join(base, "bs_skin_config_Mark.json")

d = np.load(npz_path, allow_pickle=True)
names = list(d["poseNames"])
print("npz poseNames 数量:", len(names))
print("npz poseNames 完整顺序:")
for i, n in enumerate(names):
    print(f"  [{i}] {n}")

cfg = json.load(open(cfg_path, encoding="utf-8"))
bp = cfg["blendshape_params"]
active = bp["bsSolveActivePoses"]
print("\nJSON activePoses 数量:", len(active))

print("\n逐项对照 (npz index → name → JSON active 值):")
for i, n in enumerate(names):
    av = active[i] if i < len(active) else "?"
    print(f"  [{i}] {n:20s} = {av}")

# 重点检查几个 pose
print("\n关键 pose 定位:")
for key in ["jawOpen", "mouthClose", "jawLeft", "jawRight", "mouthLeft", "mouthRight",
            "eyeBlinkLeft", "eyeWideLeft", "mouthSmileLeft", "mouthFunnel"]:
    if key in names:
        idx = names.index(key)
        print(f"  {key:16s} npz_index={idx}  JSON_active={active[idx] if idx < len(active) else '?'}")
    else:
        print(f"  {key:16s} 不在 npz poseNames")
