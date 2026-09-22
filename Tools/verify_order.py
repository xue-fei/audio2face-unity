# -*- coding: utf-8 -*-
"""验证：JSON 数组顺序 vs npz poseNames 顺序，确认 remap 是否必要。"""
import numpy as np
import json

NPZ = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0\bs_skin_Mark.npz"
JSON = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0\bs_skin_config_Mark.json"

d = np.load(NPZ, allow_pickle=True)
names = [str(n) for n in d['poseNames']]
names = names[1:]  # 去掉 neutral
print("npz poseNames (去掉 neutral, 共 %d):" % len(names))
for i, n in enumerate(names):
    print(f"  {i:2d}: {n}")

with open(JSON, encoding='utf-8') as f:
    cfg = json.load(f)['blendshape_params']

active = cfg['bsSolveActivePoses']
print(f"\nJSON bsSolveActivePoses 长度={len(active)}")
print("\n逐位对照（JSON 值 → npz 顺序的 pose 名）:")
for i in range(min(len(names), len(active))):
    flag = "  <-- active=0" if active[i] == 0 else ""
    print(f"  索引{i:2d} {names[i]:20s} = {active[i]}{flag}")

# 对照官方 stylization active_poses（jawLeft/jawRight/mouthLeft/mouthRight 应为 1）
print("\n=== 结论 ===")
print("如果 JSON 数组就是 npz 顺序，那上面每个 active 值应对应正确的 pose 名，无需 remap")
