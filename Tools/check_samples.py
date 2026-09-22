import json, numpy as np, onnxruntime as ort

BASE = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
with open(f"{BASE}\\network_info.json") as f:
    info = json.load(f)
p = info["params"]
skin, tongue = p["skin_size"], p["tongue_size"]
total = skin + tongue + p["jaw_size"] + p["eyes_size"]
frames = p["num_frames_left_truncate"] + p["num_frames_center"] + p["num_frames_right_truncate"]
buf = info["audio_params"]["buffer_len"]
noise_steps = p["num_diffusion_steps"] + 1
identities = p["identities"]
mark_idx = identities.index("Mark")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess = ort.InferenceSession(f"{BASE}\\network.onnx", so, providers=["CPUExecutionProvider"])

rng = np.random.default_rng(42)
noise = rng.standard_normal((1, noise_steps, frames, total)).astype(np.float32)
identity = np.zeros((1, len(identities)), dtype=np.float32)
identity[0, mark_idx] = 1.0
emotion = np.zeros((1, 30, 10), dtype=np.float32)
latents = np.zeros((2, 2, 1, p["gru_latent_dim"]), dtype=np.float32)

silence = np.zeros((1, buf), dtype=np.float32)
pred, lat = sess.run(
    ["prediction", "output_latents"],
    {"window": silence, "identity": identity, "emotion": emotion,
     "input_latents": latents, "noise": noise},
)

flat = pred.flatten()
print(f"\nPython 采样值（与 Unity 对比用）:")
for idx in [100, 1000, 10000, 100000, 1000000, 10000000]:
    if idx < len(flat):
        print(f"  [{idx}] = {flat[idx]:.9f}")

# 关键：统计不同幅值区间的数量，与 Unity 的 zero/small/medium/large 对比
abs_flat = np.abs(flat)
zero_count = np.sum(abs_flat < 1e-6)
small_count = np.sum((abs_flat >= 1e-6) & (abs_flat < 0.1))
medium_count = np.sum((abs_flat >= 0.1) & (abs_flat <= 1.0))
large_count = np.sum(abs_flat > 1.0)
print(f"\nPython 分布: zero(<1e-6)={zero_count} small(1e-6~0.1)={small_count} medium(0.1~1)={medium_count} large(>1)={large_count} / {len(flat)}")
