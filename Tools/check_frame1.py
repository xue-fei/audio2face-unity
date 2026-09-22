"""打印 Python 模型第二帧的前 20 个值，用于和 Unity 的 pred[frame1 0..19] 对比。"""
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

# 打印第一帧和第二帧的前 20 个值
print(f"Python pred shape: {pred.shape}")
print(f"Python pred[0, 0, 0..19] (frame0):")
for i in range(20):
    print(f"  [{i}] = {pred[0, 0, i]:.9f}")
print(f"\nPython pred[0, 1, 0..19] (frame1):")
for i in range(20):
    print(f"  [{i}] = {pred[0, 1, i]:.9f}")
print(f"\n帧0 vs 帧1 差异: {np.abs(pred[0,0,:20] - pred[0,1,:20]).max():.9f}")
