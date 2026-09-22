import numpy as np, onnxruntime as ort, json, os, wave

base = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
onnx_path = os.path.join(base, "network.onnx")
npz_path = os.path.join(base, "bs_skin_Mark.npz")
cfg_path = os.path.join(base, "bs_skin_config_Mark.json")
wav_path = r"G:\MyProject\audio2face-unity\Audio2Face-3D-Samples\example_audio\Mark_neutral.wav"

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
print("=== 模型输入 ===")
for i in sess.get_inputs():
    print(f"  {i.name}: shape={i.shape} dtype={i.type}")
print("=== 模型输出 ===")
for o in sess.get_outputs():
    print(f"  {o.name}: shape={o.shape} dtype={o.type}")
