# -*- coding: utf-8 -*-
"""精确验证：模型 delta 投影到 jawOpen 等 blendshape 的权重量级。

复现求解器核心：target = delta (skin段，掩码后)，b = D^T·target，w ≈ b / ||D||^2
看 jawOpen 的权重应该是多大。
"""
import numpy as np
import onnxruntime as ort
import wave

MODEL = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0\network.onnx"
NPZ = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0\bs_skin_Mark.npz"
WAV = r"G:\MyProject\audio2face-unity\Audio2Face-3D-Samples\example_audio\Mark_neutral.wav"

SKIN_SIZE = 72006
TOTAL_DIM = SKIN_SIZE + 16806 + 15 + 4
FRAMES = 60
CENTER = 30
LEFT = 15
BUF_LEN = 16000


def load_wav(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate(); nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1: data = data.reshape(-1, nch).mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(data)-1, int(len(data)*16000/sr))
        data = np.interp(idx, np.arange(len(data)), data)
    return data.astype(np.float32)


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(n) for n in d['poseNames']]
    print("npz keys (pose 数组):", [k for k in d.files if k not in ('poseNames', 'neutral', 'frontalMask')][:10], "...")

    # 关键 pose 模板 delta RMS
    print("\n--- blendshape 模板 delta RMS ---")
    for name in ['jawOpen', 'mouthClose', 'mouthSmileLeft', 'eyeBlinkLeft', 'browDownLeft', 'cheekPuff']:
        if name in d.files:
            pose = d[name].astype(np.float32)
            rms = float(np.sqrt(np.mean(pose**2)))
            mx = float(np.max(np.abs(pose)))
            print(f"  {name}: RMS={rms:.4f} max={mx:.4f}")

    # 加载 frontalMask
    frontal = d['frontalMask'].astype(np.int64) if 'frontalMask' in d.files else None
    if frontal is not None:
        mask_pos = np.concatenate([[3*v, 3*v+1, 3*v+2] for v in frontal])
        print(f"\nfrontalMask: {len(frontal)} 顶点 -> {len(mask_pos)} 分量")
    else:
        mask_pos = np.arange(SKIN_SIZE)

    # 模型输出
    so = ort.SessionOptions()
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(1234)
    identity = np.zeros(3, np.float32); identity[2] = 1.0
    emotion = np.zeros((CENTER, 10), np.float32)
    audio = load_wav(WAV)
    mid = len(audio)//2
    window = (audio[mid:mid+BUF_LEN] * 1.3).astype(np.float32)
    noise = rng.standard_normal(3*FRAMES*TOTAL_DIM).astype(np.float32)
    latents = np.zeros(1024, np.float32)
    inp = {
        "window": window.reshape(1, BUF_LEN),
        "identity": identity.reshape(1, 3),
        "emotion": emotion.reshape(1, CENTER, 10),
        "input_latents": latents.reshape(2, 2, 1, 256),
        "noise": noise.reshape(1, 3, FRAMES, TOTAL_DIM),
    }
    pred = sess.run(["prediction"], inp)[0].reshape(FRAMES, TOTAL_DIM)

    # 中心一帧的 skin delta（语音）
    skin = pred[LEFT + CENTER//2, :SKIN_SIZE].astype(np.float32)
    skin_mask = skin[mask_pos]

    # 静音对照
    sil_inp = dict(inp)
    sil_inp["window"] = np.zeros((1, BUF_LEN), np.float32)
    pred_sil = sess.run(["prediction"], sil_inp)[0].reshape(FRAMES, TOTAL_DIM)
    skin_sil = pred_sil[LEFT + CENTER//2, :SKIN_SIZE].astype(np.float32)[mask_pos]

    print(f"\n--- 模型 delta（中心帧，掩码后 {len(mask_pos)} 分量）---")
    print(f"  语音 skin delta RMS = {float(np.sqrt(np.mean(skin_mask**2))):.4f}")
    print(f"  静音 skin delta RMS = {float(np.sqrt(np.mean(skin_sil**2))):.4f}")
    diff = skin_mask - skin_sil
    print(f"  语音-静音 diff RMS = {float(np.sqrt(np.mean(diff**2))):.4f}")

    # 简单投影：w ≈ (D^T·target) / (||D||^2) 对每个 pose 单独算（忽略耦合）
    print("\n--- 简单投影权重 w = dot(pose, delta) / dot(pose, pose) ---")
    frontal_v = frontal.astype(np.int64)
    for name in ['jawOpen', 'mouthClose', 'mouthSmileLeft', 'mouthFunnel', 'eyeBlinkLeft', 'cheekPuff']:
        if name not in d.files:
            continue
        pose_full = d[name].astype(np.float32)  # (24002, 3)
        pose = pose_full[frontal_v].reshape(-1).astype(np.float32)
        denom = float(np.dot(pose, pose))
        if denom < 1e-12:
            continue
        w_voice = float(np.dot(pose, skin_mask) / denom)
        w_sil = float(np.dot(pose, skin_sil) / denom)
        print(f"  {name}: 语音 w={w_voice:+.4f}  静音 w={w_sil:+.4f}  差值={w_voice-w_sil:+.4f}")


if __name__ == "__main__":
    main()
