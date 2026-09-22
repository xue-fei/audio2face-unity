# -*- coding: utf-8 -*-
"""流式复现：连续喂音频窗口，看 jawOpen 投影权重是否随 GRU 演化而衰减。

复现 Unity 的连续推理：每次用滑动窗口 + 上次 output_latents 作为 input_latents。
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
STRIDE = 8000


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
    frontal = d['frontalMask'].astype(np.int64)
    jaw = d['jawOpen'].astype(np.float32)[frontal].reshape(-1)
    denom = float(np.dot(jaw, jaw))
    mask_pos = np.concatenate([[3*v, 3*v+1, 3*v+2] for v in frontal])

    so = ort.SessionOptions()
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(1234)
    identity = np.zeros(3, np.float32); identity[2] = 1.0
    emotion = np.zeros((CENTER, 10), np.float32)
    audio = load_wav(WAV)
    noise = rng.standard_normal(3*FRAMES*TOTAL_DIM).astype(np.float32)
    latents = np.zeros(1024, np.float32)

    print(f"音频长度={len(audio)} 采样 = {len(audio)/16000:.2f}s")

    # 连续推理：滑动窗口，stride=8000
    n_windows = (len(audio) - BUF_LEN) // STRIDE
    print(f"可推理窗口数={n_windows}")

    def infer(window):
        nonlocal latents
        inp = {
            "window": window.reshape(1, BUF_LEN),
            "identity": identity.reshape(1, 3),
            "emotion": emotion.reshape(1, CENTER, 10),
            "input_latents": latents.reshape(2, 2, 1, 256),
            "noise": noise.reshape(1, 3, FRAMES, TOTAL_DIM),
        }
        pred, lat = sess.run(["prediction", "output_latents"], inp)
        latents = lat.reshape(-1)
        return pred.reshape(FRAMES, TOTAL_DIM)

    # 先静音几步（模拟标定 + 预热）
    print("\n--- 静音预热 3 步 ---")
    for i in range(3):
        infer(np.zeros(BUF_LEN, np.float32))

    print("\n--- 连续音频推理，jawOpen 投影权重 ---")
    for k in range(min(n_windows, 12)):
        start = k * STRIDE
        window = (audio[start:start+BUF_LEN] * 1.3).astype(np.float32)
        pred = infer(window)
        skin = pred[LEFT + CENTER//2, :SKIN_SIZE].astype(np.float32)[mask_pos]
        w = float(np.dot(jaw, skin) / denom)
        print(f"  窗口#{k}: jawOpen 投影={w:+.4f}")


if __name__ == "__main__":
    main()
