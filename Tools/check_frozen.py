# -*- coding: utf-8 -*-
"""验证：固定 window 反复推理，GRU 是否收敛到小输出（复现 Unity #7+ 衰减）。

对比：滑动窗口（正常流式） vs 冻结窗口（Unity 疑似行为）。
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

    def make_session():
        return sess, np.zeros(1024, np.float32)

    def infer(sess, latents, window):
        inp = {
            "window": window.reshape(1, BUF_LEN),
            "identity": identity.reshape(1, 3),
            "emotion": emotion.reshape(1, CENTER, 10),
            "input_latents": latents.reshape(2, 2, 1, 256),
            "noise": noise.reshape(1, 3, FRAMES, TOTAL_DIM),
        }
        pred, lat = sess.run(["prediction", "output_latents"], inp)
        return pred.reshape(FRAMES, TOTAL_DIM), lat.reshape(-1)

    def jaw_w(pred):
        skin = pred[LEFT + CENTER//2, :SKIN_SIZE].astype(np.float32)[mask_pos]
        return float(np.dot(jaw, skin) / denom)

    # === 场景 A：滑动窗口流式 ===
    print("=== 场景 A：滑动窗口流式（正常） ===")
    sess_a, lat_a = make_session()
    for _ in range(3):
        _, lat_a = infer(sess_a, lat_a, np.zeros(BUF_LEN, np.float32))
    for k in range(6):
        start = k * STRIDE
        w = (audio[start:start+BUF_LEN] * 1.3).astype(np.float32)
        pred, lat_a = infer(sess_a, lat_a, w)
        print(f"  窗口#{k} (滑动): jawOpen={jaw_w(pred):+.4f}")

    # === 场景 B：冻结窗口反复推理 ===
    print("\n=== 场景 B：冻结同一窗口反复推理（Unity 疑似） ===")
    sess_b, lat_b = make_session()
    for _ in range(3):
        _, lat_b = infer(sess_b, lat_b, np.zeros(BUF_LEN, np.float32))
    frozen = (audio[16000:32000] * 1.3).astype(np.float32)
    for k in range(10):
        pred, lat_b = infer(sess_b, lat_b, frozen)
        print(f"  帧#{k} (冻结): jawOpen={jaw_w(pred):+.4f}  gruRMS={float(np.sqrt(np.mean(lat_b**2))):.4f}")


if __name__ == "__main__":
    main()
