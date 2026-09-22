# -*- coding: utf-8 -*-
"""验证：模型输出 skin delta 的 scale vs npz blendshape 模板的 scale 是否匹配。

如果两者量级差很多，求解器解出的权重就会偏小/偏大。
"""
import numpy as np
import onnxruntime as ort
import wave

MODEL = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0\network.onnx"
NPZ = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0\bs_skin_Mark.npz"
WAV = r"G:\MyProject\audio2face-unity\Audio2Face-3D-Samples\example_audio\Mark_neutral.wav"

SKIN_SIZE = 72006
TONGUE_SIZE = 16806
JAW_SIZE = 15
EYES_SIZE = 4
TOTAL_DIM = SKIN_SIZE + TONGUE_SIZE + JAW_SIZE + EYES_SIZE
FRAMES = 60
CENTER = 30
LEFT = 15
BUF_LEN = 16000


def load_wav(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate(); nch = w.getnchannels(); sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1: data = data.reshape(-1, nch).mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(data)-1, int(len(data)*16000/sr))
        data = np.interp(idx, np.arange(len(data)), data)
    return data.astype(np.float32)


def main():
    print("=== scale 匹配验证 ===")
    # 1. npz blendshape 模板
    d = np.load(NPZ, allow_pickle=True)
    names = d['poseNames']
    print("npz poseNames:", names)
    neutral = d['neutral']
    print(f"neutral RMS = {float(np.sqrt(np.mean(neutral**2))):.4f}, shape={neutral.shape}")

    # 关键 blendshape 的 delta RMS
    for name in ['jawOpen', 'mouthClose', 'mouthSmileLeft', 'eyeBlinkLeft']:
        if name in names:
            pose = d[name]
            rms = float(np.sqrt(np.mean(pose**2)))
            mx = float(np.max(np.abs(pose)))
            print(f"  {name}: delta RMS={rms:.4f} max={mx:.4f}")

    # 2. 模型输出 skin delta
    so = ort.SessionOptions()
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(1234)
    identity = np.zeros(3, np.float32); identity[2] = 1.0
    emotion = np.zeros((CENTER, 10), np.float32)
    audio = load_wav(WAV)
    mid = len(audio)//2
    window = (audio[mid:mid+BUF_LEN] * 1.3).astype(np.float32)
    if len(window) < BUF_LEN:
        w2 = np.zeros(BUF_LEN, np.float32); w2[:len(window)] = window; window = w2
    noise = rng.standard_normal(3*FRAMES*TOTAL_DIM).astype(np.float32)
    latents = np.zeros(2*2*256, np.float32)
    inp = {
        "window": window.reshape(1, BUF_LEN),
        "identity": identity.reshape(1, 3),
        "emotion": emotion.reshape(1, CENTER, 10),
        "input_latents": latents.reshape(2, 2, 1, 256),
        "noise": noise.reshape(1, 3, FRAMES, TOTAL_DIM),
    }
    pred = sess.run(["prediction"], inp)[0].reshape(FRAMES, TOTAL_DIM)
    # 中心帧 skin 段 delta
    skin = pred[LEFT:LEFT+CENTER, :SKIN_SIZE]
    print(f"\n模型输出 skin delta (中心{SKIN_SIZE//3}顶点×{CENTER}帧): RMS={float(np.sqrt(np.mean(skin**2))):.4f}")

    # 看中心某一帧（比如第 15 帧）的 skin delta 顶点位移分布
    one = skin[CENTER//2]
    print(f"  单帧 skin delta: RMS={float(np.sqrt(np.mean(one**2))):.4f} max={float(np.max(np.abs(one))):.4f}")
    # 分区域（前 1/3 顶点 = 可能是上部，后 1/3 = 下部）
    nv = SKIN_SIZE//3
    lo = one[2*nv//3*3:]  # 粗略下部
    print(f"  粗略下部 1/3: RMS={float(np.sqrt(np.mean(lo**2))):.4f}")

    # 3. 对比：模型 delta vs 模板 delta 的比值
    print("\n=== 结论 ===")
    print("如果模型输出 delta RMS 远小于/大于 blendshape 模板 delta RMS，则求解权重会偏小/偏大")


if __name__ == "__main__":
    main()
