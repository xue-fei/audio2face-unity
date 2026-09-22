# -*- coding: utf-8 -*-
"""决定性实验：真实语音 vs 静音，模型输出是否变化。

对比：静音 window 和真实 Mark 语音 window 喂进同一 ONNX，
看 prediction 的 skin 段（中心帧）是否明显变化。
"""
import numpy as np
import onnxruntime as ort
import wave
import os

MODEL = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0\network.onnx"
WAV = r"G:\MyProject\audio2face-unity\Audio2Face-3D-Samples\example_audio\Mark_neutral.wav"

SKIN_SIZE = 72006
TONGUE_SIZE = 16806
JAW_SIZE = 15
EYES_SIZE = 4
TOTAL_DIM = SKIN_SIZE + TONGUE_SIZE + JAW_SIZE + EYES_SIZE  # 88831
FRAMES = 60
CENTER = 30
LEFT = 15
NUM_DIFF_STEPS = 2
GRU_LAYERS = 2
GRU_DIM = 256
BUF_LEN = 16000
INPUT_STRENGTH = 1.3  # Mark stylization


def load_wav_mono_16k(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise RuntimeError(f"unsupported sampwidth {sw}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    if sr != 16000:
        # 简单线性重采样
        ratio = 16000 / sr
        out_n = int(len(data) * ratio)
        idx = np.linspace(0, len(data) - 1, out_n)
        data = np.interp(idx, np.arange(len(data)), data)
    return data.astype(np.float32)


def run_once(sess, window, identity, emotion, latents, noise):
    inp = {
        "window": window.reshape(1, BUF_LEN),
        "identity": identity.reshape(1, 3),
        "emotion": emotion.reshape(1, CENTER, 10),
        "input_latents": latents.reshape(GRU_LAYERS, NUM_DIFF_STEPS, 1, GRU_DIM),
        "noise": noise.reshape(1, NUM_DIFF_STEPS + 1, FRAMES, TOTAL_DIM),
    }
    out = sess.run(["prediction", "output_latents"], inp)
    return out[0].reshape(FRAMES, TOTAL_DIM), out[1]


def skin_stats(pred):
    # 中心 30 帧的 skin 段 RMS / max
    skin = pred[LEFT:LEFT + CENTER, :SKIN_SIZE]
    rms = float(np.sqrt(np.mean(skin ** 2)))
    mx = float(np.max(np.abs(skin)))
    return rms, mx


def main():
    print("=== 真实语音 vs 静音 模型响应实验 ===")
    so = ort.SessionOptions()
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    print("input names:", sess.get_inputs()[0].name, [i.shape for i in sess.get_inputs()])

    rng = np.random.RandomState(1234)
    identity = np.zeros(3, np.float32); identity[2] = 1.0  # Mark
    emotion = np.zeros((CENTER, 10), np.float32)

    audio = load_wav_mono_16k(WAV)
    print(f"音频: {WAV}")
    print(f"  长度={len(audio)} 采样  RMS={float(np.sqrt(np.mean(audio**2))):.4f}")

    # 取音频中间一段 16000 样本作为窗口
    if len(audio) < BUF_LEN:
        window = np.zeros(BUF_LEN, np.float32)
        window[:len(audio)] = audio
    else:
        mid = len(audio) // 2
        window = audio[mid:mid + BUF_LEN]
    window = (window * INPUT_STRENGTH).astype(np.float32)

    # 静音窗口
    silence = np.zeros(BUF_LEN, np.float32)

    # 用同一个噪声、同一个 GRU 初始状态（全 0），跑静音和语音
    noise = rng.standard_normal((NUM_DIFF_STEPS + 1) * FRAMES * TOTAL_DIM).astype(np.float32)
    latents0 = np.zeros((GRU_LAYERS * NUM_DIFF_STEPS * GRU_DIM), np.float32)

    print("\n--- 静音 ---")
    pred_sil, lat_sil = run_once(sess, silence, identity, emotion, latents0, noise)
    rms_s, mx_s = skin_stats(pred_sil)
    print(f"  中心帧 skin RMS={rms_s:.4f} max={mx_s:.4f}")

    print("--- 真实语音 (Mark_neutral, ×1.3) ---")
    pred_sp, lat_sp = run_once(sess, window, identity, emotion, latents0, noise)
    rms_p, mx_p = skin_stats(pred_sp)
    print(f"  中心帧 skin RMS={rms_p:.4f} max={mx_p:.4f}")

    diff = pred_sp - pred_sil
    print(f"\n--- 差异 ---")
    print(f"  中心帧 skin 段 diff RMS={float(np.sqrt(np.mean(diff[LEFT:LEFT+CENTER,:SKIN_SIZE]**2))):.4f}")
    print(f"  prediction 全量 diff RMS={float(np.sqrt(np.mean(diff**2))):.4f}")

    rel = rms_p / rms_s if rms_s > 0 else 0
    print(f"\n结论: 语音/静音 skin RMS 比 = {rel:.4f}  (应该明显 >1 才对)")


if __name__ == "__main__":
    main()
