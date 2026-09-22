import numpy as np, onnxruntime as ort, json, os, wave

base = r"G:\MyProject\audio2face-unity\Assets\StreamingAssets\nvidiaAudio2Face-3D-v3.0"
onnx_path = os.path.join(base, "network.onnx")
bs_npz = os.path.join(base, "bs_skin_Mark.npz")
cfg_path = os.path.join(base, "bs_skin_config_Mark.json")
md_npz = os.path.join(base, "model_data_Mark.npz")
wav_path = r"G:\MyProject\audio2face-unity\Audio2Face-3D-Samples\example_audio\Mark_neutral.wav"

md = np.load(md_npz, allow_pickle=True)
bs = np.load(bs_npz, allow_pickle=True)

neutral_skin = md["neutral_skin"].astype(np.float64).reshape(-1)   # (72006,)
eye_close = md["eye_close_pose_delta"].astype(np.float64).reshape(-1)
lip_open = md["lip_open_pose_delta"].astype(np.float64).reshape(-1)
bs_neutral = bs["neutral"].astype(np.float64).reshape(-1)

print("neutral_skin 长度:", len(neutral_skin))
print("bs_neutral 长度:", len(bs_neutral))
diff = np.abs(neutral_skin - bs_neutral).max()
print(f"neutral_skin vs bs_neutral 最大差 = {diff:.6f}")

names = [n.decode() if isinstance(n, bytes) else n for n in bs["poseNames"]]
pose_names = names[1:]  # 去掉 neutral

# frontalMask → 分量索引
fm = bs["frontalMask"].astype(np.int64)
mask_positions = np.concatenate([fm*3, fm*3+1, fm*3+2])
mask_positions.sort()
print("frontalMask 顶点数:", len(fm), "分量数:", len(mask_positions))

# wav
with wave.open(wav_path, "rb") as w:
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    raw = w.readframes(n)
if sw == 2:
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
else:
    pcm = np.frombuffer(raw, dtype=np.float32)
if ch > 1:
    pcm = pcm.reshape(-1, ch)[:, 0]
if sr != 16000:
    from scipy.signal import resample_poly
    pcm = resample_poly(pcm, 16000, sr).astype(np.float32)
print(f"pcm: {len(pcm)} samples ({len(pcm)/16000:.2f}s) RMS={np.sqrt(np.mean(pcm**2)):.5f}")

# noise (xorshift128 + Box-Muller, int 版)
def fill_gaussian(n, seed):
    def u32(v): return v & 0xFFFFFFFF
    x = u32(seed * 2654435761)
    y, z, w = 362437, 1013904223, 1664525
    if x == 0: x = 0x9E3779B9
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, 2):
        t = u32(x ^ u32(x << 11))
        x, y, z = y, z, w
        w = u32(u32(w ^ (w >> 19)) ^ u32(t ^ (t >> 8)))
        u1 = (w + 0.5) / 4294967296.0
        t = u32(x ^ u32(x << 11))
        x, y, z = y, z, w
        w = u32(u32(w ^ (w >> 19)) ^ u32(t ^ (t >> 8)))
        u2 = (w + 0.5) / 4294967296.0
        mag = np.sqrt(-2.0 * np.log(u1))
        out[i] = mag * np.cos(2*np.pi*u2)
        if i+1 < n: out[i+1] = mag * np.sin(2*np.pi*u2)
    return out

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
BUFFER, STRIDE, NOISE_STEPS, FRAMES, DIM = 16000, 8000, 3, 60, 88831
identity = np.zeros((1,3), np.float32); identity[0,2] = 1.0  # Mark
emotion = np.zeros((1,30,10), np.float32)
latents = np.zeros((2,2,1,256), np.float32)
noise = fill_gaussian(NOISE_STEPS*FRAMES*DIM, 1234).reshape(1, NOISE_STEPS, FRAMES, DIM)
INPUT_STRENGTH = 1.3

def run_inference(window, lat):
    win = (window * INPUT_STRENGTH).reshape(1,BUFFER).astype(np.float32)
    feeds = {"window": win, "identity": identity, "emotion": emotion,
             "input_latents": lat, "noise": noise}
    return sess.run(None, feeds)

# animator params
skin_strength, lower_strength, upper_strength = 1.1, 1.4, 1.0
lower_smoothing, upper_smoothing = 0.0023, 0.001
face_mask_level, face_mask_softness = 0.6, 0.0085
eyelid_open_offset, lip_open_offset = 0.06, -0.03
blink_offset, blink_strength = 0.0, 1.0
dt = 1.0/60.0
eye_coef = -eyelid_open_offset + blink_offset*blink_strength
lip_coef = lip_open_offset

ys = neutral_skin[1::3]
minY, maxY = ys.min(), ys.max()
t = (ys - minY)/(maxY - minY)
mask_lower = 1.0/(1.0 + np.exp(-(face_mask_level - t)/face_mask_softness))
mask_lower_comp = np.repeat(mask_lower, 3)  # 顶点 -> 分量 (72006,)
aLo = 1.0 - 0.5**(dt/lower_smoothing)
aUp = 1.0 - 0.5**(dt/upper_smoothing)

# 求解器
cfg = json.load(open(cfg_path, encoding="utf-8"))["blendshape_params"]
active = cfg["bsSolveActivePoses"]; symmetry = cfg["bsSolveSymmetryPoses"]
L1 = cfg["strengthL1regularization"]; L2 = cfg["strengthL2regularization"]
temp_reg = cfg["strengthTemporalSmoothing"]; sym_reg = cfg["strengthSymmetry"]
template_bb = cfg["templateBBSize"]

bsn = bs_neutral.reshape(-1,3)
target_bb = float(np.linalg.norm(bsn.max(axis=0) - bsn.min(axis=0)))
sf = (target_bb/template_bb)**2
print(f"scaleFactor = {sf:.4f}")

active_idx = [i for i in range(52) if active[i] != 0]
K = len(active_idx); P = len(mask_positions)
D = np.zeros((K,P), np.float64)
for j, ai in enumerate(active_idx):
    D[j] = bs[pose_names[ai]].astype(np.float64).reshape(-1)[mask_positions]

ATA = D @ D.T
AMat = ATA + L1*L1*(0.25*sf)*np.ones((K,K)) + L2*(10.0*sf)*np.eye(K) + temp_reg*(100.0*sf)*np.eye(K)
rows = []
for a in range(K):
    aid = symmetry[active_idx[a]]
    if aid >= 0:
        for b in range(K):
            if b != a and symmetry[active_idx[b]] == aid:
                row = np.zeros(K); row[a]=1; row[b]=-1
                rows.append(row); break
if rows:
    SM = np.array(rows); AMat += sym_reg*(10.0*sf)*(SM.T@SM)

neutral_masked = bs_neutral[mask_positions]
from scipy.optimize import lsq_linear

def solve_bvls(td, prev):
    b = D @ td + temp_reg*sf*prev
    res = lsq_linear(AMat, b, bounds=(0,1), tol=1e-10)
    return res.x

jaw_open_idx = pose_names.index("jawOpen")
mouth_close_idx = pose_names.index("mouthClose")
active_map = {pose_names[ai]: j for j, ai in enumerate(active_idx)}

left_pad = right_pad = 16000
audio = np.concatenate([np.zeros(left_pad,np.float32), pcm, np.zeros(right_pad,np.float32)])

lo_arr = up_arr = None
prev = np.zeros(K)
jaw_seq, mc_seq = [], []
start = 0
while start + BUFFER <= len(audio):
    window = audio[start:start+BUFFER]
    pred, lat = run_inference(window, latents)
    latents = lat
    pred = pred[0]
    for f in range(15, 45):
        delta = pred[f, 0:72006].astype(np.float64)
        p = skin_strength*delta + eye_close*eye_coef + lip_open*lip_coef
        if lo_arr is None:
            lo_arr = np.tile(p, 3).copy(); up_arr = np.tile(p, 3).copy()
        lo_arr[0:72006] = p
        lo_arr[72006:144012] += (lo_arr[0:72006] - lo_arr[72006:144012])*aLo
        lo_arr[144012:] += (lo_arr[72006:144012] - lo_arr[144012:])*aLo
        up_arr[0:72006] = p
        up_arr[72006:144012] += (up_arr[0:72006] - up_arr[72006:144012])*aUp
        up_arr[144012:] += (up_arr[72006:144012] - up_arr[144012:])*aUp
        l2 = lo_arr[144012:]; u2 = up_arr[144012:]
        abs_pos = neutral_skin + u2*upper_strength*(1-mask_lower_comp) + l2*lower_strength*mask_lower_comp
        td = abs_pos[mask_positions] - neutral_masked
        x = solve_bvls(td, prev)
        prev = x.copy()
        full = np.zeros(52)
        for j, ai in enumerate(active_idx): full[ai] = x[j]
        jaw_seq.append(full[jaw_open_idx]); mc_seq.append(full[mouth_close_idx])
    start += STRIDE

print(f"\n总中心帧 = {len(jaw_seq)}")
print("jawOpen:", " ".join(f"{v:.3f}" for v in jaw_seq))
print(f"jawOpen max={max(jaw_seq):.4f} min={min(jaw_seq):.4f}")
print("mouthClose:", " ".join(f"{v:.3f}" for v in mc_seq))
print(f"mouthClose max={max(mc_seq):.4f}")
