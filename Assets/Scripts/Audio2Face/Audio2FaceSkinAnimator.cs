using System;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// 几何后处理的统一入口。标定静态偏置时要按「求解器实际看到的那条路」来算，
    /// skin / tongue 走的是不同实现，用接口把流程统一掉。
    /// </summary>
    public interface IA2FGeometryAnimator
    {
        bool IsReady { get; }
        int Size { get; }
        void Animate(float[] deltas, int srcOffset, float[] output, int dstOffset, float dt);
    }

    /// <summary>
    /// Audio2Face SDK <c>AnimatorSkin</c> 的 C# 移植（animator_cuda.cu / animator.h）。
    ///
    /// 这一步在以前的版本里被整个漏掉了——模型输出的 prediction 直接喂给了求解器。
    /// 少了它就不只是少了两个静态 pose，上下脸分区与强度（lowerFaceStrength）也全丢了。
    ///
    /// 注意：参数<b>必须按身份从 model_config_{identity}.json 读</b>（见 Audio2FaceAnimatorConfig），
    /// 不能照抄 animator.h 的 REFL 默认值：本模型 lower_face_strength 配置里是 1.0，不是默认的 1.3。
    /// 这里的 <see cref="Params.SdkDefault"/> 只是「读不到配置」时的兜底。
    ///
    /// 完整链路（SDK AnimatorSkin::Animate）：
    /// <code>
    ///   Step1  pose = skinStrength * delta
    ///              + eyeClosePoseDelta * (-eyelidOpenOffset + blinkOffset * blinkStrength)
    ///              + lipOpenPoseDelta  * lipOpenOffset
    ///   平滑   lower = Interp(pose, lowerFaceSmoothing)
    ///          upper = Interp(pose, upperFaceSmoothing)
    ///   Step2  out   = neutral
    ///              + upper * upperFaceStrength * (1 - maskLower)
    ///              + lower * lowerFaceStrength * maskLower
    /// </code>
    /// maskLower 是按顶点 Y 做的软分割：Y 低于 faceMaskLevel 的顶点算下半脸。
    /// 输出是<b>绝对顶点坐标</b>，所以喂给求解器时 TargetIsDelta 必须设成 false。
    ///
    /// 性能要点（2026-09 优化，实测 animator 曾是后处理里最贵的一段）：
    ///   1. <b>掩码视图</b>：solver 只按 frontalMask 读 ~18% 的顶点，而插值状态是<b>逐分量独立</b>的
    ///      （l1[i] 只依赖 l1[i]/l2[i] 自身），所以完全可以只重建被掩码挑中的顶点。
    ///      走 <see cref="AnimateMasked"/> 后工作量直接降到 1/5.6。
    ///   2. <b>干掉插值第 0 层</b>：SDK 的 Interpolator degree=2 有 3 层，dataArr[0] 只是 raw 输入
    ///      缓冲，写完立刻被读回来算 l1，从来没有跨帧价值。改成寄存器变量后状态数组从 3 层降到 2 层。
    ///   3. <b>内层 3 分量手动展开</b>：mask / 插值系数 / 强度都是逐顶点的，展开后每顶点只取一次，
    ///      三条分量链互相独立，指令级并行更好。
    ///   4. alpha = 1 - 0.5^(dt/smoothing) 每帧都一样，按 (dt, smoothing) 缓存，省掉两次 Mathf.Pow。
    /// </summary>
    public sealed class Audio2FaceSkinAnimator : IA2FGeometryAnimator
    {
        [Serializable]
        public class Params
        {
            // 下面这组是 animator.h REFL_AUTO 的「代码默认值」，仅当读不到
            // model_config_{identity}.json 时才用。本模型的真实参数见 Audio2FaceAnimatorConfig。
            public float lowerFaceSmoothing = 0.0023f;
            public float upperFaceSmoothing = 0.001f;
            public float lowerFaceStrength = 1.3f;      // ⚠ 代码默认；本模型配置里是 1.0
            public float upperFaceStrength = 1f;
            public float faceMaskLevel = 0.6f;
            public float faceMaskSoftness = 0.0085f;
            public float skinStrength = 1f;
            public float blinkStrength = 1f;
            public float eyelidOpenOffset = 0.06f;
            public float lipOpenOffset = -0.03f;        // ⚠ 代码默认；本模型配置里是 0.0 / -0.02
            public float blinkOffset = 0f;

            public static Params SdkDefault => new Params();
        }

        private readonly float[] _neutral;          // P (顶点*3)
        private readonly float[] _eyeCloseDelta;    // P
        private readonly float[] _lipOpenDelta;     // P
        private readonly float[] _maskLower;        // 顶点数
        private readonly int _size;                 // P
        private readonly int _vertexCount;

        // 时域插值状态（degree=2）：只保留真正跨帧的 2 层，第 0 层（raw）用寄存器变量
        private readonly float[] _loL1, _loL2, _upL1, _upL2;

        // ── 掩码视图（frontalMask）───────────────────────────────────────────────
        // solver 只会读 _maskPositions 挑出来的那部分分量，插值又是逐分量独立的，
        // 所以可以只算掩码顶点。读仍按全量下标 (3v+c)，写按紧凑下标 (3k+c)。
        private int[] _maskVerts;        // 掩码顶点在全量网格里的下标
        private float[] _maskLowerM;     // 掩码顶点对应的 faceMask 值
        private float[] _mLoL1, _mLoL2, _mUpL1, _mUpL2;
        private int _maskVertexCount;
        private int _maskedSize;

        private bool _initialized;
        private bool _initializedMasked;

        private float _maskMinY, _maskSpan;   // faceMask 的归一化基准（必须沿用全量 neutral 的 min/max）

        // alpha 缓存：dt 与 smoothing 都没变时不必重算 Pow
        private float _alphaDt = -1f;
        private float _alphaLoSmooth = -1f;
        private float _alphaUpSmooth = -1f;
        private float _aLo = 1f, _aUp = 1f;

        private Params _params;

        public bool IsReady => _neutral != null;
        public int Size => _size;
        public float[] Neutral => _neutral;
        public Params CurrentParams => _params;

        /// <summary>掩码视图是否已启用（SetMask 成功且掩码确实比全量小）。</summary>
        public bool HasMaskView => _maskVerts != null && _maskVertexCount > 0 && _maskVertexCount < _vertexCount;

        /// <summary>最近一次 Step2 之前、平滑之后的上下脸分量，调试用。</summary>
        public float LowerStrength => _params != null ? _params.lowerFaceStrength : 1f;

        public Audio2FaceSkinAnimator(Audio2FaceModelData data, Params p = null)
        {
            if (data == null || !data.IsReady) return;

            _params = p ?? Params.SdkDefault;
            _neutral = data.NeutralSkin;
            _eyeCloseDelta = data.EyeClosePoseDelta;
            _lipOpenDelta = data.LipOpenPoseDelta;
            _size = _neutral.Length;
            _vertexCount = _size / 3;

            if (_eyeCloseDelta.Length != _size || _lipOpenDelta.Length != _size)
            {
                Debug.LogError($"[Audio2Face] model_data 尺寸不一致: neutral={_size} " +
                               $"eyeClose={_eyeCloseDelta.Length} lipOpen={_lipOpenDelta.Length}");
                _neutral = null;
                return;
            }

            _maskLower = new float[_vertexCount];
            ComputeFaceMaskLower();

            _loL1 = new float[_size];
            _loL2 = new float[_size];
            _upL1 = new float[_size];
            _upL2 = new float[_size];
        }

        public void SetParams(Params p)
        {
            if (p != null) _params = p;
        }

        /// <summary>
        /// 对应 SDK 的 AnimatorSkin::SetBlinkOffset —— 眨眼是<b>应用层每帧</b>喂进来的，
        /// SDK 不自带随机眨眼生成器。Step1 里它走
        /// <c>eyeCloseDelta * (-eyelidOpenOffset + blinkOffset * blinkStrength)</c>，
        /// 0 = 睁眼，1 = 闭合。恒为 0 会导致眼睛永远不眨（只剩 -eyelidOpenOffset 的静态睁眼偏置）。
        /// 由 <see cref="A2FBlinkGenerator"/> 按帧的音频时间戳驱动。
        /// </summary>
        public void SetBlinkOffset(float value)
        {
            if (_params == null) return;
            _params.blinkOffset = value < 0f ? 0f : (value > 1f ? 1f : value);
        }

        /// <summary>
        /// 装上 solver 的 frontalMask（<b>分量级</b>，长度 = 3 × 掩码顶点数，值 = 3v+c）。
        /// 之后 <see cref="AnimateMasked"/> 只重建这些顶点，输出紧凑布局直接喂
        /// <see cref="BlendshapeSolver.SolveMasked"/>。
        ///
        /// faceMask 的归一化基准沿用<b>全量 neutral 的 minY/maxY</b>，
        /// 否则掩码顶点自己的 Y 跨度会让 mask 值整体偏移，输出就和全量路径对不上了。
        /// </summary>
        public void SetMask(int[] compMask)
        {
            _maskVerts = null;
            _maskLowerM = null;
            _mLoL1 = _mLoL2 = _mUpL1 = _mUpL2 = null;
            _maskVertexCount = 0;
            _maskedSize = 0;
            _initializedMasked = false;

            if (compMask == null || compMask.Length == 0) return;
            if (compMask.Length % 3 != 0)
            {
                Debug.LogError($"[Audio2Face] SetMask: 掩码长度 {compMask.Length} 不是 3 的倍数");
                return;
            }

            int vm = compMask.Length / 3;
            var verts = new int[vm];
            for (int k = 0; k < vm; k++)
            {
                int c0 = compMask[3 * k];
                if (c0 % 3 != 0)
                {
                    Debug.LogError("[Audio2Face] SetMask: 掩码不是按顶点对齐的（每 3 个连续分量一组）");
                    return;
                }
                int v = c0 / 3;
                if (v < 0 || v >= _vertexCount)
                {
                    Debug.LogError($"[Audio2Face] SetMask: 顶点下标越界 {v}（顶点数 {_vertexCount}）");
                    return;
                }
                verts[k] = v;
            }

            _maskVerts = verts;
            _maskVertexCount = vm;
            _maskedSize = compMask.Length;

            float soft = Mathf.Max(1e-6f, _params.faceMaskSoftness);
            float level = _params.faceMaskLevel;
            var maskM = new float[vm];
            for (int k = 0; k < vm; k++)
            {
                float t = (_neutral[verts[k] * 3 + 1] - _maskMinY) / _maskSpan;   // 0=最低 1=最高
                maskM[k] = 1f / (1f + Mathf.Exp(-(level - t) / soft));
            }
            _maskLowerM = maskM;

            _mLoL1 = new float[_maskedSize];
            _mLoL2 = new float[_maskedSize];
            _mUpL1 = new float[_maskedSize];
            _mUpL2 = new float[_maskedSize];
        }

        /// <summary>
        /// faceMaskLower[v] = sigmoid((level - (y_v - minY) / (maxY - minY)) / softness)
        /// y_v 取 neutral 的 Y 分量（下标 1）。对应 AnimatorSkinGetYKernel +
        /// AnimatorSkinGetFaceMaskLowerKernel。
        /// </summary>
        private void ComputeFaceMaskLower()
        {
            float minY = float.MaxValue, maxY = float.MinValue;
            var y = new float[_vertexCount];
            for (int v = 0; v < _vertexCount; v++)
            {
                float val = _neutral[v * 3 + 1];
                y[v] = val;
                if (val < minY) minY = val;
                if (val > maxY) maxY = val;
            }

            float span = maxY - minY;
            if (span <= 1e-9f)
            {
                Debug.LogError("[Audio2Face] neutral 顶点 Y 跨度为零，faceMask 退化");
                for (int v = 0; v < _vertexCount; v++) _maskLower[v] = 1f;
                _maskMinY = minY;
                _maskSpan = 1f;
                return;
            }

            _maskMinY = minY;
            _maskSpan = span;

            float soft = Mathf.Max(1e-6f, _params.faceMaskSoftness);
            for (int v = 0; v < _vertexCount; v++)
            {
                float t = (y[v] - minY) / span;               // 0=最低 1=最高
                _maskLower[v] = 1f / (1f + Mathf.Exp(-(_params.faceMaskLevel - t) / soft));
            }
        }

        /// <summary>重置时域平滑状态（换一段音频、或标定时调用）。</summary>
        public void Reset()
        {
            _initialized = false;
            _initializedMasked = false;
            _alphaDt = -1f;
            if (_loL1 != null)
            {
                Array.Clear(_loL1, 0, _loL1.Length);
                Array.Clear(_loL2, 0, _loL2.Length);
                Array.Clear(_upL1, 0, _upL1.Length);
                Array.Clear(_upL2, 0, _upL2.Length);
            }
            if (_mLoL1 != null)
            {
                Array.Clear(_mLoL1, 0, _mLoL1.Length);
                Array.Clear(_mLoL2, 0, _mLoL2.Length);
                Array.Clear(_mUpL1, 0, _mUpL1.Length);
                Array.Clear(_mUpL2, 0, _mUpL2.Length);
            }
        }

        /// <summary>
        /// 跑一帧（全量顶点，输出到与 neutral 同布局的缓冲）。
        /// </summary>
        /// <param name="deltas">模型输出（prediction）</param>
        /// <param name="srcOffset">deltas 里 skin 段的起始下标</param>
        /// <param name="output">输出绝对顶点，长度至少 dstOffset + Size</param>
        /// <param name="dt">帧间隔（秒），60fps → 1/60</param>
        public void Animate(float[] deltas, int srcOffset, float[] output, int dstOffset, float dt)
        {
            if (!IsReady) return;

            float eyeCoef = -_params.eyelidOpenOffset + _params.blinkOffset * _params.blinkStrength;
            float lipCoef = _params.lipOpenOffset;
            float skinStrength = _params.skinStrength;
            UpdateAlphas(dt);

            float aLo = _aLo, aUp = _aUp;
            float lowerStrength = _params.lowerFaceStrength;
            float upperStrength = _params.upperFaceStrength;

            var neu = _neutral;
            var eye = _eyeCloseDelta;
            var lip = _lipOpenDelta;
            var msk = _maskLower;
            var lo1 = _loL1;
            var lo2 = _loL2;
            var up1 = _upL1;
            var up2 = _upL2;

            if (!_initialized)
            {
                // InterpolatorInitDataKernel：两层全部初始化成当前 raw
                for (int i = 0; i < _size; i++)
                {
                    float p = skinStrength * deltas[srcOffset + i]
                            + eye[i] * eyeCoef
                            + lip[i] * lipCoef;
                    lo1[i] = p; lo2[i] = p;
                    up1[i] = p; up2[i] = p;
                }
                _initialized = true;
            }

            // 按顶点外层、3 个分量完全展开：faceMask 每顶点只取一次，全部顺序访存
            for (int v = 0; v < _vertexCount; v++)
            {
                float m = msk[v];
                float uS = upperStrength * (1f - m);
                float lS = lowerStrength * m;

                int i0 = v * 3, i1 = i0 + 1, i2 = i0 + 2;
                int s0 = srcOffset + i0;
                int d0 = dstOffset + i0;

                // --- 分量 0 ---
                float p0 = skinStrength * deltas[s0] + eye[i0] * eyeCoef + lip[i0] * lipCoef;
                float t = lo1[i0] + (p0 - lo1[i0]) * aLo; lo1[i0] = t;
                float b = lo2[i0] + (t - lo2[i0]) * aLo; lo2[i0] = b;
                float q = up1[i0] + (p0 - up1[i0]) * aUp; up1[i0] = q;
                float e = up2[i0] + (q - up2[i0]) * aUp; up2[i0] = e;
                output[d0] = neu[i0] + e * uS + b * lS;

                // --- 分量 1 ---
                float p1 = skinStrength * deltas[s0 + 1] + eye[i1] * eyeCoef + lip[i1] * lipCoef;
                t = lo1[i1] + (p1 - lo1[i1]) * aLo; lo1[i1] = t;
                b = lo2[i1] + (t - lo2[i1]) * aLo; lo2[i1] = b;
                q = up1[i1] + (p1 - up1[i1]) * aUp; up1[i1] = q;
                e = up2[i1] + (q - up2[i1]) * aUp; up2[i1] = e;
                output[d0 + 1] = neu[i1] + e * uS + b * lS;

                // --- 分量 2 ---
                float p2 = skinStrength * deltas[s0 + 2] + eye[i2] * eyeCoef + lip[i2] * lipCoef;
                t = lo1[i2] + (p2 - lo1[i2]) * aLo; lo1[i2] = t;
                b = lo2[i2] + (t - lo2[i2]) * aLo; lo2[i2] = b;
                q = up1[i2] + (p2 - up1[i2]) * aUp; up1[i2] = q;
                e = up2[i2] + (q - up2[i2]) * aUp; up2[i2] = e;
                output[d0 + 2] = neu[i2] + e * uS + b * lS;
            }
        }

        /// <summary>
        /// 跑一帧，但<b>只重建 frontalMask 挑出来的顶点</b>，输出紧凑布局（长度 = 3 × 掩码顶点数）。
        /// 结果必须喂给 <see cref="BlendshapeSolver.SolveMasked"/>。
        ///
        /// 数值上与 <see cref="Animate"/> + 掩码取样<b>逐位一致</b>：插值状态是逐分量独立的
        /// （l1[i] 只用到 l1[i]、l2[i]），faceMask 也沿用全量 neutral 的 Y 归一化基准。
        /// </summary>
        public void AnimateMasked(float[] deltas, int srcOffset, float[] output, int dstOffset, float dt)
        {
            if (!IsReady || !HasMaskView) return;

            float eyeCoef = -_params.eyelidOpenOffset + _params.blinkOffset * _params.blinkStrength;
            float lipCoef = _params.lipOpenOffset;
            float skinStrength = _params.skinStrength;
            UpdateAlphas(dt);

            float aLo = _aLo, aUp = _aUp;
            float lowerStrength = _params.lowerFaceStrength;
            float upperStrength = _params.upperFaceStrength;

            var neu = _neutral;
            var eye = _eyeCloseDelta;
            var lip = _lipOpenDelta;
            var verts = _maskVerts;
            var msk = _maskLowerM;
            var lo1 = _mLoL1;
            var lo2 = _mLoL2;
            var up1 = _mUpL1;
            var up2 = _mUpL2;
            int vm = _maskVertexCount;

            if (!_initializedMasked)
            {
                for (int k = 0; k < vm; k++)
                {
                    int i0 = verts[k] * 3;
                    int q0 = k * 3;
                    for (int c = 0; c < 3; c++)
                    {
                        float p = skinStrength * deltas[srcOffset + i0 + c]
                                + eye[i0 + c] * eyeCoef
                                + lip[i0 + c] * lipCoef;
                        lo1[q0 + c] = p; lo2[q0 + c] = p;
                        up1[q0 + c] = p; up2[q0 + c] = p;
                    }
                }
                _initializedMasked = true;
            }

            for (int k = 0; k < vm; k++)
            {
                float m = msk[k];
                float uS = upperStrength * (1f - m);
                float lS = lowerStrength * m;

                int i0 = verts[k] * 3;      // 读：全量网格
                int q0 = k * 3;             // 写：紧凑视图
                int s0 = srcOffset + i0;
                int d0 = dstOffset + q0;

                float p0 = skinStrength * deltas[s0] + eye[i0] * eyeCoef + lip[i0] * lipCoef;
                float t = lo1[q0] + (p0 - lo1[q0]) * aLo; lo1[q0] = t;
                float b = lo2[q0] + (t - lo2[q0]) * aLo; lo2[q0] = b;
                float q = up1[q0] + (p0 - up1[q0]) * aUp; up1[q0] = q;
                float e = up2[q0] + (q - up2[q0]) * aUp; up2[q0] = e;
                output[d0] = neu[i0] + e * uS + b * lS;

                float p1 = skinStrength * deltas[s0 + 1] + eye[i0 + 1] * eyeCoef + lip[i0 + 1] * lipCoef;
                t = lo1[q0 + 1] + (p1 - lo1[q0 + 1]) * aLo; lo1[q0 + 1] = t;
                b = lo2[q0 + 1] + (t - lo2[q0 + 1]) * aLo; lo2[q0 + 1] = b;
                q = up1[q0 + 1] + (p1 - up1[q0 + 1]) * aUp; up1[q0 + 1] = q;
                e = up2[q0 + 1] + (q - up2[q0 + 1]) * aUp; up2[q0 + 1] = e;
                output[d0 + 1] = neu[i0 + 1] + e * uS + b * lS;

                float p2 = skinStrength * deltas[s0 + 2] + eye[i0 + 2] * eyeCoef + lip[i0 + 2] * lipCoef;
                t = lo1[q0 + 2] + (p2 - lo1[q0 + 2]) * aLo; lo1[q0 + 2] = t;
                b = lo2[q0 + 2] + (t - lo2[q0 + 2]) * aLo; lo2[q0 + 2] = b;
                q = up1[q0 + 2] + (p2 - up1[q0 + 2]) * aUp; up1[q0 + 2] = q;
                e = up2[q0 + 2] + (q - up2[q0 + 2]) * aUp; up2[q0 + 2] = e;
                output[d0 + 2] = neu[i0 + 2] + e * uS + b * lS;
            }
        }

        private void UpdateAlphas(float dt)
        {
            if (_params.lowerFaceSmoothing > 0f)
            {
                if (dt != _alphaDt || _params.lowerFaceSmoothing != _alphaLoSmooth)
                {
                    _aLo = 1f - Mathf.Pow(0.5f, dt / _params.lowerFaceSmoothing);
                    _alphaLoSmooth = _params.lowerFaceSmoothing;
                }
            }
            else _aLo = 1f;

            if (_params.upperFaceSmoothing > 0f)
            {
                if (dt != _alphaDt || _params.upperFaceSmoothing != _alphaUpSmooth)
                {
                    _aUp = 1f - Mathf.Pow(0.5f, dt / _params.upperFaceSmoothing);
                    _alphaUpSmooth = _params.upperFaceSmoothing;
                }
            }
            else _aUp = 1f;

            _alphaDt = dt;
        }

        /// <summary>把 prediction 的 skin 段直接当 delta 输出（不跑 animator），用于开关对比。</summary>
        public static void Passthrough(float[] deltas, int srcOffset, float[] output, int dstOffset, int size)
        {
            Array.Copy(deltas, srcOffset, output, dstOffset, size);
        }
    }

    /// <summary>
    /// SDK <c>AnimatorTongue</c>：resultPos = neutral + resultPos * strength，
    /// 再给 Y 分量加 heightOffset、Z 分量加 depthOffset（SDK 默认 1.5 / 0.2 / 0.13）。
    /// 输出同样是绝对顶点，所以 tongue 求解器的 TargetIsDelta 也要设成 false。
    /// </summary>
    public sealed class Audio2FaceTongueAnimator : IA2FGeometryAnimator
    {
        private readonly float[] _neutral;
        private readonly int _size;
        private float _strength = 1.5f;
        private float _heightOffset = 0.2f;
        private float _depthOffset = 0.13f;

        public bool IsReady => _neutral != null;
        public int Size => _size;

        public Audio2FaceTongueAnimator(Audio2FaceModelData data)
        {
            if (data == null || data.NeutralTongue == null) return;
            _neutral = data.NeutralTongue;
            _size = _neutral.Length;
        }

        /// <summary>按身份覆盖参数（来源 model_config_{identity}.json：Mark 1.5/0.2/0.13，Claire·James 1.3/0/0）。</summary>
        public void SetParams(float strength, float heightOffset, float depthOffset)
        {
            _strength = strength;
            _heightOffset = heightOffset;
            _depthOffset = depthOffset;
        }

        /// <summary>舌头没有时域平滑，dt 参数只是为了对skin 接口保持一致。</summary>
        public void Animate(float[] deltas, int srcOffset, float[] output, int dstOffset, float dt)
            => Animate(deltas, srcOffset, output, dstOffset);

        public void Animate(float[] deltas, int srcOffset, float[] output, int dstOffset)
        {
            if (!IsReady) return;
            for (int i = 0; i < _size; i++)
            {
                float v = _neutral[i] + deltas[srcOffset + i] * _strength;
                int c = i % 3;
                if (c == 1) v += _heightOffset;
                else if (c == 2) v += _depthOffset;
                output[dstOffset + i] = v;
            }
        }
    }
}
