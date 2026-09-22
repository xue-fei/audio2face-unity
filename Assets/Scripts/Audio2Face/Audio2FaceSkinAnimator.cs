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

        // Interpolator 状态（degree = 2 → 3 层）
        private readonly float[] _loArr;
        private readonly float[] _upArr;
        private bool _initialized;

        private Params _params;

        public bool IsReady => _neutral != null;
        public int Size => _size;
        public float[] Neutral => _neutral;
        public Params CurrentParams => _params;

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

            _loArr = new float[_size * 3];
            _upArr = new float[_size * 3];
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
                return;
            }

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
            if (_loArr != null) Array.Clear(_loArr, 0, _loArr.Length);
            if (_upArr != null) Array.Clear(_upArr, 0, _upArr.Length);
        }

        /// <summary>
        /// 跑一帧。
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

            float aLo = _params.lowerFaceSmoothing > 0f
                ? 1f - Mathf.Pow(0.5f, dt / _params.lowerFaceSmoothing) : 1f;
            float aUp = _params.upperFaceSmoothing > 0f
                ? 1f - Mathf.Pow(0.5f, dt / _params.upperFaceSmoothing) : 1f;

            int s = _size;

            if (!_initialized)
            {
                // InterpolatorInitDataKernel：三层全部初始化成当前 raw
                for (int i = 0; i < s; i++)
                {
                    float p = skinStrength * deltas[srcOffset + i]
                            + _eyeCloseDelta[i] * eyeCoef
                            + _lipOpenDelta[i] * lipCoef;
                    _loArr[i] = p; _loArr[s + i] = p; _loArr[2 * s + i] = p;
                    _upArr[i] = p; _upArr[s + i] = p; _upArr[2 * s + i] = p;
                }
                _initialized = true;
            }

            float lowerStrength = _params.lowerFaceStrength;
            float upperStrength = _params.upperFaceStrength;

            // 按顶点外层、分量内层：faceMask 每 3 个分量才取一次，且全部顺序访存
            for (int v = 0; v < _vertexCount; v++)
            {
                float m = _maskLower[v];
                int i3 = v * 3;

                for (int c = 0; c < 3; c++)
                {
                    int i = i3 + c;

                    // --- Step1 ---
                    float p = skinStrength * deltas[srcOffset + i]
                            + _eyeCloseDelta[i] * eyeCoef
                            + _lipOpenDelta[i] * lipCoef;

                    // --- Interpolator (degree=2)：dataArr[0]=raw, dataArr[k] += (dataArr[k-1]-dataArr[k])*alpha ---
                    _loArr[i] = p;
                    float l1 = _loArr[s + i] + (_loArr[i] - _loArr[s + i]) * aLo;
                    _loArr[s + i] = l1;
                    float l2 = _loArr[2 * s + i] + (l1 - _loArr[2 * s + i]) * aLo;
                    _loArr[2 * s + i] = l2;

                    _upArr[i] = p;
                    float u1 = _upArr[s + i] + (_upArr[i] - _upArr[s + i]) * aUp;
                    _upArr[s + i] = u1;
                    float u2 = _upArr[2 * s + i] + (u1 - _upArr[2 * s + i]) * aUp;
                    _upArr[2 * s + i] = u2;

                    // --- Step2 ---
                    output[dstOffset + i] = _neutral[i]
                                          + u2 * upperStrength * (1f - m)
                                          + l2 * lowerStrength * m;
                }
            }
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
