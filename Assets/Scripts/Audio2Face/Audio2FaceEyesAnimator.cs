using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// SDK <c>AnimatorEyes</c> 的 C# 移植（animator.cpp ComputeEyesRotation）。
    ///
    /// 眼球同样<b>不走 blendshape</b>：网络输出的 eyes 段（eyes_size = 4）是
    /// 左右眼各 2 个旋转角，动画器叠上静态 offset 和「微跳视（saccade）」后输出欧拉角（度）。
    ///
    /// 公式（animator.cpp:797）：
    /// <code>
    ///   right.x = rightEyeballRotationOffsetX + eyeballsStrength * res[0] + saccadeStrength * saccadeRot[2*i + 0]
    ///   right.y = rightEyeballRotationOffsetY + eyeballsStrength * res[1] + saccadeStrength * saccadeRot[2*i + 1]
    ///   left.x  = leftEyeballRotationOffsetX  + eyeballsStrength * res[2] + saccadeStrength * saccadeRot[2*i + 0]
    ///   left.y  = leftEyeballRotationOffsetY  + eyeballsStrength * res[3] + saccadeStrength * saccadeRot[2*i + 1]
    ///   z 恒为 0
    /// </code>
    /// saccade 是 model_data 里的 <c>saccade_rot_matrix (5000, 2)</c>：预生成的一段随机眼动，
    /// 按 <b>30fps</b> 推进（SDK 注释明确写了 "Assume the saccade information was meant for 30 FPS"），
    /// 起点由 saccadeSeed 决定，到末尾回绕。这是让眼睛「活着」而不是死盯前方的唯一来源。
    /// </summary>
    public sealed class Audio2FaceEyesAnimator
    {
        public struct Params
        {
            public float eyeballsStrength;
            public float saccadeStrength;
            public float rightEyeballRotationOffsetX;
            public float rightEyeballRotationOffsetY;
            public float leftEyeballRotationOffsetX;
            public float leftEyeballRotationOffsetY;
            public float saccadeSeed;

            public static Params Default => new Params
            {
                eyeballsStrength = 1f,
                saccadeStrength = 0.9f,
                saccadeSeed = 0f,
            };
        }

        /// <summary>SDK 里 saccade 是按 30fps 生成的，与播放帧率无关。</summary>
        private const float SaccadeFps = 30f;

        private readonly float[] _saccadeRot;
        private readonly int _saccadeFrames;
        private Params _params;
        private float _liveTime;

        public bool IsReady => _saccadeRot != null && _saccadeFrames > 0;

        public Audio2FaceEyesAnimator(float[] saccadeRot, Params p)
        {
            if (saccadeRot == null || saccadeRot.Length < 2) return;
            _saccadeRot = saccadeRot;
            _saccadeFrames = saccadeRot.Length / 2;
            _params = p;
            Reset();
        }

        public void SetParams(Params p) => _params = p;

        /// <summary>重置到起点（换音频、标定时调用）。</summary>
        public void Reset()
        {
            _liveTime = 0f;
        }

        /// <summary>
        /// 推进 saccade 游标。对应 SDK 的 IncrementLiveTime，每帧调一次，dt 用真实帧间隔。
        /// </summary>
        public void IncrementLiveTime(float dt)
        {
            if (!IsReady) return;
            _liveTime += dt * SaccadeFps;
            _liveTime = Wrap(_liveTime);
        }

        private float Wrap(float v)
        {
            float r = v % _saccadeFrames;
            if (r < 0f) r += _saccadeFrames;
            return r;
        }

        private int FrameIndex
        {
            get
            {
                float total = Wrap(_params.saccadeSeed + _liveTime);
                int idx = (int)total;
                if (idx < 0) idx = 0;
                if (idx >= _saccadeFrames) idx = _saccadeFrames - 1;
                return idx;
            }
        }

        /// <summary>
        /// 算一帧的左右眼欧拉角（度，长度 3，z 恒 0）。
        /// </summary>
        public bool ComputeEyesRotation(float[] right3, float[] left3, float[] prediction, int offset)
        {
            if (!IsReady) return false;
            if (right3 == null || right3.Length < 3 || left3 == null || left3.Length < 3) return false;
            if (prediction == null || offset < 0 || offset + 4 > prediction.Length) return false;

            int i = 2 * FrameIndex;
            float sx = _saccadeRot[i];
            float sy = _saccadeRot[i + 1];

            float s = _params.eyeballsStrength;
            float k = _params.saccadeStrength;

            right3[0] = _params.rightEyeballRotationOffsetX + s * prediction[offset + 0] + k * sx;
            right3[1] = _params.rightEyeballRotationOffsetY + s * prediction[offset + 1] + k * sy;
            right3[2] = 0f;

            left3[0] = _params.leftEyeballRotationOffsetX + s * prediction[offset + 2] + k * sx;
            left3[1] = _params.leftEyeballRotationOffsetY + s * prediction[offset + 3] + k * sy;
            left3[2] = 0f;

            return true;
        }
    }
}
