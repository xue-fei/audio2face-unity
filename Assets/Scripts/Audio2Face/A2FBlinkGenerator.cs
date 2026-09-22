using System;
using System.Collections.Generic;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// 随机眨眼调度器 —— 产出 SDK <c>AnimatorSkin</c> 需要的 <c>blinkOffset</c>。
    ///
    /// <b>为什么必须自己写</b>：Audio2Face-3D SDK 里眨眼不是模型算出来的。
    /// animator_cuda.cu 的 AnimatorSkinComposeKernelStep1 是
    /// <code>
    ///   pose = skinStrength * result
    ///        + eyeCloseDelta * (-eyelidOpenOffset + blinkOffset * blinkStrength)
    ///        + lipOpenDelta  * lipOpenOffset
    /// </code>
    /// <c>blinkOffset</c> 是<b>应用层每帧喂进来</b>的参数（animator.h 的 SetBlinkOffset，
    /// 默认 0、取值 [0,1]），SDK 内部没有任何随机眨眼生成器。不喂它就只剩
    /// <c>-eyelidOpenOffset</c>（-0.06）这一项静态偏置，眼睛会略微睁大、且<b>一次都不眨</b>。
    /// 日志里 eyeBlinkL/R 全程 ≤0.022 就是这个原因。
    ///
    /// <b>时刻用帧自己的音频时间戳，不用墙钟</b>：推理是提前算的（队列里会积压 30~60 帧），
    /// 用墙钟会让眨眼落在错误的画面上；用帧时间戳，眨眼就固定落在动画时间轴的同一位置，
    /// 与音频播放进度天然对齐。
    /// </summary>
    public sealed class A2FBlinkGenerator
    {
        public bool enabled = true;

        [Tooltip("两次眨眼之间的最小间隔（秒）")]
        public float minInterval = 2.5f;

        [Tooltip("两次眨眼之间的最大间隔（秒）")]
        public float maxInterval = 6.0f;

        [Tooltip("单次眨眼时长（秒）。真人一次眨眼约 0.1~0.2s")]
        public float duration = 0.16f;

        [Tooltip("「闭上」占整个眨眼的比例。睁眼比闭眼慢，所以取 0.4 左右比较自然")]
        public float closeFraction = 0.4f;

        [Tooltip("连眨两下的概率")]
        public float doubleBlinkChance = 0.15f;

        private readonly List<float> _starts = new List<float>();
        private System.Random _rng;
        private float _scheduledUpTo;

        public A2FBlinkGenerator(int seed = 0)
        {
            Reset(seed);
        }

        /// <summary>重置排期（换一段音频、或重新标定时调用）。</summary>
        public void Reset(int seed = 0)
        {
            _rng = new System.Random(seed);
            _starts.Clear();
            // 排期游标从 0 起算：第一次眨眼的时刻 = 0 + NextInterval() ∈ [minInterval, maxInterval]。
            // ⚠ 之前写成 _scheduledUpTo = minInterval，等于「先静默 minInterval 秒，再等一个
            // interval」，第一次眨眼被推到 5.0~8.5s（实测日志 7.60s），开头 8 秒完全不眨眼。
            _scheduledUpTo = 0f;
        }

        private float NextInterval()
        {
            float span = Math.Max(0f, maxInterval - minInterval);
            return minInterval + (float)_rng.NextDouble() * span;
        }

        private void EnsureScheduled(float t)
        {
            float need = t + duration + 0.5f;
            int guard = 0;
            while (_scheduledUpTo < need && guard++ < 4096)
            {
                float s = _scheduledUpTo + NextInterval();
                _starts.Add(s);

                if (doubleBlinkChance > 0f && _rng.NextDouble() < doubleBlinkChance)
                {
                    // 双眨眼：第一下结束后很快再来一次
                    s += duration * (1.3f + (float)_rng.NextDouble() * 0.8f);
                    _starts.Add(s);
                }
                _scheduledUpTo = s;
            }
        }

        private static float SmoothStep(float x)
        {
            if (x < 0f) x = 0f;
            else if (x > 1f) x = 1f;
            return x * x * (3f - 2f * x);
        }

        /// <summary>
        /// 返回 t 时刻的 blinkOffset：0 = 睁眼，1 = 完全闭合。
        /// <paramref name="t"/> 必须是<b>单调不减</b>的帧时间戳（秒）。
        /// </summary>
        public float Evaluate(float t)
        {
            if (!enabled || duration <= 1e-6f) return 0f;

            EnsureScheduled(t);

            float v = 0f;
            for (int i = _starts.Count - 1; i >= 0; i--)
            {
                float st = _starts[i];
                if (st > t) continue;
                if (t - st > duration) break;          // _starts 升序，后面的只会更早，可以直接停

                float u = (t - st) / duration;
                float s = u < closeFraction
                    ? SmoothStep(u / closeFraction)                                    // 闭：0 → 1
                    : SmoothStep(1f - (u - closeFraction) / (1f - closeFraction));    // 睁：1 → 0
                if (s > v) v = s;
            }

            // 丢掉早已过期的排期，避免长时间运行后列表无限增长
            while (_starts.Count > 0 && _starts[0] + duration < t - 1f)
                _starts.RemoveAt(0);

            return v;
        }
    }
}
