using System;
using UnityEngine;

namespace Audio2Emotion
{
    /// <summary>
    /// Audio2Emotion 后处理器
    /// 对齐 Audio2Face-3D-SDK 的 postprocess.cpp 流程：
    ///   1. ApplyEmotionContrast: softmax(logits * contrast)
    ///   2. ApplyNullifyUnmappedEmotion: 将 neutral 置零
    ///   3. ApplyMaxEmotions: 保留 top-N 情绪
    ///   4. ApplyA2FEmotionIndexConversion: 6维 → 10维 blendshape 映射
    ///   5. ApplyBlending: 时序混合 (liveBlendCoef)
    ///   6. ApplyPreferredEmotion: 偏好情绪混合
    ///   7. ApplyTransitionSmoothing: 过渡平滑
    ///   8. ApplyEmotionStrength: 强度缩放
    /// </summary>
    public class Audio2EmotionPostProcessor : IDisposable
    {
        private Audio2EmotionConfig _config;

        // 上一帧的 A2F 情绪输出（10维，用于时序混合）
        private float[] _prevEmotion;

        // 上一帧混合后的情绪输出（10维，用于过渡平滑）
        private float[] _prevBlendedEmotion;

        // 首帧标志
        private bool _firstFrame = true;

        // 工作缓冲区
        private float[] _a2eEmotions;  // 6维，推理输出
        private float[] _a2fEmotions;  // 10维，blendshape 输出

        public Audio2EmotionPostProcessor(Audio2EmotionConfig config)
        {
            _config = config;
            _prevEmotion = new float[10];
            _prevBlendedEmotion = new float[10];
            _a2eEmotions = new float[6];
            _a2fEmotions = new float[10];
        }

        /// <summary>
        /// 处理推理结果
        /// </summary>
        /// <param name="rawResult">原始推理结果</param>
        /// <returns>处理后的结果</returns>
        public Audio2EmotionResult Process(Audio2EmotionResult rawResult)
        {
            // 1. ApplyEmotionContrast: softmax(logits * contrast)
            ApplyEmotionContrast(rawResult.RawProbabilities);

            // 2. ApplyNullifyUnmappedEmotion: neutral 置零
            ApplyNullifyUnmappedEmotion();

            // 3. ApplyMaxEmotions: 保留 top-N
            ApplyMaxEmotions();

            // 4. ApplyA2FEmotionIndexConversion: 6维 → 10维
            ApplyA2FEmotionIndexConversion();

            // 5. ApplyBlending: 时序混合
            ApplyBlending();

            // 6. ApplyPreferredEmotion: 偏好情绪
            if (_config.enablePreferredEmotion)
            {
                ApplyPreferredEmotion();
            }

            // 7. ApplyTransitionSmoothing: 过渡平滑
            if (!_firstFrame)
            {
                ApplyTransitionSmoothing();
            }

            // 8. ApplyEmotionStrength: 强度缩放
            ApplyEmotionStrength();

            // 保存状态
            Array.Copy(_a2fEmotions, _prevEmotion, 10);
            Array.Copy(_a2fEmotions, _prevBlendedEmotion, 10);
            _firstFrame = false;

            // 填充结果
            FillResult(rawResult);

            return rawResult;
        }

        /// <summary>
        /// 1. 情绪对比度增强: softmax(logits * contrast)
        /// 对齐 SDK: ApplyEmotionContrast
        /// </summary>
        private void ApplyEmotionContrast(float[] logits)
        {
            // 复制到工作缓冲区
            Array.Copy(logits, _a2eEmotions, 6);

            // 乘以对比度系数
            for (int i = 0; i < 6; i++)
            {
                _a2eEmotions[i] *= _config.emotionContrast;
            }

            // Softmax
            Softmax(_a2eEmotions);
        }

        /// <summary>
        /// 2. 将未映射的情绪置零 (neutral)
        /// 对齐 SDK: ApplyNullifyUnmappedEmotion
        /// </summary>
        private void ApplyNullifyUnmappedEmotion()
        {
            // neutral 在 EmotionBlendshapeIndex 中映射为 -1
            for (int i = 0; i < 6; i++)
            {
                if (Audio2EmotionConfig.EmotionBlendshapeIndex[i] == -1)
                {
                    _a2eEmotions[i] = 0.0f;
                }
            }
        }

        /// <summary>
        /// 3. 保留 top-N 情绪，其余置零
        /// 对齐 SDK: ApplyMaxEmotions
        /// </summary>
        private void ApplyMaxEmotions()
        {
            int maxEmotions = _config.maxEmotions;
            if (maxEmotions >= 6) return;

            // 创建索引数组并按概率排序
            int[] indices = new int[6];
            for (int i = 0; i < 6; i++) indices[i] = i;

            // 简单选择排序：找到最大的 maxEmotions 个
            for (int i = 0; i < maxEmotions; i++)
            {
                int maxIdx = i;
                for (int j = i + 1; j < 6; j++)
                {
                    if (_a2eEmotions[indices[j]] > _a2eEmotions[indices[maxIdx]])
                    {
                        maxIdx = j;
                    }
                }
                // 交换
                int temp = indices[i];
                indices[i] = indices[maxIdx];
                indices[maxIdx] = temp;
            }

            // 将非 top-N 的情绪置零
            bool[] keep = new bool[6];
            for (int i = 0; i < maxEmotions; i++)
            {
                keep[indices[i]] = true;
            }
            for (int i = 0; i < 6; i++)
            {
                if (!keep[i])
                {
                    _a2eEmotions[i] = 0.0f;
                }
            }
        }

        /// <summary>
        /// 4. 6维情绪 → 10维 blendshape 映射
        /// 对齐 SDK: ApplyA2FEmotionIndexConversion
        /// </summary>
        private void ApplyA2FEmotionIndexConversion()
        {
            // 初始化输出
            for (int i = 0; i < 10; i++)
            {
                _a2fEmotions[i] = 0.0f;
            }

            // 映射
            for (int i = 0; i < 6; i++)
            {
                int j = Audio2EmotionConfig.EmotionBlendshapeIndex[i];
                if (j >= 0 && j < 10)
                {
                    _a2fEmotions[j] = _a2eEmotions[i];
                }
            }
        }

        /// <summary>
        /// 5. 时序混合: liveBlendCoef * prevEmotion + (1 - liveBlendCoef) * current
        /// 对齐 SDK: ApplyBlending
        /// </summary>
        private void ApplyBlending()
        {
            float blendCoef = _config.liveBlendCoef;
            for (int i = 0; i < 10; i++)
            {
                _a2fEmotions[i] = blendCoef * _prevEmotion[i] + (1.0f - blendCoef) * _a2fEmotions[i];
            }
        }

        /// <summary>
        /// 6. 偏好情绪混合
        /// 对齐 SDK: ApplyBlending (preferred emotion)
        /// </summary>
        private void ApplyPreferredEmotion()
        {
            float strength = _config.preferredEmotionStrength;
            for (int i = 0; i < 10; i++)
            {
                _a2fEmotions[i] = strength * _config.preferredEmotion[i] + (1.0f - strength) * _a2fEmotions[i];
            }
        }

        /// <summary>
        /// 7. 过渡平滑: w * prevBlended + (1-w) * current, w = dt / transitionTime
        /// 对齐 SDK: ApplyTransitionSmoothing
        /// </summary>
        private void ApplyTransitionSmoothing()
        {
            float dt = _config.fixedDt;
            float transitionTime = Mathf.Max(_config.transitionSmoothing, 0.001f);
            float w = Mathf.Min(dt / transitionTime, 1.0f);

            for (int i = 0; i < 10; i++)
            {
                _a2fEmotions[i] = w * _prevBlendedEmotion[i] + (1.0f - w) * _a2fEmotions[i];
            }
        }

        /// <summary>
        /// 8. 情绪强度缩放
        /// 对齐 SDK: ApplyEmotionStrength
        /// </summary>
        private void ApplyEmotionStrength()
        {
            float strength = _config.emotionStrength;
            for (int i = 0; i < 10; i++)
            {
                _a2fEmotions[i] *= strength;
            }
        }

        /// <summary>
        /// Softmax 激活
        /// </summary>
        private void Softmax(float[] values)
        {
            float maxVal = float.MinValue;
            for (int i = 0; i < values.Length; i++)
            {
                maxVal = Mathf.Max(maxVal, values[i]);
            }

            float sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Mathf.Exp(values[i] - maxVal);
                sum += values[i];
            }

            if (sum > 0)
            {
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] /= sum;
                }
            }
        }

        /// <summary>
        /// 填充最终结果
        /// </summary>
        private void FillResult(Audio2EmotionResult result)
        {
            // ProcessedWeights: 6维概率（softmax 后的值）
            Array.Copy(_a2eEmotions, result.ProcessedWeights, 6);

            // FinalOutput: 10维 blendshape 输出
            Array.Copy(_a2fEmotions, result.FinalOutput, 10);

            // 主导情绪
            result.DominantEmotionIndex = 0;
            result.DominantEmotionConfidence = result.ProcessedWeights[0];
            for (int i = 1; i < 6; i++)
            {
                if (result.ProcessedWeights[i] > result.DominantEmotionConfidence)
                {
                    result.DominantEmotionConfidence = result.ProcessedWeights[i];
                    result.DominantEmotionIndex = i;
                }
            }
        }

        /// <summary>
        /// 重置后处理器状态
        /// </summary>
        public void Reset()
        {
            _firstFrame = true;
            Array.Clear(_prevEmotion, 0, _prevEmotion.Length);
            Array.Clear(_prevBlendedEmotion, 0, _prevBlendedEmotion.Length);
            Array.Clear(_a2eEmotions, 0, _a2eEmotions.Length);
            Array.Clear(_a2fEmotions, 0, _a2fEmotions.Length);
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            _prevEmotion = null;
            _prevBlendedEmotion = null;
            _a2eEmotions = null;
            _a2fEmotions = null;
        }
    }
}
