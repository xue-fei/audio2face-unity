using System;
using UnityEngine;

namespace Audio2Emotion
{
    /// <summary>
    /// Audio2Emotion 推理结果
    /// </summary>
    [Serializable]
    public class Audio2EmotionResult
    {
        /// <summary>
        /// 原始情绪概率 [angry, disgust, fear, happy, neutral, sad]
        /// </summary>
        public float[] RawProbabilities = new float[6];

        /// <summary>
        /// 后处理后的情绪权重 [angry, disgust, fear, happy, neutral, sad]
        /// </summary>
        public float[] ProcessedWeights = new float[6];

        /// <summary>
        /// 混合后的最终输出（含偏好情绪、平滑等）
        /// </summary>
        public float[] FinalOutput = new float[10];

        /// <summary>
        /// 主导情绪索引
        /// </summary>
        public int DominantEmotionIndex;

        /// <summary>
        /// 主导情绪置信度
        /// </summary>
        public float DominantEmotionConfidence;

        /// <summary>
        /// 时间戳
        /// </summary>
        public float Timestamp;

        /// <summary>
        /// 推理耗时（毫秒）
        /// </summary>
        public float InferenceTimeMs;

        /// <summary>
        /// 获取指定情绪的概率
        /// </summary>
        public float GetEmotionWeight(int emotionIndex)
        {
            if (emotionIndex >= 0 && emotionIndex < 6)
                return ProcessedWeights[emotionIndex];
            return 0f;
        }

        /// <summary>
        /// 获取指定情绪的名称
        /// </summary>
        public string GetEmotionName(int emotionIndex)
        {
            if (emotionIndex >= 0 && emotionIndex < Audio2EmotionConfig.EmotionNames.Length)
                return Audio2EmotionConfig.EmotionNames[emotionIndex];
            return "unknown";
        }

        /// <summary>
        /// 获取所有情绪的字符串表示
        /// </summary>
        public override string ToString()
        {
            string[] rawStr = new string[RawProbabilities.Length];
            for (int i = 0; i < RawProbabilities.Length; i++) rawStr[i] = RawProbabilities[i].ToString("F3");
            string[] outStr = new string[FinalOutput.Length];
            for (int i = 0; i < FinalOutput.Length; i++) outStr[i] = FinalOutput[i].ToString("F3");

            return $"[{DominantEmotionIndex}:{Audio2EmotionConfig.EmotionNames[DominantEmotionIndex]}={DominantEmotionConfidence:F3}] " +
                   $"raw=[{string.Join(",", rawStr)}] " +
                   $"out=[{string.Join(",", outStr)}] " +
                   $"time={InferenceTimeMs:F1}ms";
        }
    }
}
