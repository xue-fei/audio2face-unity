using System;
using UnityEngine;

namespace Audio2Emotion
{
    /// <summary>
    /// Audio2Emotion v2.2 模型配置
    /// 模型架构: Wav2Vec2-Large-LV60
    /// 输入: 原始 16kHz PCM float32 音频
    /// 输出: 6 种情绪概率 [angry, disgust, fear, happy, neutral, sad]
    /// </summary>
    [Serializable]
    public class Audio2EmotionConfig
    {
        [Header("模型配置")]
        [Tooltip("模型在 StreamingAssets 下的相对路径")]
        public string modelPath = "Audio2Emotion-v2.2/network.onnx";

        [Header("音频参数")]
        [Tooltip("采样率 (模型要求 16000Hz)")]
        public int sampleRate = 16000;

        [Tooltip("每次推理的音频长度（秒）")]
        public float inferenceWindowSeconds = 0.5f;

        [Tooltip("音频帧移（秒）")]
        public float frameShiftSeconds = 0.033f;

        [Header("执行提供器")]
        [Tooltip("是否使用 CUDA GPU 加速")]
        public bool useCUDA = true;

        [Tooltip("CUDA 设备 ID")]
        public int cudaDeviceId = 0;

        [Header("后处理参数")]
        [Tooltip("输出情绪长度（历史帧数）")]
        public int outputEmotionLength = 10;

        [Tooltip("情绪对比度增强系数")]
        public float emotionContrast = 1.0f;

        [Tooltip("实时混合系数（历史 vs 当前）")]
        public float liveBlendCoef = 0.7f;

        [Tooltip("情绪强度")]
        public float emotionStrength = 0.6f;

        [Tooltip("最大情绪数量")]
        public int maxEmotions = 6;

        [Tooltip("固定时间步长")]
        public float fixedDt = 0.033f;

        [Tooltip("过渡平滑系数")]
        public float transitionSmoothing = 0.5f;

        [Header("偏好情绪")]
        [Tooltip("是否启用偏好情绪")]
        public bool enablePreferredEmotion = false;

        [Tooltip("偏好情绪权重")]
        public float preferredEmotionStrength = 0.5f;

        [Tooltip("偏好情绪向量")]
        public float[] preferredEmotion = new float[10];

        [Header("调试")]
        [Tooltip("是否打印调试信息")]
        public bool debugMode = false;

        /// <summary>
        /// 每次推理的采样点数
        /// </summary>
        public int InferenceWindowSize => (int)(sampleRate * inferenceWindowSeconds);

        /// <summary>
        /// 帧移采样点数
        /// </summary>
        public int FrameShiftSize => (int)(sampleRate * frameShiftSeconds);

        /// <summary>
        /// 情绪名称
        /// </summary>
        public static readonly string[] EmotionNames = {
            "angry", "disgust", "fear", "happy", "neutral", "sad"
        };

        /// <summary>
        /// 情绪对应 Audio2Face blendshape 索引
        /// -1 表示无对应
        /// </summary>
        public static readonly int[] EmotionBlendshapeIndex = {
            1,  // angry
            3,  // disgust
            4,  // fear
            6,  // happy
            -1, // neutral
            9   // sad
        };
    }
}
