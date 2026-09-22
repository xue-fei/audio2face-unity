using System;
using UnityEngine;

namespace Audio2Emotion
{
    /// <summary>
    /// Audio2Emotion 管理器
    /// 提供一站式推理接口，管理 Session、PostProcessor 和音频缓冲
    /// 可直接挂载到 GameObject 使用
    /// </summary>
    public class Audio2EmotionManager : IDisposable
    {
        private Audio2EmotionSession _session;
        private Audio2EmotionPostProcessor _postProcessor;
        private Audio2EmotionConfig _config;

        // 音频缓冲（支持流式音频输入）
        private float[] _audioBuffer;
        private int _bufferIndex;

        // 最新结果
        private Audio2EmotionResult _latestResult;
        private readonly object _resultLock = new object();

        /// <summary>
        /// 是否已初始化
        /// </summary>
        public bool IsInitialized { get; private set; }

        /// <summary>
        /// 最新推理结果（线程安全拷贝）
        /// </summary>
        public Audio2EmotionResult LatestResult
        {
            get
            {
                lock (_resultLock)
                {
                    return _latestResult;
                }
            }
        }

        /// <summary>
        /// 初始化
        /// </summary>
        public void Initialize(Audio2EmotionConfig config)
        {
            _config = config;

            _session = new Audio2EmotionSession();
            _session.Initialize(config);

            _postProcessor = new Audio2EmotionPostProcessor(config);

            _audioBuffer = new float[config.InferenceWindowSize];
            _bufferIndex = 0;

            IsInitialized = true;

            Debug.Log($"[Audio2EmotionManager] Initialized with window={config.InferenceWindowSize}, frame={config.FrameShiftSize}");
        }

        /// <summary>
        /// 处理音频帧并执行推理
        /// 将音频添加到缓冲区，当缓冲区满时触发推理
        /// </summary>
        /// <param name="audioFrame">音频帧数据</param>
        /// <returns>如果有推理结果返回 true</returns>
        public bool ProcessAudioFrame(float[] audioFrame)
        {
            if (!IsInitialized) return false;

            int frameIndex = 0;
            bool resultReady = false;

            while (frameIndex < audioFrame.Length)
            {
                int remaining = audioFrame.Length - frameIndex;
                int space = _config.FrameShiftSize - _bufferIndex;
                int copyLength = Math.Min(remaining, space);

                Array.Copy(audioFrame, frameIndex, _audioBuffer, _bufferIndex, copyLength);
                _bufferIndex += copyLength;
                frameIndex += copyLength;

                if (_bufferIndex >= _config.FrameShiftSize)
                {
                    // 执行推理
                    var rawResult = _session.Infer(_audioBuffer, _config.FrameShiftSize);

                    // 后处理
                    _latestResult = _postProcessor.Process(rawResult);
                    resultReady = true;

                    // 保留重叠部分（滑动窗口）
                    int overlap = _config.InferenceWindowSize - _config.FrameShiftSize;
                    if (overlap > 0)
                    {
                        Array.Copy(_audioBuffer, _config.FrameShiftSize, _audioBuffer, 0, overlap);
                    }
                    _bufferIndex = overlap;
                }
            }

            return resultReady;
        }

        /// <summary>
        /// 直接推理整个音频块
        /// </summary>
        /// <param name="audioData">完整音频数据</param>
        /// <returns>推理结果</returns>
        public Audio2EmotionResult Infer(float[] audioData)
        {
            if (!IsInitialized) return null;

            var rawResult = _session.Infer(audioData, audioData.Length);
            _latestResult = _postProcessor.Process(rawResult);
            return _latestResult;
        }

        /// <summary>
        /// 重置缓冲区和后处理器状态
        /// </summary>
        public void Reset()
        {
            _bufferIndex = 0;
            Array.Clear(_audioBuffer, 0, _audioBuffer.Length);
            _postProcessor?.Reset();
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            IsInitialized = false;
            _session?.Dispose();
            _session = null;
            _postProcessor?.Dispose();
            _postProcessor = null;
            _audioBuffer = null;
        }
    }
}
