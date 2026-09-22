using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Audio2Emotion
{
    /// <summary>
    /// Audio2Emotion ONNX 推理会话
    /// 负责加载模型、管理输入缓冲区、执行推理
    /// </summary>
    public class Audio2EmotionSession : IDisposable
    {
        private InferenceSession _session;
        private Audio2EmotionConfig _config;
        private bool _disposed;
        private string _modelFullPath;

        // 输入缓冲区（避免每帧分配）
        private float[] _inputBuffer;
        private DenseTensor<float> _inputTensor;
        private NamedOnnxValue _inputNamedValue;

        // 同步
        private readonly object _lock = new object();

        /// <summary>
        /// 模型输入名称
        /// </summary>
        public IReadOnlyList<string> InputNames => _session?.InputNames;

        /// <summary>
        /// 模型输出名称
        /// </summary>
        public IReadOnlyList<string> OutputNames => _session?.OutputNames;

        /// <summary>
        /// 会话是否已初始化
        /// </summary>
        public bool IsInitialized => _session != null && !_disposed;

        /// <summary>
        /// 初始化推理会话
        /// </summary>
        public void Initialize(Audio2EmotionConfig config)
        {
            _config = config;

            // 构建完整路径
            _modelFullPath = Path.Combine(Application.streamingAssetsPath, config.modelPath);

            if (!File.Exists(_modelFullPath))
            {
                throw new FileNotFoundException($"ONNX model not found: {_modelFullPath}");
            }

            // 配置会话选项
            SessionOptions options;

            if (config.useCUDA)
            {
                try
                {
                    options = SessionOptions.MakeSessionOptionWithCudaProvider(config.cudaDeviceId);
                    Debug.Log($"[Audio2Emotion] Using CUDA device {config.cudaDeviceId}");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[Audio2Emotion] CUDA not available, falling back to CPU: {ex.Message}");
                    options = new SessionOptions();
                }
            }
            else
            {
                options = new SessionOptions();
            }

            // 设置图优化级别
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            options.InterOpNumThreads = 1;
            options.IntraOpNumThreads = 1;

            // 创建会话
            _session = new InferenceSession(_modelFullPath, options);

            // 打印模型信息
            if (config.debugMode)
            {
                Debug.Log($"[Audio2Emotion] Model loaded: {_modelFullPath}");
                Debug.Log($"[Audio2Emotion] Inputs: {string.Join(", ", _session.InputNames)}");
                Debug.Log($"[Audio2Emotion] Outputs: {string.Join(", ", _session.OutputNames)}");

                foreach (var input in _session.InputMetadata)
                {
                    Debug.Log($"[Audio2Emotion] Input '{input.Key}': type={input.Value.ElementType}, dims=[{string.Join(",", input.Value.Dimensions)}]");
                }
                foreach (var output in _session.OutputMetadata)
                {
                    Debug.Log($"[Audio2Emotion] Output '{output.Key}': type={output.Value.ElementType}, dims=[{string.Join(",", output.Value.Dimensions)}]");
                }
            }

            // 预分配输入缓冲区
            int windowSize = config.InferenceWindowSize;
            _inputBuffer = new float[windowSize];
            _inputTensor = new DenseTensor<float>(new[] { 1, windowSize });
            _inputNamedValue = NamedOnnxValue.CreateFromTensor(_session.InputNames[0], _inputTensor);
        }

        /// <summary>
        /// 执行推理
        /// </summary>
        /// <param name="audioData">16kHz PCM float32 音频数据</param>
        /// <param name="length">有效数据长度</param>
        /// <returns>推理结果</returns>
        public Audio2EmotionResult Infer(float[] audioData, int length)
        {
            if (_disposed || _session == null)
            {
                throw new InvalidOperationException("Session not initialized or disposed");
            }

            var result = new Audio2EmotionResult();
            var stopwatch = Stopwatch.StartNew();

            lock (_lock)
            {
                try
                {
                    // 填充输入缓冲区（零填充）
                    int windowSize = _config.InferenceWindowSize;
                    int copyLength = Math.Min(length, windowSize);
                    Array.Clear(_inputBuffer, 0, windowSize);
                    Array.Copy(audioData, _inputBuffer, copyLength);

                    // 复制到 tensor
                    for (int i = 0; i < windowSize; i++)
                    {
                        _inputTensor[0, i] = _inputBuffer[i];
                    }

                    // 执行推理
                    using (var outputs = _session.Run(new[] { _inputNamedValue }))
                    {
                        // 解析输出
                        ParseOutput(outputs, result);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[Audio2Emotion] Inference error: {ex.Message}");
                    throw;
                }
            }

            stopwatch.Stop();
            result.InferenceTimeMs = (float)stopwatch.Elapsed.TotalMilliseconds;
            result.Timestamp = Time.time;

            return result;
        }

        /// <summary>
        /// 解析模型输出
        /// </summary>
        private void ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Audio2EmotionResult result)
        {
            foreach (var output in outputs)
            {
                if (output.Value is DenseTensor<float> tensor)
                {
                    int dim0 = tensor.Dimensions[0];
                    int dim1 = tensor.Dimensions.Length > 1 ? tensor.Dimensions[1] : 1;

                    // 输出形状: [batch, num_emotions] 或 [batch, num_emotions, ...]
                    if (dim1 >= 6)
                    {
                        // 取第一个 batch
                        for (int i = 0; i < 6; i++)
                        {
                            result.RawProbabilities[i] = tensor[0, i];
                        }
                    }
                    else if (dim0 >= 6 && dim1 == 1)
                    {
                        // 可能是 [num_emotions, 1]
                        for (int i = 0; i < 6; i++)
                        {
                            result.RawProbabilities[i] = tensor[i, 0];
                        }
                    }
                    else
                    {
                        // 尝试展平读取
                        for (int i = 0; i < 6 && i < tensor.Length; i++)
                        {
                            result.RawProbabilities[i] = tensor.GetValue(i);
                        }
                    }
                }
            }

            // 计算主导情绪
            result.DominantEmotionIndex = 0;
            result.DominantEmotionConfidence = result.RawProbabilities[0];
            for (int i = 1; i < 6; i++)
            {
                if (result.RawProbabilities[i] > result.DominantEmotionConfidence)
                {
                    result.DominantEmotionConfidence = result.RawProbabilities[i];
                    result.DominantEmotionIndex = i;
                }
            }
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;

            lock (_lock)
            {
                _inputNamedValue = null;
                _inputTensor = null;
                _inputBuffer = null;
                _session?.Dispose();
                _session = null;
            }

            _disposed = true;
        }

        ~Audio2EmotionSession()
        {
            Dispose();
        }
    }
}
