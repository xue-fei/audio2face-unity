using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// Audio2Face v3.0 扩散模型的 ONNX Runtime 推理封装。
    ///
    /// 模型 I/O（实测）：
    ///   输入 window        [1, 16000]        1 秒 @16kHz 音频
    ///   输入 identity      [1, 3]            one-hot（Claire/James/Mark）
    ///   输入 emotion       [1, 30, 10]       30 帧 × 10 维情绪
    ///   输入 input_latents [2, 2, 1, 256]    2 层双向 GRU 隐状态
    ///   输入 noise         [1, 3, 60, 88831] N(0,1)，3 = 扩散步数 + 1
    ///   输出 prediction    [1, 60, 88831]    60 帧几何（skin+tongue+jaw+eyes）
    ///   输出 output_latents[2, 2, 1, 256]    必须回传成下一次的 input_latents
    /// </summary>
    public sealed class Audio2FaceDiffusionModel : IDisposable
    {
        private InferenceSession _session;
        private Audio2FaceNetworkInfo _info;
        private Audio2FaceDiffusionConfig _config;

        private float[] _window;
        private float[] _identity;
        private float[] _emotion;
        private float[] _latents;
        private float[] _noise;
        private float[] _prediction;

        private List<NamedOnnxValue> _inputs;
        private List<string> _outputNames;

        private int _framesPerRun;
        private int _totalDim;
        private int _emotionFrames;
        private int _emotionDim;

        public bool IsInitialized => _session != null;
        public Audio2FaceNetworkInfo Info => _info;
        public int FramesPerRun => _framesPerRun;
        public int TotalDim => _totalDim;
        public float[] Prediction => _prediction;

        /// <summary>
        /// 输入音频增益（对应 SDK 的 inputStrength，ReadAudioBuffer 里对音频乘这个）。
        /// 官方 stylization 里 Mark=1.3、Claire/James=1.0，由 Pipeline 构造时从
        /// Audio2FaceAnimatorConfig.InputStrength 写入。
        /// </summary>
        public float InputStrength { get; set; } = 1f;

        public void Initialize(Audio2FaceDiffusionConfig config)
        {
            _config = config;
            string folder = config.ModelFolderFullPath;

            string infoPath = System.IO.Path.Combine(folder, "network_info.json");
            _info = Audio2FaceNetworkInfo.Load(infoPath);

            string modelPath = System.IO.Path.Combine(folder, "network.onnx");
            if (!System.IO.File.Exists(modelPath))
                throw new System.IO.FileNotFoundException($"模型不存在: {modelPath}");

            var options = CreateSessionOptions(config);
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

            _session = new InferenceSession(modelPath, options);

            ReadMetadata();
            AllocateBuffers(config);
            BuildInputs();

            // 关键诊断：打印模型实际的输入/输出名。如果 output_latents 的真实名字不同，
            // Run 里的按名拷贝会静默跳过 → GRU 状态永远为零 → 输出与音频完全脱节。
            Debug.Log($"[Audio2Face] 模型输入: [{string.Join(", ", _session.InputMetadata.Keys)}]");
            Debug.Log($"[Audio2Face] 模型输出: [{string.Join(", ", _session.OutputNames)}]");

            if (config.debugMode)
            {
                Debug.Log($"[Audio2Face] 模型加载完成: {modelPath}");
                Debug.Log($"[Audio2Face] type={_info.ModelType} output={_info.OutputType} " +
                          $"identities=[{string.Join(",", _info.Identities)}]");
                Debug.Log($"[Audio2Face] frames/run={_framesPerRun} 每帧维度={_totalDim} " +
                          $"步长={_info.StrideSamples} 样本 帧率={_info.FrameRate:F1}fps 预热={_info.WarmupInferences} 次");
            }
        }

        private static SessionOptions CreateSessionOptions(Audio2FaceDiffusionConfig config)
        {
            if (config.executionProvider == Audio2FaceExecutionProvider.TensorRT)
            {
                try
                {
                    var opt = SessionOptions.MakeSessionOptionWithTensorrtProvider(config.deviceId);
                    Debug.Log($"[Audio2Face] 使用 TensorRT EP (device {config.deviceId})");
                    return opt;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[Audio2Face] TensorRT 不可用，回退 CUDA: {ex.Message}");
                }
            }

            if (config.executionProvider != Audio2FaceExecutionProvider.CPU)
            {
                try
                {
                    var opt = SessionOptions.MakeSessionOptionWithCudaProvider(config.deviceId);
                    Debug.Log($"[Audio2Face] 使用 CUDA EP (device {config.deviceId})");
                    return opt;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[Audio2Face] CUDA 不可用，回退 CPU: {ex.Message}");
                }
            }

            Debug.LogWarning("[Audio2Face] 使用 CPU EP，扩散模型会非常慢");
            return new SessionOptions();
        }

        private void ReadMetadata()
        {
            var predMeta = _session.OutputMetadata["prediction"];
            var dims = predMeta.Dimensions;
            // dims = [-1, frames, totalDim]
            _framesPerRun = dims.Length >= 2 ? dims[dims.Length - 2] : _info.FramesPerInference;
            _totalDim = dims.Length >= 1 ? dims[dims.Length - 1] : _info.TotalDim;

            if (_session.InputMetadata.ContainsKey("emotion"))
            {
                var em = _session.InputMetadata["emotion"].Dimensions;
                _emotionFrames = em.Length >= 2 ? em[em.Length - 2] : _info.FramesCenter;
                _emotionDim = em.Length >= 1 ? em[em.Length - 1] : 10;
            }
            else
            {
                _emotionFrames = _info.FramesCenter;
                _emotionDim = 10;
            }

            if (_totalDim != _info.TotalDim)
            {
                Debug.LogWarning($"[Audio2Face] 模型输出维度 {_totalDim} 与 network_info 计算的 " +
                                 $"{_info.TotalDim} 不一致，以模型为准");
            }
        }

        private void AllocateBuffers(Audio2FaceDiffusionConfig config)
        {
            int windowLen = _info.BufferLength;

            _window = new float[windowLen];
            _identity = new float[Mathf.Max(1, _info.Identities.Length)];

            int idIdx = _info.IdentityIndex(config.identity);
            if (idIdx < 0 && _identity.Length > 0) idIdx = 0;
            if (idIdx >= 0) _identity[idIdx] = 1f;

            _emotion = new float[_emotionFrames * _emotionDim];
            _latents = new float[2 * 2 * 1 * Mathf.Max(1, _info.GruLatentDim)];
            _prediction = new float[_framesPerRun * _totalDim];

            int noiseSteps = Mathf.Max(1, _info.NumDiffusionSteps + 1);
            long noiseCount = (long)noiseSteps * _framesPerRun * _totalDim;
            _noise = new float[noiseCount];
            FillGaussian(_noise, config.noiseSeed);

            if (config.debugMode)
            {
                Debug.Log($"[Audio2Face] 缓冲: window={windowLen} emotion=[{_emotionFrames},{_emotionDim}] " +
                          $"latents={_latents.Length} noise={_noise.Length} ({_noise.Length * 4f / 1048576f:F0} MB)");
            }
        }

        private void BuildInputs()
        {
            _inputs = new List<NamedOnnxValue>();
            _outputNames = new List<string>(_session.OutputNames);

            int windowLen = _window.Length;
            int idLen = _identity.Length;
            int noiseSteps = _noise.Length / (_framesPerRun * _totalDim);

            _inputs.Add(NamedOnnxValue.CreateFromTensor("window",
                new DenseTensor<float>(new Memory<float>(_window), new[] { 1, windowLen })));
            _inputs.Add(NamedOnnxValue.CreateFromTensor("identity",
                new DenseTensor<float>(new Memory<float>(_identity), new[] { 1, idLen })));
            _inputs.Add(NamedOnnxValue.CreateFromTensor("emotion",
                new DenseTensor<float>(new Memory<float>(_emotion), new[] { 1, _emotionFrames, _emotionDim })));
            _inputs.Add(NamedOnnxValue.CreateFromTensor("input_latents",
                new DenseTensor<float>(new Memory<float>(_latents), new[] { 2, 2, 1, _latents.Length / 4 })));
            _inputs.Add(NamedOnnxValue.CreateFromTensor("noise",
                new DenseTensor<float>(new Memory<float>(_noise), new[] { 1, noiseSteps, _framesPerRun, _totalDim })));
        }

        /// <summary>填充标准正态 N(0,1) 噪声（对应 SDK 的 curandGenerateNormal(0, 1)）。</summary>
        /// <remarks>
        /// 16M 个分量（61MB）每次推理都要重填，System.Random + Box-Muller 要几百毫秒，直接吃掉实时预算。
        /// 这里换成 xorshift128 自己攒的均匀随机数，同样的 Box-Muller 变换，快一个量级。
        /// 另外默认走 constantNoise（只在初始化时生成一次），跟 SDK 的同名选项一致，输出也确定。
        /// </remarks>
        private static void FillGaussian(float[] data, int seed)
        {
            uint x = (uint)seed * 2654435761u;
            uint y = 362437u, z = 1013904223u, w = 1664525u;
            if (x == 0) x = 0x9E3779B9u;

            int n = data.Length;
            for (int i = 0; i < n; i += 2)
            {
                // xorshift128
                uint t = x ^ (x << 11);
                x = y; y = z; z = w;
                w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
                double u1 = (w + 0.5) * (1.0 / 4294967296.0);

                t = x ^ (x << 11);
                x = y; y = z; z = w;
                w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
                double u2 = (w + 0.5) * (1.0 / 4294967296.0);

                double mag = Math.Sqrt(-2.0 * Math.Log(u1));
                data[i] = (float)(mag * Math.Cos(2.0 * Math.PI * u2));
                if (i + 1 < n) data[i + 1] = (float)(mag * Math.Sin(2.0 * Math.PI * u2));
            }
        }

        /// <summary>换一个噪声种子重新生成噪声（自检用：看输出对噪声有多敏感）。</summary>
        public void SetNoiseSeed(int seed)
        {
            if (_noise == null) return;
            FillGaussian(_noise, seed);
        }

        /// <summary>噪声张量是否每次推理都重生成（SDK 默认 constantNoise=true，即只生成一次）。</summary>
        public bool ConstantNoise => _config != null && _config.constantNoise;

        /// <summary>设置情绪向量（10 维），会复制到 emotion 的所有帧。</summary>
        public void SetEmotion(float[] emotions)
        {
            if (_emotion == null || emotions == null) return;
            int dim = Mathf.Min(_emotionDim, emotions.Length);
            for (int f = 0; f < _emotionFrames; f++)
            {
                int dst = f * _emotionDim;
                for (int d = 0; d < _emotionDim; d++) _emotion[dst + d] = d < dim ? emotions[d] : 0f;
            }
        }

        /// <summary>重置 GRU 状态（换一段音频时调用）。</summary>
        public void ResetState()
        {
            if (_latents != null) Array.Clear(_latents, 0, _latents.Length);
            Array.Clear(_window, 0, _window.Length);
        }

        /// <summary>
        /// 跑一次推理。
        /// </summary>
        /// <param name="audioWindow">长度必须等于 BufferLength（16000）</param>
        public void Run(float[] audioWindow)
        {
            if (_session == null) throw new InvalidOperationException("未初始化");

            // SDK 的 inputStrength 在这里应用：对输入音频窗口整体缩放。
            float istr = InputStrength;
            if (Mathf.Approximately(istr, 1f))
            {
                Array.Copy(audioWindow, _window, Mathf.Min(audioWindow.Length, _window.Length));
            }
            else
            {
                int n = Mathf.Min(audioWindow.Length, _window.Length);
                for (int i = 0; i < n; i++) _window[i] = audioWindow[i] * istr;
            }
            if (!_config.constantNoise) FillGaussian(_noise, Environment.TickCount);

            // 音频信号检测：计算 RMS 判断是否有真实音频输入
            float audioRms = 0f;
            if (_window != null && _window.Length > 0)
            {
                double sq = 0.0;
                for (int i = 0; i < _window.Length; i++) sq += (double)_window[i] * _window[i];
                audioRms = (float)Math.Sqrt(sq / _window.Length);
                _lastAudioRms = audioRms;
            }

            using (var results = _session.Run(_inputs, _outputNames))
            {
                foreach (var r in results)
                {
                    if (r.Name == "prediction")
                    {
                        var dense = r.AsTensor<float>() as DenseTensor<float>;
                        if (dense != null)
                        {
                            dense.Buffer.Span.CopyTo(new Span<float>(_prediction));
                        }
                        else
                        {
                            int i = 0;
                            foreach (var v in r.AsEnumerable<float>())
                            {
                                if (i >= _prediction.Length) break;
                                _prediction[i++] = v;
                            }
                        }
                    }
                    else if (r.Name == "output_latents")
                    {
                        var dense = r.AsTensor<float>() as DenseTensor<float>;
                        if (dense != null) dense.Buffer.Span.CopyTo(new Span<float>(_latents));
                    }
                }
            }

            // NaN/Inf 检测 + 输出统计（前 5 次每次都输出，之后每 10 次输出一次）
            _runCount++;
            bool shouldLog = (_runCount <= 5) || (_runCount % 10 == 0);
            
            if (shouldLog)
            {
                int nanCount = 0;
                float predRms = 0f;
                float skinRms = 0f;
                float tongueRms = 0f;
                float skinMax = 0f;
                
                for (int i = 0; i < _prediction.Length; i++)
                {
                    float v = _prediction[i];
                    if (float.IsNaN(v) || float.IsInfinity(v)) nanCount++;
                    double abs = Math.Abs(v);
                    predRms += v * v;
                    if (i < _info.SkinSize)
                    {
                        skinRms += v * v;
                        if (abs > skinMax) skinMax = (float)abs;
                    }
                    else if (i < _info.SkinSize + _info.TongueSize)
                    {
                        tongueRms += v * v;
                    }
                }
                predRms = (float)Math.Sqrt(predRms / _prediction.Length);
                skinRms = (float)Math.Sqrt(skinRms / Math.Max(1, _info.SkinSize));
                tongueRms = (float)Math.Sqrt(tongueRms / Math.Max(1, _info.TongueSize));

                // GRU latents NaN 检测
                int latNan = 0;
                for (int i = 0; i < _latents.Length; i++)
                    if (float.IsNaN(_latents[i]) || float.IsInfinity(_latents[i])) latNan++;

                if (nanCount > 0 || latNan > 0)
                {
                    Debug.LogWarning($"[Audio2Face] ⚠ NaN/Inf: prediction={nanCount}/{_prediction.Length}, latents={latNan}/{_latents.Length} → 推理损坏，重置 GRU");
                    ResetState();
                }
                else
                {
                    string audioStatus = audioRms < 1e-6f ? "静音" : (audioRms < 0.01f ? "很弱" : "正常");
                    if (_runCount <= 10 || _config.debugMode)
                    {
                        double latSq = 0.0;
                        for (int i = 0; i < _latents.Length; i++) latSq += (double)_latents[i] * _latents[i];
                        float latRms = (float)Math.Sqrt(latSq / _latents.Length);
                        Debug.Log($"[Audio2Face] 推理#{_runCount}: audioRMS={audioRms:F6}({audioStatus}) predRMS={predRms:F4} skinRMS={skinRms:F4} skinMax={skinMax:F4} gruRMS={latRms:F4}");
                    }

                    // 音频有信号但皮肤 RMS 为 0 = 模型没响应
                    if (audioRms > 0.001f && skinRms < 1e-6f)
                    {
                        Debug.LogWarning($"[Audio2Face] ⚠ 音频有信号但模型皮肤输出为 0！音频未到达模型或 GRU 状态错误");
                    }
                }
            }
        }

        private int _runCount = 0;
        /// <summary>最后一次推理的音频 RMS（外部可读）。</summary>
        public float AudioRMS => _lastAudioRms;
        private float _lastAudioRms = 0f;

        public void Dispose()
        {
            _session?.Dispose();
            _session = null;
            _window = null;
            _identity = null;
            _emotion = null;
            _latents = null;
            _noise = null;
            _prediction = null;
        }
    }
}
