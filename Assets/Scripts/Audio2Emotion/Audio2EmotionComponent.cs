using UnityEngine;
using UnityEngine.Events;

namespace Audio2Emotion
{
    /// <summary>
    /// Audio2Emotion Unity 组件
    /// 可直接挂载到 GameObject，提供 Inspector 配置和事件回调
    /// </summary>
    public class Audio2EmotionComponent : MonoBehaviour
    {
        [Header("模型配置")]
        public Audio2EmotionConfig Config = new Audio2EmotionConfig();

        [Header("音频输入")]
        [Tooltip("如果为空则使用系统默认麦克风")]
        public AudioSource AudioSource;

        [Header("自动启动")]
        public bool AutoStart = true;

        [Header("事件")]
        public UnityEvent<Audio2EmotionResult> OnEmotionDetected;
        public UnityEvent<string> OnEmotionChanged;

        private Audio2EmotionManager _manager;
        private string _lastDominantEmotion;
        private bool _isRunning;

        void Start()
        {
            if (AutoStart)
            {
                StartInference();
            }
        }

        void OnDestroy()
        {
            StopInference();
        }

        void Update()
        {
            if (!_isRunning || _manager == null) return;

            var result = _manager.LatestResult;
            if (result == null) return;

            // 触发事件
            OnEmotionDetected?.Invoke(result);

            // 情绪变化检测
            string currentEmotion = result.GetEmotionName(result.DominantEmotionIndex);
            if (currentEmotion != _lastDominantEmotion)
            {
                _lastDominantEmotion = currentEmotion;
                OnEmotionChanged?.Invoke(currentEmotion);
            }
        }

        /// <summary>
        /// 开始推理
        /// </summary>
        public void StartInference()
        {
            if (_isRunning) return;

            _manager = new Audio2EmotionManager();
            _manager.Initialize(Config);
            _isRunning = true;

            if (Config.debugMode)
            {
                Debug.Log("[Audio2EmotionComponent] Inference started");
            }
        }

        /// <summary>
        /// 停止推理
        /// </summary>
        public void StopInference()
        {
            if (!_isRunning) return;

            _manager?.Dispose();
            _manager = null;
            _isRunning = false;

            if (Config.debugMode)
            {
                Debug.Log("[Audio2EmotionComponent] Inference stopped");
            }
        }

        /// <summary>
        /// 处理外部音频数据
        /// </summary>
        public void ProcessAudio(float[] audioData)
        {
            if (!_isRunning || _manager == null) return;
            _manager.ProcessAudioFrame(audioData);
        }

        /// <summary>
        /// 直接推理音频块
        /// </summary>
        public Audio2EmotionResult Infer(float[] audioData)
        {
            if (!_isRunning || _manager == null) return null;
            return _manager.Infer(audioData);
        }
    }
}
