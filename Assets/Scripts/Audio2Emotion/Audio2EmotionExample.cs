using UnityEngine;

namespace Audio2Emotion
{
    /// <summary>
    /// Audio2Emotion 使用示例
    /// 展示如何以编程方式初始化和使用推理管线
    /// </summary>
    public class Audio2EmotionExample : MonoBehaviour
    {
        [Header("配置")]
        public Audio2EmotionConfig Config;

        [Header("测试音频")]
        public AudioClip TestClip;

        private Audio2EmotionManager _manager;

        void Start()
        {
            // 方式1：使用 Manager（代码方式）
            _manager = new Audio2EmotionManager();
            _manager.Initialize(Config);

            // 如果有测试音频，执行推理
            if (TestClip != null)
            {
                float[] audioData = new float[TestClip.samples * TestClip.channels];
                TestClip.GetData(audioData, 0);

                // 重采样到 16kHz（如果需要）
                float[] resampled = ResampleTo16kHz(audioData, TestClip.channels, TestClip.frequency);

                var result = _manager.Infer(resampled);
                Debug.Log($"推理结果: {result}");
            }
        }

        void OnDestroy()
        {
            _manager?.Dispose();
        }

        /// <summary>
        /// 简单重采样到 16kHz（最近邻）
        /// </summary>
        float[] ResampleTo16kHz(float[] data, int channels, int originalSampleRate)
        {
            if (originalSampleRate == 16000) return data;

            float ratio = 16000f / originalSampleRate;
            int newLength = (int)(data.Length / channels * ratio);
            float[] result = new float[newLength];

            for (int i = 0; i < newLength; i++)
            {
                int srcIdx = (int)(i / ratio) * channels;
                result[i] = data[srcIdx]; // 取左声道
            }

            return result;
        }

        /// <summary>
        /// 从麦克风实时推理示例
        /// </summary>
        public void StartMicrophoneInference()
        {
            if (Microphone.devices.Length == 0)
            {
                Debug.LogWarning("No microphone found");
                return;
            }

            var mic = Microphone.devices[0];
            int micBufferSec = 1;
            int micSampleRate = 16000;

            AudioSource audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.clip = Microphone.Start(mic, true, micBufferSec, micSampleRate);
            audioSource.loop = true;
            while (!(Microphone.GetPosition(mic) > 0)) { }
            audioSource.Play();

            // 使用协程或 Update 读取音频数据并推理
            StartCoroutine(ProcessMicrophoneAudio(audioSource));
        }

        System.Collections.IEnumerator ProcessMicrophoneAudio(AudioSource source)
        {
            float[] readBuffer = new float[1600]; // 100ms @ 16kHz

            while (true)
            {
                if (source.isPlaying)
                {
                    int pos = source.timeSamples;
                    source.clip.GetData(readBuffer, pos);

                    // 检查是否全部为零（静音）
                    bool hasAudio = false;
                    for (int i = 0; i < readBuffer.Length; i++)
                    {
                        if (Mathf.Abs(readBuffer[i]) > 0.001f)
                        {
                            hasAudio = true;
                            break;
                        }
                    }

                    if (hasAudio)
                    {
                        var result = _manager.ProcessAudioFrame(readBuffer);
                        if (result && Config.debugMode)
                        {
                            Debug.Log($"[Mic] {_manager.LatestResult}");
                        }
                    }
                }
                yield return new WaitForSeconds(0.033f);
            }
        }
    }
}
