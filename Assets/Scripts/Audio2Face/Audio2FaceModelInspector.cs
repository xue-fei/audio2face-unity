using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// Audio2Face 模型检查器
    /// 用于验证 network.onnx 的输入输出格式
    /// </summary>
    public class Audio2FaceModelInspector : MonoBehaviour
    {
        [Header("模型路径")]
        public string modelPath = "nvidiaAudio2Face-3D-v3.0/network.onnx";

        void Start()
        {
            InspectModel();
        }

        void InspectModel()
        {
            string fullPath = Path.Combine(Application.streamingAssetsPath, modelPath);
            Debug.Log($"[Audio2FaceModelInspector] Model path: {fullPath}");
            Debug.Log($"[Audio2FaceModelInspector] File exists: {File.Exists(fullPath)}");

            if (!File.Exists(fullPath))
            {
                Debug.LogError($"[Audio2FaceModelInspector] Model not found: {fullPath}");
                return;
            }

            // 文件大小
            var fileInfo = new FileInfo(fullPath);
            Debug.Log($"[Audio2FaceModelInspector] File size: {fileInfo.Length / (1024 * 1024):F1} MB");

            try
            {
                using (var session = new InferenceSession(fullPath))
                {
                    Debug.Log($"[Audio2FaceModelInspector] === Model Inputs ===");
                    foreach (var input in session.InputMetadata)
                    {
                        Debug.Log($"  Input '{input.Key}': type={input.Value.ElementType}, dims=[{string.Join(",", input.Value.Dimensions)}]");
                    }

                    Debug.Log($"[Audio2FaceModelInspector] === Model Outputs ===");
                    foreach (var output in session.OutputMetadata)
                    {
                        Debug.Log($"  Output '{output.Key}': type={output.Value.ElementType}, dims=[{string.Join(",", output.Value.Dimensions)}]");
                    }

                    // 推断模型类型
                    AnalyzeModelType(session);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Audio2FaceModelInspector] Failed to load model: {ex.Message}");
            }
        }

        void AnalyzeModelType(InferenceSession session)
        {
            int inputCount = session.InputNames.Count;
            int outputCount = session.OutputNames.Count;

            Debug.Log($"[Audio2FaceModelInspector] === Model Analysis ===");
            Debug.Log($"  Total inputs: {inputCount}");
            Debug.Log($"  Total outputs: {outputCount}");

            // 判断是否为扩散模型
            bool hasWindow = session.InputNames.Contains("window");
            bool hasIdentity = session.InputNames.Contains("identity");
            bool hasEmotion = session.InputNames.Contains("emotion");
            bool hasLatents = session.InputNames.Contains("input_latents");
            bool hasNoise = session.InputNames.Contains("noise");

            if (hasWindow && hasIdentity && hasEmotion && hasLatents && hasNoise)
            {
                Debug.Log("  Model type: DIFFUSION (multi-input with window, identity, emotion, latents, noise)");
            }
            else if (inputCount == 1)
            {
                Debug.Log("  Model type: REGRESSION (single input)");
            }
            else
            {
                Debug.Log($"  Model type: UNKNOWN (inputs: {string.Join(", ", session.InputNames)})");
            }

            // 分析输出维度
            if (session.OutputMetadata.Count > 0)
            {
                var output = session.OutputMetadata.Values.GetEnumerator();
                output.MoveNext();
                var dims = output.Current.Dimensions;
                long totalElements = 1;
                foreach (var dim in dims)
                {
                    totalElements *= dim;
                }
                Debug.Log($"  Output total elements: {totalElements}");

                if (totalElements == 52)
                {
                    Debug.Log("  Output type: BLENDSHAPE WEIGHTS (52 ARKit shapes)");
                }
                else if (totalElements >= 72000)
                {
                    Debug.Log("  Output type: GEOMETRY VERTICES (skin + tongue)");
                }
                else if (totalElements >= 16000)
                {
                    Debug.Log("  Output type: TONGUE VERTICES");
                }
            }
        }
    }
}
