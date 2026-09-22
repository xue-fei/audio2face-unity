using System;
using UnityEngine;

namespace Audio2Face
{
    [Serializable]
    public class NetworkInfoFile
    {
        public IdBlock id;
        public NetworkParamsBlock @params;
        public AudioParamsBlock audio_params;
    }

    [Serializable]
    public class IdBlock
    {
        public string type;
        public string actor;
        public string version;
        public string output;
    }

    [Serializable]
    public class NetworkParamsBlock
    {
        public string[] emotions;
        public float[] default_emotion;
        public string[] identities;
        public int skin_size;
        public int tongue_size;
        public int jaw_size;
        public int eyes_size;
        public int num_diffusion_steps;
        public int num_gru_layers;
        public int gru_latent_dim;
        public int num_frames_left_truncate;
        public int num_frames_right_truncate;
        public int num_frames_center;
    }

    [Serializable]
    public class AudioParamsBlock
    {
        public int buffer_len;
        public int padding_left;
        public int padding_right;
        public int samplerate;
    }

    /// <summary>
    /// nvidiaAudio2Face-3D-v3.0 的 network_info.json 运行时封装。
    /// 实测值（Mark/Claire/James 通用）：
    ///   skin=72006, tongue=16806, jaw=15, eyes=4 → 每帧 88831 维
    ///   frames = 15(左截断) + 30(中心) + 15(右截断) = 60
    ///   buffer_len=16000 @16kHz, 步长 = 16000 * 30 / 60 = 8000 样本
    /// </summary>
    public sealed class Audio2FaceNetworkInfo
    {
        public string ModelType { get; private set; }
        public string OutputType { get; private set; }
        public string[] Identities { get; private set; }
        public int SkinSize { get; private set; }
        public int TongueSize { get; private set; }
        public int JawSize { get; private set; }
        public int EyesSize { get; private set; }
        public int TotalDim { get; private set; }
        public int FramesLeftTruncate { get; private set; }
        public int FramesCenter { get; private set; }
        public int FramesRightTruncate { get; private set; }
        public int FramesPerInference { get; private set; }
        public int BufferLength { get; private set; }
        public int PaddingLeft { get; private set; }
        public int PaddingRight { get; private set; }
        public int SampleRate { get; private set; }
        public int NumGruLayers { get; private set; }
        public int GruLatentDim { get; private set; }
        public int NumDiffusionSteps { get; private set; }
        public int SkinOffset { get; private set; }
        public int TongueOffset { get; private set; }
        public int JawOffset { get; private set; }
        public int EyesOffset { get; private set; }

        /// <summary>每次推理推进的样本数（= 中心帧对应的音频长度）</summary>
        public int StrideSamples
        {
            get { return FramesPerInference > 0 ? (BufferLength * FramesCenter) / FramesPerInference : BufferLength; }
        }

        /// <summary>输出帧率（帧/秒）</summary>
        public float FrameRate
        {
            get { return BufferLength > 0 ? FramesPerInference * (float)SampleRate / BufferLength : 60f; }
        }

        /// <summary>预热推理次数：丢弃 padding_left 覆盖到的窗口</summary>
        public int WarmupInferences
        {
            get { return StrideSamples > 0 ? Mathf.CeilToInt((float)PaddingLeft / StrideSamples) : 0; }
        }

        /// <summary>目标帧偏移（样本）= bufferLength * numFramesLeftTruncate / nbFramesPerInference。
        /// SDK 用它确定「中心帧在窗口内的位置」。流式对齐时，第一个正式帧的时间戳
        /// = TargetOffsetSamples（音频样本），据此对齐音频播放与动画消费。</summary>
        public int TargetOffsetSamples
        {
            get { return FramesPerInference > 0 ? (BufferLength * FramesLeftTruncate) / FramesPerInference : 0; }
        }

        public static Audio2FaceNetworkInfo Load(string path)
        {
            var text = System.IO.File.ReadAllText(path);
            var raw = JsonUtility.FromJson<NetworkInfoFile>(text);
            if (raw == null || raw.@params == null)
                throw new Exception($"network_info.json 解析失败: {path}");

            var p = raw.@params;
            var a = raw.audio_params;

            var info = new Audio2FaceNetworkInfo();
            info.ModelType = raw.id != null ? raw.id.type : "unknown";
            info.OutputType = raw.id != null ? raw.id.output : "unknown";
            info.Identities = p.identities ?? new string[0];
            info.SkinSize = p.skin_size;
            info.TongueSize = p.tongue_size;
            info.JawSize = p.jaw_size;
            info.EyesSize = p.eyes_size;
            info.TotalDim = p.skin_size + p.tongue_size + p.jaw_size + p.eyes_size;
            info.FramesLeftTruncate = p.num_frames_left_truncate;
            info.FramesCenter = p.num_frames_center;
            info.FramesRightTruncate = p.num_frames_right_truncate;
            info.FramesPerInference = p.num_frames_left_truncate + p.num_frames_center + p.num_frames_right_truncate;
            info.NumGruLayers = p.num_gru_layers;
            info.GruLatentDim = p.gru_latent_dim;
            info.NumDiffusionSteps = p.num_diffusion_steps;
            info.BufferLength = a != null ? a.buffer_len : 16000;
            info.PaddingLeft = a != null ? a.padding_left : 0;
            info.PaddingRight = a != null ? a.padding_right : 0;
            info.SampleRate = a != null ? a.samplerate : 16000;

            info.SkinOffset = 0;
            info.TongueOffset = info.SkinSize;
            info.JawOffset = info.TongueOffset + info.TongueSize;
            info.EyesOffset = info.JawOffset + info.JawSize;
            return info;
        }

        public int IdentityIndex(string name)
        {
            for (int i = 0; i < Identities.Length; i++)
            {
                if (string.Equals(Identities[i], name, StringComparison.OrdinalIgnoreCase)) return i;
            }
            return -1;
        }
    }
}
