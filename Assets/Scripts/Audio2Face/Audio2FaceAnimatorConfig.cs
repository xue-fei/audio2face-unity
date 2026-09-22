using System;
using System.IO;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// model_config_{identity}.json 的运行时封装，并叠一层「官方 stylization face_params」。
    ///
    /// 关键（两层都要搞清，别只读一层）：
    /// 1. SDK 的 AnimatorSkin/Tongue 参数<b>按身份</b>从 model_config_{identity}.json 读
    ///    （parse_helper.cpp: ModelInfoOwner::Init → ReadAnimatorParams_INTERNAL 读 "config" 段）。
    ///    这个文件里 lower_face_strength 三身份全是 1.0、lip_open_offset 是 0.0/-0.02/0.0。
    /// 2. 但 NVIDIA 官方完整服务（Audio2Face-3D Samples 仓库的 configs/*_diffusion_stylization_config.yaml）
    ///    在 model_config 之上<b>还叠了一层 stylization face_params</b>，这才是运行时真正生效的口型增益：
    ///    Mark:   lower_face_strength=1.4  input_strength=1.3  skin_strength=1.1  lip_close_offset=-0.03
    ///    Claire: lower_face_strength=1.25 input_strength=1.0  skin_strength=1.0  lip_close_offset=0.0
    ///    James:  lower_face_strength=1.2  input_strength=1.0  skin_strength=1.0  lip_close_offset=-0.02
    ///
    /// 结论：animator.h 的 REFL 默认 1.3/-0.03 只是「配置缺失时兜底」；model_config 的 1.0 是 SDK 库
    /// 的基线；官方 stylization 的 1.4 才是最终运行时值。只读 model_config 会把下半脸增益丢掉 40%，
    /// 这正是「嘴不明显」的根因之一。本类的 Load 会在 model_config 之上叠加官方 stylization 值。
    /// </summary>
    [Serializable]
    public class Audio2FaceAnimatorConfig
    {
        [Serializable]
        public class ConfigBlock
        {
            public float input_strength = 1f;
            public float upper_face_smoothing = 0.001f;
            public float lower_face_smoothing = 0.0023f;
            public float upper_face_strength = 1f;
            public float lower_face_strength = 1f;      // 配置值；SDK 代码默认 1.3，本模型不用
            public float face_mask_level = 0.6f;
            public float face_mask_softness = 0.0085f;
            public float skin_strength = 1f;
            public float blink_strength = 1f;
            public float lip_open_offset = 0f;          // 配置值；SDK 代码默认 -0.03，本模型不用
            public float tongue_strength = 1.5f;
            public float tongue_height_offset = 0.2f;
            public float tongue_depth_offset = 0.13f;
            public float eyelid_open_offset = 0.06f;    // Mark=0.06，Claire/James=0.0

            // ---- AnimatorTeeth（jaw_size=15 → 4x4 刚体变换，驱动下颌骨/牙齿）----
            public float lower_teeth_strength = 1f;
            public float lower_teeth_height_offset = 0f;
            public float lower_teeth_depth_offset = 0f;

            // ---- AnimatorEyes（eyes_size=4 → 左右眼欧拉角，度）----
            public float eyeballs_strength = 1f;
            public float saccade_strength = 0.9f;
            public float right_eye_rot_x_offset = 0f;
            public float right_eye_rot_y_offset = -2f;
            public float left_eye_rot_x_offset = 0f;
            public float left_eye_rot_y_offset = 2f;
            public float eye_saccade_seed = 0f;
        }

        public ConfigBlock config = new ConfigBlock();

        /// <summary>喂给网络的音频增益（SDK ReadAudioBuffer 的 inputStrength）。本模型三个身份都是 1.0。</summary>
        public float InputStrength => config != null ? config.input_strength : 1f;

        /// <summary>
        /// 官方 stylization face_params（Audio2Face-3D-Samples/configs/*_diffusion_stylization_config.yaml）。
        /// 这是完整服务在 model_config 之上叠加的<b>运行时口型增益</b>，三身份不同。
        /// </summary>
        [Serializable]
        public class StylizationFaceParams
        {
            public float lower_face_strength;
            public float input_strength;
            public float skin_strength;
            public float lip_close_offset;   // 对应 model_config 的 lip_open_offset（命名不同、同一字段）
            public float tongue_strength;     // Mark=1.3  Claire=1.0  James=1.0（model_config 基线是 1.5/1.3/1.3）
            public float tongue_height_offset; // 官方全 0（model_config 基线是 0.2/0.0/0.0）
            public float tongue_depth_offset; // 官方全 0（model_config 基线是 0.13/0.0/0.0）
        }

        private static readonly System.Collections.Generic.Dictionary<string, StylizationFaceParams> OfficialStylization =
            new System.Collections.Generic.Dictionary<string, StylizationFaceParams>(StringComparer.OrdinalIgnoreCase)
            {
                { "mark",   new StylizationFaceParams { lower_face_strength = 1.4f,  input_strength = 1.3f, skin_strength = 1.1f, lip_close_offset = -0.03f, tongue_strength = 1.3f, tongue_height_offset = 0.0f, tongue_depth_offset = 0.0f } },
                { "claire", new StylizationFaceParams { lower_face_strength = 1.25f, input_strength = 1.0f, skin_strength = 1.0f, lip_close_offset = 0.0f,  tongue_strength = 1.0f, tongue_height_offset = 0.0f, tongue_depth_offset = 0.0f } },
                { "james",  new StylizationFaceParams { lower_face_strength = 1.2f,  input_strength = 1.0f, skin_strength = 1.0f, lip_close_offset = -0.02f, tongue_strength = 1.0f, tongue_height_offset = 0.0f, tongue_depth_offset = 0.0f } },
            };

        /// <summary>转成 AnimatorSkin 的参数结构。</summary>
        public Audio2FaceSkinAnimator.Params ToSkinParams()
        {
            var c = config ?? new ConfigBlock();
            return new Audio2FaceSkinAnimator.Params
            {
                lowerFaceSmoothing = c.lower_face_smoothing,
                upperFaceSmoothing = c.upper_face_smoothing,
                lowerFaceStrength  = c.lower_face_strength,
                upperFaceStrength  = c.upper_face_strength,
                faceMaskLevel      = c.face_mask_level,
                faceMaskSoftness   = c.face_mask_softness,
                skinStrength       = c.skin_strength,
                blinkStrength      = c.blink_strength,
                eyelidOpenOffset   = c.eyelid_open_offset,
                lipOpenOffset      = c.lip_open_offset,
                blinkOffset        = 0f,   // SDK 明确写死 0（配置文件里没有这一项）
            };
        }

        /// <summary>转成 AnimatorTeeth 的参数结构（下颌/牙齿走刚体变换，不走 blendshape）。</summary>
        public Audio2FaceJawAnimator.Params ToTeethParams()
        {
            var c = config ?? new ConfigBlock();
            return new Audio2FaceJawAnimator.Params
            {
                lowerTeethStrength     = c.lower_teeth_strength,
                lowerTeethHeightOffset = c.lower_teeth_height_offset,
                lowerTeethDepthOffset  = c.lower_teeth_depth_offset,
            };
        }

        /// <summary>转成 AnimatorEyes 的参数结构（眼球旋转，度）。</summary>
        public Audio2FaceEyesAnimator.Params ToEyesParams()
        {
            var c = config ?? new ConfigBlock();
            return new Audio2FaceEyesAnimator.Params
            {
                eyeballsStrength            = c.eyeballs_strength,
                saccadeStrength             = c.saccade_strength,
                rightEyeballRotationOffsetX = c.right_eye_rot_x_offset,
                rightEyeballRotationOffsetY = c.right_eye_rot_y_offset,
                leftEyeballRotationOffsetX  = c.left_eye_rot_x_offset,
                leftEyeballRotationOffsetY  = c.left_eye_rot_y_offset,
                saccadeSeed                 = c.eye_saccade_seed,
            };
        }

        /// <summary>读不到 / 解析失败时返回 null，调用方回退到 SDK 代码默认值。</summary>
        public static Audio2FaceAnimatorConfig Load(string folder, string identity)
        {
            string path = Path.Combine(folder, $"model_config_{identity}.json");
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[Audio2Face] 找不到 {path}，animator 参数回退到 SDK 代码默认值（lowerFaceStrength=1.3 等）");
                return null;
            }
            try
            {
                var cfg = JsonUtility.FromJson<Audio2FaceAnimatorConfig>(File.ReadAllText(path));
                if (cfg == null || cfg.config == null)
                {
                    Debug.LogWarning($"[Audio2Face] {path} 解析失败，animator 参数回退到 SDK 代码默认值");
                    return null;
                }

                // 在 model_config 基线之上叠加官方 stylization face_params（运行时口型增益）。
                // 这是 NVIDIA 完整服务（Audio2Face-3D-Samples 的 *_diffusion_stylization_config.yaml）真正用的值。
                if (OfficialStylization.TryGetValue(identity ?? "", out var sty))
                {
                    cfg.config.lower_face_strength = sty.lower_face_strength;
                    cfg.config.input_strength      = sty.input_strength;
                    cfg.config.skin_strength       = sty.skin_strength;
                    cfg.config.lip_open_offset     = sty.lip_close_offset;
                    cfg.config.tongue_strength     = sty.tongue_strength;
                    cfg.config.tongue_height_offset = sty.tongue_height_offset;
                    cfg.config.tongue_depth_offset  = sty.tongue_depth_offset;
                }
                return cfg;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Audio2Face] 读取 {path} 失败: {e.Message}");
                return null;
            }
        }
    }
}
