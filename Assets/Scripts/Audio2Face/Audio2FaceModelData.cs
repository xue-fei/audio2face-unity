using System.IO;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// model_data_{identity}.npz：skin animator 需要的 neutral 与两个静态 pose delta。
    ///
    /// 它和 bs_skin_{identity}.npz 不是一回事，别混用：
    ///   - model_data 的 neutral_skin 是 animator 的基准网格（prediction 叠在它上面）
    ///   - bs_skin 的 neutral 是求解器算 target 时减掉的那个
    /// 两者数值几乎相同（Mark 实测差 ~0.015 RMS），但语义不同。
    ///
    /// npz 内容（实测）：
    ///   neutral_skin          (24002, 3) → 72006 = network_info.skin_size
    ///   neutral_tongue        (5602, 3)  → 16806 = tongue_size
    ///   neutral_jaw           (5, 3)
    ///   eye_close_pose_delta  (24002, 3)
    ///   lip_open_pose_delta   (24002, 3)
    ///   saccade_rot_matrix    (5000, 2)
    /// </summary>
    public sealed class Audio2FaceModelData
    {
        public float[] NeutralSkin { get; private set; }
        public float[] NeutralTongue { get; private set; }
        public float[] NeutralJaw { get; private set; }
        public float[] SaccadeRotMatrix { get; private set; }
        public float[] EyeClosePoseDelta { get; private set; }
        public float[] LipOpenPoseDelta { get; private set; }

        public int SkinVertexCount => NeutralSkin != null ? NeutralSkin.Length / 3 : 0;
        public int TongueVertexCount => NeutralTongue != null ? NeutralTongue.Length / 3 : 0;

        /// <summary>skin animator 需要的三件东西都齐了才算可用。</summary>
        public bool IsReady => NeutralSkin != null && EyeClosePoseDelta != null && LipOpenPoseDelta != null;

        public static Audio2FaceModelData Load(string folder, string identity)
        {
            string path = Path.Combine(folder, $"model_data_{identity}.npz");
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[Audio2Face] 找不到 {path}，skin animator 将不可用");
                return null;
            }

            var npz = NpzFile.Load(path);
            if (npz == null) return null;

            var data = new Audio2FaceModelData();
            data.NeutralSkin = npz.Has("neutral_skin") ? npz.GetFloats("neutral_skin") : null;
            data.NeutralTongue = npz.Has("neutral_tongue") ? npz.GetFloats("neutral_tongue") : null;
            data.NeutralJaw = npz.Has("neutral_jaw") ? npz.GetFloats("neutral_jaw") : null;
            data.SaccadeRotMatrix = npz.Has("saccade_rot_matrix") ? npz.GetFloats("saccade_rot_matrix") : null;
            data.EyeClosePoseDelta = npz.Has("eye_close_pose_delta") ? npz.GetFloats("eye_close_pose_delta") : null;
            data.LipOpenPoseDelta = npz.Has("lip_open_pose_delta") ? npz.GetFloats("lip_open_pose_delta") : null;

            if (!data.IsReady)
                Debug.LogWarning($"[Audio2Face] {path} 缺少 neutral_skin / eye_close_pose_delta / lip_open_pose_delta");

            return data;
        }
    }
}
