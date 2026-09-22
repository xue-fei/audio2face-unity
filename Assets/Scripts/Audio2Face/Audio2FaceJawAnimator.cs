using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// SDK <c>AnimatorTeeth</c> 的 C# 移植（animator.cpp ComputeJawTransform + math_utils.cpp rigidXform）。
    ///
    /// 官方完整管线里下颌/牙齿<b>不走 blendshape</b>，而是走刚体变换：
    /// <code>
    ///   sample-a2f-low-level-api-fullface/main.cpp:
    ///     animatorTeeth->ComputeJawTransform({jawTransform, 16}, transferJaw);
    /// </code>
    /// 网络输出的 jaw 段（jaw_size = 15）是 <b>5 个下颌标记点的位移</b>（不是欧拉角、不是四元数）。
    /// ComputeJawTransform 把它叠上 neutral_jaw 后，用 Kabsch 拟合出一个 4x4 刚体变换，
    /// 直接驱动下颌骨 —— 牙齿挂在骨骼上，所以牙齿才会跟着动。
    /// 只驱动 blendshape 的话嘴巴会「张开」但牙齿纹丝不动，就是这个环节缺失。
    ///
    /// 公式（animator.cpp:579）：
    /// <code>
    ///   jawPose.x = result.x * lowerTeethStrength
    ///   jawPose.y = result.y * lowerTeethStrength + lowerTeethHeightOffset
    ///   jawPose.z = result.z * lowerTeethStrength + lowerTeethDepthOffset
    ///   jawPose  += neutralJaw
    ///   jawTransform4x4 = rigidXform(jawPose, neutralJaw, 5)
    /// </code>
    /// rigidXform 是标准 Kabsch：H = B^T·A → SVD → R = V·diag(1,1,det(V·U^T))·U^T，t = aMean − R·bMean。
    /// 这里用 <b>Horn 四元数法</b>等价求解（构造 4x4 对称矩阵 N，最大特征向量即最优旋转四元数），
    /// 避免引入 3x3 SVD；数值上与 Eigen JacobiSVD 一致到浮点精度。
    /// 输出是<b>列主序</b> 4x4（与 Eigen ColMajor 内存布局一致），平移在下标 12/13/14。
    /// </summary>
    public sealed class Audio2FaceJawAnimator
    {
        public struct Params
        {
            public float lowerTeethStrength;
            public float lowerTeethHeightOffset;
            public float lowerTeethDepthOffset;

            public static Params Default => new Params
            {
                lowerTeethStrength = 1f,
                lowerTeethHeightOffset = 0f,
                lowerTeethDepthOffset = 0f,
            };
        }

        private readonly float[] _neutralJaw;
        private readonly float[] _jawPose;
        private readonly int _nbPoints;
        private Params _params;

        public bool IsReady => _neutralJaw != null && _nbPoints >= 3;

        /// <summary>上一次解出的刚体旋转角（度）。驱动骨骼时用它最直观。</summary>
        public float LastAngleDeg { get; private set; }

        /// <summary>
        /// 上一次解出的旋转轴（A2F 模型坐标系，单位向量）。驱动骨骼时用 angle+axis 而不是 4x4，
        /// 因为 4x4 里的平移是「绕模型原点旋转」的杠杆臂残差（实测最大 10.7 单位），
        /// 再叠加一次绕骨骼轴的旋转会重复计算、把整个下颌推出去。
        /// </summary>
        public Vector3 LastAxis { get; private set; } = new Vector3(1f, 0f, 0f);

        /// <summary>
        /// 下颌标记点质心的真实位移（A2F 模型坐标系）。这才是「下颌掉下来多少」。
        /// 实测 speak 时 Y ≈ −0.7（脸高约 35 单位），Z ≈ −0.12。
        /// </summary>
        public Vector3 LastCentroidDelta { get; private set; }

        /// <summary>
        /// 原始 Kabsch 平移 t = aMean − R·bMean。含杠杆臂，<b>不要</b>直接拿去驱动骨骼，
        /// 只作诊断用（日志里能看到它比质心位移大一个量级）。
        /// </summary>
        public Vector3 LastTranslation { get; private set; }

        public Audio2FaceJawAnimator(float[] neutralJaw, Params p)
        {
            if (neutralJaw == null || neutralJaw.Length < 9 || neutralJaw.Length % 3 != 0)
            {
                _nbPoints = 0;
                return;
            }
            _neutralJaw = neutralJaw;
            _jawPose = new float[neutralJaw.Length];
            _nbPoints = neutralJaw.Length / 3;
            _params = p;
        }

        public void SetParams(Params p) => _params = p;

        /// <summary>
        /// 算一帧的下颌 4x4 变换。
        /// </summary>
        /// <param name="jawTransform">输出，长度 16，列主序（平移在 12/13/14）。</param>
        /// <param name="prediction">模型输出。</param>
        /// <param name="offset">jaw 段起点（Audio2FaceNetworkInfo.JawOffset + row）。</param>
        public bool ComputeJawTransform(float[] jawTransform, float[] prediction, int offset)
        {
            if (!IsReady || jawTransform == null || jawTransform.Length < 16) return false;
            if (prediction == null || offset < 0 || offset + _neutralJaw.Length > prediction.Length) return false;

            int n = _nbPoints;
            double dx = 0, dy = 0, dz = 0;
            for (int i = 0; i < n; i++)
            {
                float px = prediction[offset + 3 * i + 0] * _params.lowerTeethStrength;
                float py = prediction[offset + 3 * i + 1] * _params.lowerTeethStrength + _params.lowerTeethHeightOffset;
                float pz = prediction[offset + 3 * i + 2] * _params.lowerTeethStrength + _params.lowerTeethDepthOffset;

                dx += px; dy += py; dz += pz;

                _jawPose[3 * i + 0] = px + _neutralJaw[3 * i + 0];
                _jawPose[3 * i + 1] = py + _neutralJaw[3 * i + 1];
                _jawPose[3 * i + 2] = pz + _neutralJaw[3 * i + 2];
            }

            // 质心真实位移 = mean(delta)（a = neutral + delta，b = neutral，所以 aMean − bMean 就是它）
            float inv = 1f / n;
            LastCentroidDelta = new Vector3((float)dx * inv, (float)dy * inv, (float)dz * inv);

            return RigidXform(jawTransform, _jawPose, _neutralJaw, n);
        }

        /// <summary>
        /// Kabsch：求把点集 b 刚体变换到点集 a 的最优 (R, t)。Horn 四元数法。
        /// </summary>
        private bool RigidXform(float[] m, float[] a, float[] b, int n)
        {
            double amx = 0, amy = 0, amz = 0, bmx = 0, bmy = 0, bmz = 0;
            for (int i = 0; i < n; i++)
            {
                amx += a[3 * i]; amy += a[3 * i + 1]; amz += a[3 * i + 2];
                bmx += b[3 * i]; bmy += b[3 * i + 1]; bmz += b[3 * i + 2];
            }
            double inv = 1.0 / n;
            amx *= inv; amy *= inv; amz *= inv;
            bmx *= inv; bmy *= inv; bmz *= inv;

            // H = b_delta^T * a_delta，H[j][k] = Σ_i b_delta[i][j] * a_delta[i][k]
            double sxx = 0, sxy = 0, sxz = 0;
            double syx = 0, syy = 0, syz = 0;
            double szx = 0, szy = 0, szz = 0;
            for (int i = 0; i < n; i++)
            {
                double bx = b[3 * i] - bmx, by = b[3 * i + 1] - bmy, bz = b[3 * i + 2] - bmz;
                double ax = a[3 * i] - amx, ay = a[3 * i + 1] - amy, az = a[3 * i + 2] - amz;
                sxx += bx * ax; sxy += bx * ay; sxz += bx * az;
                syx += by * ax; syy += by * ay; syz += by * az;
                szx += bz * ax; szy += bz * ay; szz += bz * az;
            }

            // Horn 的 4x4 对称矩阵 N（q = [w, x, y, z]，最大特征值对应的特征向量即最优旋转）
            double[,] N = new double[4, 4];
            N[0, 0] = sxx + syy + szz;
            N[0, 1] = N[1, 0] = syz - szy;
            N[0, 2] = N[2, 0] = szx - sxz;
            N[0, 3] = N[3, 0] = sxy - syx;

            N[1, 1] = sxx - syy - szz;
            N[1, 2] = N[2, 1] = sxy + syx;
            N[1, 3] = N[3, 1] = szx + sxz;

            N[2, 2] = -sxx + syy - szz;
            N[2, 3] = N[3, 2] = syz + szy;

            N[3, 3] = -sxx - syy + szz;

            // 幂迭代求主特征向量。旋转对应的特征值 λ_max = Σ 奇异值，与另外三个
            // （±σ 量级）分离良好，下颌这种小旋转也收敛得很快。
            double w = 1, x = 0, y = 0, z = 0;
            for (int it = 0; it < 64; it++)
            {
                double nw = N[0, 0] * w + N[0, 1] * x + N[0, 2] * y + N[0, 3] * z;
                double nx = N[1, 0] * w + N[1, 1] * x + N[1, 2] * y + N[1, 3] * z;
                double ny = N[2, 0] * w + N[2, 1] * x + N[2, 2] * y + N[2, 3] * z;
                double nz = N[3, 0] * w + N[3, 1] * x + N[3, 2] * y + N[3, 3] * z;
                double len = System.Math.Sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
                if (len < 1e-12) { w = 1; x = 0; y = 0; z = 0; break; }
                w = nw / len; x = nx / len; y = ny / len; z = nz / len;
            }

            // 四元数 → 旋转矩阵（右手系，q = w + xi + yj + zk）
            double r00 = 1 - 2 * (y * y + z * z);
            double r01 = 2 * (x * y - w * z);
            double r02 = 2 * (x * z + w * y);
            double r10 = 2 * (x * y + w * z);
            double r11 = 1 - 2 * (x * x + z * z);
            double r12 = 2 * (y * z - w * x);
            double r20 = 2 * (x * z - w * y);
            double r21 = 2 * (y * z + w * x);
            double r22 = 1 - 2 * (x * x + y * y);

            // t = aMean − R·bMean
            double tx = amx - (r00 * bmx + r01 * bmy + r02 * bmz);
            double ty = amy - (r10 * bmx + r11 * bmy + r12 * bmz);
            double tz = amz - (r20 * bmx + r21 * bmy + r22 * bmz);

            // 列主序写出（与 Eigen::Matrix<float,4,4,ColMajor> 一致）
            m[0] = (float)r00; m[1] = (float)r10; m[2] = (float)r20; m[3] = 0f;
            m[4] = (float)r01; m[5] = (float)r11; m[6] = (float)r21; m[7] = 0f;
            m[8] = (float)r02; m[9] = (float)r12; m[10] = (float)r22; m[11] = 0f;
            m[12] = (float)tx; m[13] = (float)ty; m[14] = (float)tz; m[15] = 1f;

            // 幂迭代收敛出的四元数符号不定，统一取 w >= 0（最短弧），避免角度在 θ 与 360−θ 之间跳。
            if (w < 0) { w = -w; x = -x; y = -y; z = -z; }

            double vlen = System.Math.Sqrt(x * x + y * y + z * z);
            if (vlen < 1e-9)
            {
                LastAngleDeg = 0f;
                LastAxis = new Vector3(1f, 0f, 0f);
            }
            else
            {
                LastAngleDeg = (float)(2.0 * System.Math.Atan2(vlen, w) * 180.0 / System.Math.PI);
                LastAxis = new Vector3((float)(x / vlen), (float)(y / vlen), (float)(z / vlen));
            }

            LastTranslation = new Vector3((float)tx, (float)ty, (float)tz);
            return true;
        }
    }
}
