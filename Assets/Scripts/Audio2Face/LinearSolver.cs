using System;

namespace Audio2Face
{
    /// <summary>
    /// 小规模稠密线性代数工具（BVLS 求解器用）。
    /// 矩阵按行主序存放在 float[] 中：A[i * n + j]。
    /// </summary>
    public static class LinearSolver
    {
        /// <summary>
        /// 用 Householder QR 求 min |A x - b| 的最小二乘解。
        /// A: m 行 n 列（要求 m >= n）。A 与 b 会被就地修改。
        /// </summary>
        public static void LeastSquares(float[] A, int m, int n, float[] b, float[] x)
        {
            for (int i = 0; i < x.Length; i++) x[i] = 0f;

            float rankEps = 0f;
            for (int i = 0; i < m * n; i++) rankEps = Math.Max(rankEps, Math.Abs(A[i]));
            rankEps *= 1e-6f;
            if (rankEps <= 0f) rankEps = 1e-12f;

            var v = new float[m];

            for (int k = 0; k < n; k++)
            {
                // 计算第 k 列的范数
                double norm = 0.0;
                for (int i = k; i < m; i++) norm += (double)A[i * n + k] * A[i * n + k];
                norm = Math.Sqrt(norm);

                if (norm < rankEps)
                {
                    // 秩缺失：该列视为 0，回代时对应未知量取 0
                    for (int i = k; i < m; i++) A[i * n + k] = 0f;
                    continue;
                }

                double akk = A[k * n + k];
                double alpha = akk >= 0 ? -norm : norm;

                double vtv = 0.0;
                v[k] = (float)(akk - alpha);
                vtv += (double)v[k] * v[k];
                for (int i = k + 1; i < m; i++)
                {
                    v[i] = A[i * n + k];
                    vtv += (double)v[i] * v[i];
                }

                if (vtv < 1e-30) continue;

                // 对 A 的第 k..n-1 列做变换
                for (int j = k; j < n; j++)
                {
                    double dot = 0.0;
                    for (int i = k; i < m; i++) dot += (double)v[i] * A[i * n + j];
                    double scale = 2.0 * dot / vtv;
                    for (int i = k; i < m; i++) A[i * n + j] -= (float)(scale * v[i]);
                }

                // 对 b 做同样的变换
                double bdot = 0.0;
                for (int i = k; i < m; i++) bdot += (double)v[i] * b[i];
                double bscale = 2.0 * bdot / vtv;
                for (int i = k; i < m; i++) b[i] -= (float)(bscale * v[i]);
            }

            // 回代 R x = b（R 为上三角，前 n 行）
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = b[i];
                for (int j = i + 1; j < n; j++) sum -= (double)A[i * n + j] * x[j];
                double diag = A[i * n + i];
                x[i] = Math.Abs(diag) < rankEps ? 0f : (float)(sum / diag);
            }
        }
    }
}
