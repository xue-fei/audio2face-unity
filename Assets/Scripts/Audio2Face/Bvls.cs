using System;

namespace Audio2Face
{
    /// <summary>
    /// BVLS（Bounded Variable Least Squares）求解器。
    /// 直接移植自 Audio2Face-3D-SDK: audio2face-sdk/source/audio2face-core/bvls.cpp
    ///
    /// 求解：min 0.5 * |A x - b|^2,  subject to  l &lt;= x &lt;= u
    ///
    /// 注意：SDK 传入的 A 是 AMat = (D^T D + 正则项)，它是对称正定矩阵。
    /// 对对称正定 M，min |M x - b|^2 的无约束解恰好是 x = M^-1 b，
    /// 与正则化最小二乘的解一致，所以这里按 SDK 原样使用 A，不做 Cholesky 分解。
    /// </summary>
    public static class Bvls
    {
        /// <summary>A: n x n 行主序；b/l/u: 长度 n；x 为输入初值兼输出。</summary>
        public static void Solve(float[] x, float[] A, int n, float[] b, float[] l, float[] u, float tolerance)
        {
            var onBound = new int[n];
            var g = new float[n];
            var r = new float[n];
            var s = new float[n];
            var x0 = new float[n];
            var bFree = new float[n];

            InitialFeasiblePoint(x, A, n, b, l, u, onBound, x0, bFree);

            float cost = CostAndGradient(g, r, x, A, n, b);

            UpdateOnBound(onBound, x, l, u, n);
            float optimality = ComputeKktOptimality(g, onBound, n);
            float prevOptimality = optimality;

            for (int iter = 0; iter < n; iter++)
            {
                if (optimality < tolerance) break;

                int moveToFree = VarToDetach(g, x, A, n, b, l, u, r);
                if (moveToFree == -1) break;
                onBound[moveToFree] = 0;

                while (true)
                {
                    Array.Copy(x, s, n);
                    ConstrainedMin(s, A, n, b, onBound, x0, bFree);

                    float alpha = MakePointFeasible(x, s, l, u, n);
                    if (alpha == 1.0f) break;

                    UpdateOnBound(onBound, x, l, u, n);
                }

                float costNew = CostAndGradient(g, r, x, A, n, b);
                if (costNew > cost) break;

                cost = costNew;
                optimality = ComputeKktOptimality(g, onBound, n);
                if (Math.Abs(prevOptimality - optimality) < 1e-6f) break;
                prevOptimality = optimality;
            }
        }

        private static void ConstrainedMin(float[] x, float[] A, int n, float[] b, int[] onBound, float[] x0, float[] bFree)
        {
            int freeCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (onBound[i] == 0) freeCount++;
            }

            if (freeCount == 0) return;

            if (freeCount == n)
            {
                // 完全无约束，直接解 A x = b
                var acopy = new float[n * n];
                Array.Copy(A, acopy, n * n);
                var bcopy = new float[n];
                Array.Copy(b, bcopy, n);
                LinearSolver.LeastSquares(acopy, n, n, bcopy, x);
                return;
            }

            // x0 = x，但把自由分量清零
            Array.Copy(x, x0, n);
            var freeVars = new int[freeCount];
            int c = 0;
            for (int i = 0; i < n; i++)
            {
                if (onBound[i] == 0)
                {
                    freeVars[c++] = i;
                    x0[i] = 0f;
                }
            }

            // bFree = b - A * x0
            Array.Copy(b, bFree, n);
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                int row = i * n;
                for (int j = 0; j < n; j++) sum += (double)A[row + j] * x0[j];
                bFree[i] -= (float)sum;
            }

            // A_free = A(:, freeVars)，尺寸 n x freeCount
            var aFree = new float[n * freeCount];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < freeCount; j++)
                {
                    aFree[i * freeCount + j] = A[i * n + freeVars[j]];
                }
            }

            var xFree = new float[freeCount];
            LinearSolver.LeastSquares(aFree, n, freeCount, bFree, xFree);

            for (int i = 0; i < freeCount; i++) x[freeVars[i]] = xFree[i];
        }

        private static float CostAndGradient(float[] g, float[] r, float[] x, float[] A, int n, float[] b)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                int row = i * n;
                for (int j = 0; j < n; j++) sum += (double)A[row + j] * x[j];
                r[i] = (float)sum - b[i];
            }

            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < n; i++) sum += (double)A[i * n + j] * r[i];
                g[j] = (float)sum;
            }

            double cost = 0.0;
            for (int i = 0; i < n; i++) cost += (double)r[i] * r[i];
            return (float)(0.5 * cost);
        }

        private static void InitialFeasiblePoint(float[] x, float[] A, int n, float[] b, float[] l, float[] u,
                                                 int[] onBound, float[] x0, float[] bFree)
        {
            Array.Clear(onBound, 0, n);

            while (true)
            {
                ConstrainedMin(x, A, n, b, onBound, x0, bFree);

                int foundActives = 0;
                for (int i = 0; i < n; i++)
                {
                    if (onBound[i] != 0) continue;
                    if (x[i] < l[i])
                    {
                        x[i] = l[i];
                        onBound[i] = -1;
                        foundActives++;
                    }
                    else if (x[i] > u[i])
                    {
                        x[i] = u[i];
                        onBound[i] = 1;
                        foundActives++;
                    }
                }

                if (foundActives == 0) break;
            }
        }

        private static float ComputeKktOptimality(float[] g, int[] onBound, int n)
        {
            float worst = 0f;
            for (int i = 0; i < n; i++)
            {
                float v = onBound[i] == 0 ? Math.Abs(g[i]) : g[i] * onBound[i];
                if (v > worst) worst = v;
            }
            return worst;
        }

        private static int VarToDetach(float[] g, float[] x, float[] A, int n, float[] b, float[] l, float[] u, float[] r)
        {
            CostAndGradient(g, r, x, A, n, b);

            int bestIdx = -1;
            float bestVal = 0f;
            for (int i = 0; i < n; i++)
            {
                if (x[i] == l[i])
                {
                    if (g[i] < 0f)
                    {
                        float val = -g[i];
                        if (val > bestVal)
                        {
                            bestVal = val;
                            bestIdx = i;
                        }
                    }
                }
                else if (x[i] == u[i])
                {
                    if (g[i] > 0f)
                    {
                        float val = g[i];
                        if (val > bestVal)
                        {
                            bestVal = val;
                            bestIdx = i;
                        }
                    }
                }
            }
            return bestIdx;
        }

        private static float MakePointFeasible(float[] x, float[] s, float[] l, float[] u, int n)
        {
            float alpha = 1.0f;
            int idx = -1;
            float boundValue = 0f;

            for (int i = 0; i < n; i++)
            {
                if (x[i] == s[i]) continue;

                if (s[i] < l[i])
                {
                    float alphaTest = (x[i] - l[i]) / (x[i] - s[i]);
                    if (alphaTest < alpha)
                    {
                        idx = i;
                        boundValue = l[i];
                        alpha = alphaTest;
                    }
                }
                else if (s[i] > u[i])
                {
                    float alphaTest = (u[i] - x[i]) / (s[i] - x[i]);
                    if (alphaTest < alpha)
                    {
                        idx = i;
                        boundValue = u[i];
                        alpha = alphaTest;
                    }
                }
            }

            for (int i = 0; i < n; i++) x[i] += alpha * (s[i] - x[i]);
            if (idx != -1) x[idx] = boundValue;
            return alpha;
        }

        private static void UpdateOnBound(int[] onBound, float[] x, float[] l, float[] u, int n)
        {
            for (int i = 0; i < n; i++)
            {
                if (x[i] == l[i]) onBound[i] = -1;
                else if (x[i] == u[i]) onBound[i] = 1;
                else onBound[i] = 0;
            }
        }
    }
}
