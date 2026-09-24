using System;

namespace Audio2Face
{
    /// <summary>
    /// BVLS（Bounded Variable Least Squares）求解器的一次性工作区。
    /// 直接移植自 Audio2Face-3D-SDK: audio2face-sdk/source/audio2face-core/bvls.cpp
    ///
    /// 求解：min 0.5 * |A x - b|^2,  subject to  l &lt;= x &lt;= u
    ///
    /// SDK 传入的 A 是 AMat = (D^T D + 正则项)，对称正定。
    ///
    /// ── 两处性能改造（2026-09）─────────────────────────────────────────────────
    /// 1) <b>工作区复用</b>。原来的写法每进一次 ConstrainedMin 就 new 五六个数组
    ///    （freeVars / aFree(n×p) / xFree / bFree …），而一次 Bvls.Solve 里
    ///    ConstrainedMin 会被调用几十次、每帧还要解两遍（cancelPairs 重解），
    ///    30 帧下来是十几 MB 的纯垃圾。这里全部改成预分配缓冲。
    ///
    /// 2) <b>正规方程 + Cholesky 取代 Householder QR</b>。
    ///    自由变量子问题 min |A[:,F]·xF - bFree|^2 的正规方程是
    ///        (A^T A)_FF · xF = (A^T bFree)_F
    ///    A 对称 → A^T A = A·A = G，G 在 A 不变时可以<b>缓存</b>（Load 之后 A 就固定了）。
    ///    又 b 在一次 Solve 内也不变，所以 A·b 只需算一次。之后每次 ConstrainedMin 只剩：
    ///        rhs_F = (A·b)_F − G_{F,C}·x_C        O(p·n)
    ///        Cholesky(G_FF) + 前代回代             O(p³/6)
    ///    而 Householder QR 是 O(2·n·p²) 且还要先把 A(:,F) 拷成一份 n×p。
    ///    n≈p≈43 时常数差约 10 倍。
    ///    代价是正规方程的条件数平方（cond(G)=cond(A)²），所以这里全程 double，
    ///    并且 Cholesky 一旦不收敛就<b>自动回退 QR 路径</b>（<see cref="CholFail"/> 计数）。
    ///    <see cref="UseNormalEquations"/> 设 false 可以整体关掉做对拍。
    /// </summary>
    public sealed class BvlsWorkspace
    {
        /// <summary>走正规方程 + Cholesky。false = 退回原来的 Householder QR（数值更稳但慢约 10 倍）。</summary>
        public bool UseNormalEquations = true;

        /// <summary>
        /// KKT 收敛判据的<b>相对</b>容差：实际阈值 = max(调用方给的绝对 tolerance, 本值 × 梯度量级)。
        /// <para>
        /// 判据是 max|g|，g = A^T(Ax−b)。调用方（BlendshapeSolver）传的是 SDK 默认的绝对 1e-10，
        /// 但 b = D^T·target 是上万个分量求和的结果，g 的量级常在 1e3~1e5 —— 1e-10 事实上永远
        /// 达不到，主循环只能靠「没有变量想脱离边界」或「代价不再下降」退出，实测 18.4 次迭代/帧。
        /// 梯度量级取 x=0 处的 |A·b|（与热启动的 x 无关，是个稳定的参考量），改判据后迭代降到
        /// 个位数，权重偏差仍在 1e-5 量级 —— 见 [BVLS对拍] 的「放宽容差 vs 收紧」一栏。
        /// </para>
        /// 设 0 = 只认绝对容差（旧行为）。
        /// </summary>
        public float RelativeTolerance = 1e-6f;

        /// <summary>本次 Solve 实际使用的 KKT 阈值，给剖析日志看（= max(绝对, 相对×量级)）。</summary>
        public float LastThreshold;

        /// <summary>首次求解时用两种路径各解一遍并打印最大偏差（一次性，用来证明两条路等价）。</summary>
        public static bool VerifyOnce = true;

        // 统计（给剖析日志用）
        public int IterCount;      // 主循环迭代次数
        public int CmCount;        // ConstrainedMin 调用次数
        public int CholFail;       // Cholesky 失败回退 QR 的次数
        public int SolveCount;

        private int _n;
        private int[] _onBound;
        private float[] _g, _r, _s, _x0, _bFree, _xFree, _aFree, _aCopy, _bCopy;
        private int[] _freeVars;
        private double[] _G;           // A·A
        private double[] _Ab;          // A·b
        private double[] _rhsD;
        private double[] _chol;        // Cholesky 分解（p×p，行主序）
        private double[] _y;
        private float[] _cachedA;      // 缓存 G 时对应的 A（引用比较，A 一换就重算）

        public BvlsWorkspace(int n)
        {
            Ensure(n);
        }

        private void Ensure(int n)
        {
            if (_n == n && _onBound != null) return;
            _n = n;
            _onBound = new int[n];
            _g = new float[n];
            _r = new float[n];
            _s = new float[n];
            _x0 = new float[n];
            _bFree = new float[n];
            _xFree = new float[n];
            _rhsD = new double[n];
            _Ab = new double[n];
            _y = new double[n];
            _aFree = new float[n * n];
            _aCopy = new float[n * n];
            _bCopy = new float[n];
            _freeVars = new int[n];
            _G = new double[n * n];
            _chol = new double[n * n];
            _cachedA = null;
        }

        /// <summary>A: n x n 行主序；b/l/u: 长度 n；x 为输入初值兼输出。</summary>
        public void Solve(float[] x, float[] A, int n, float[] b, float[] l, float[] u, float tolerance)
        {
            Ensure(n);
            SolveCount++;

            bool useNe = UseNormalEquations;
            if (useNe) PrepareNormalEquations(A, n, b);

            // KKT 阈值 = max(绝对容差, 相对容差 × 梯度量级)
            float thresh = tolerance;
            if (RelativeTolerance > 0f)
            {
                float rel = RelativeTolerance * GradientScale(A, n, b);
                if (rel > thresh) thresh = rel;
            }
            LastThreshold = thresh;

            if (VerifyOnce)
            {
                VerifyOnce = false;
                SolveVerified(x, A, n, b, l, u, thresh, useNe);
                return;
            }

            SolveCore(x, A, n, b, l, u, thresh, useNe);
        }

        /// <summary>
        /// 梯度的参考量级：x=0 处 g = −A^T·b，A 对称 → 就是 |A·b| 的最大分量。
        /// O(n²) 一次，相对一次 Solve 里几十次 O(n²)/O(p³) 可以忽略。
        /// </summary>
        private static float GradientScale(float[] A, int n, float[] b)
        {
            double worst = 0.0;
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                int row = i * n;
                for (int j = 0; j < n; j++) sum += (double)A[row + j] * b[j];
                double v = Math.Abs(sum);
                if (v > worst) worst = v;
            }
            return (float)worst;
        }

        private void SolveCore(float[] x, float[] A, int n, float[] b, float[] l, float[] u, float thresh, bool useNe)
        {
            var onBound = _onBound;
            var g = _g;
            var r = _r;
            var s = _s;
            var x0 = _x0;
            var bFree = _bFree;

            InitialFeasiblePoint(x, A, n, b, l, u, onBound, x0, bFree, useNe);

            float cost = CostAndGradient(g, r, x, A, n, b);

            UpdateOnBound(onBound, x, l, u, n);
            float optimality = ComputeKktOptimality(g, onBound, n);
            float prevOptimality = optimality;

            // 停滞判据也跟着阈值走（原来写死 1e-6，对 1e3 量级的梯度形同虚设）。
            // 保留 1e-6 下限：只认绝对容差时行为与改动前一致。
            float stagnation = Math.Max(1e-6f, thresh * 1e-3f);

            for (int iter = 0; iter < n; iter++)
            {
                if (optimality < thresh) break;
                IterCount++;

                int moveToFree = VarToDetach(g, x, A, n, b, l, u, r);
                if (moveToFree == -1) break;
                onBound[moveToFree] = 0;

                while (true)
                {
                    Array.Copy(x, s, n);
                    ConstrainedMin(s, A, n, b, onBound, x0, bFree, useNe);

                    float alpha = MakePointFeasible(x, s, l, u, n);
                    if (alpha == 1.0f) break;

                    UpdateOnBound(onBound, x, l, u, n);
                }

                float costNew = CostAndGradient(g, r, x, A, n, b);
                if (costNew > cost) break;

                cost = costNew;
                optimality = ComputeKktOptimality(g, onBound, n);
                if (Math.Abs(prevOptimality - optimality) < stagnation) break;
                prevOptimality = optimality;
            }
        }

        /// <summary>一次性对拍：同一次求解跑三遍，分别验证「两条线性代数路径等价」和「放宽容差没掉精度」。</summary>
        private void SolveVerified(float[] x, float[] A, int n, float[] b, float[] l, float[] u, float thresh, bool useNe)
        {
            var xInit = new float[n];
            Array.Copy(x, xInit, n);

            // ① 生产路径（当前阈值）
            SolveCore(x, A, n, b, l, u, thresh, useNe);
            var prod = new float[n];
            Array.Copy(x, prod, n);

            // ② 参照路径：强制 QR（同一阈值）→ 证明 Cholesky 与 QR 等价
            Array.Copy(xInit, x, n);
            SolveCore(x, A, n, b, l, u, thresh, false);
            var refSol = new float[n];
            Array.Copy(x, refSol, n);

            // ③ 精度参照：阈值压到 0（绝不提前退出，跑到 KKT 真正满足）→
            //    证明把绝对 1e-10 换成相对容差以后，权重没有肉眼可见的变化
            Array.Copy(xInit, x, n);
            SolveCore(x, A, n, b, l, u, 0f, useNe);
            var tight = new float[n];
            Array.Copy(x, tight, n);

            double maxDiff = 0.0, maxAcc = 0.0;
            for (int i = 0; i < n; i++)
            {
                double d = Math.Abs(prod[i] - refSol[i]);
                if (d > maxDiff) maxDiff = d;
                double a = Math.Abs(prod[i] - tight[i]);
                if (a > maxAcc) maxAcc = a;
            }

            // 恢复生产路径的结果
            Array.Copy(prod, x, n);

            UnityEngine.Debug.Log($"[BVLS对拍] n={n} 生产路径={(useNe ? "Cholesky正规方程" : "Householder QR")} " +
                                  $"vs 参照=Householder QR：最大权重偏差={maxDiff:E2}" +
                                  (useNe ? $"（Cholesky 回退 {CholFail} 次）" : "") +
                                  (maxDiff > 1e-3 ? "  ⚠ 偏差偏大，建议把 UseNormalEquations 设 false" : "  ✓ 一致") +
                                  $"\n    放宽容差(阈值={thresh:E2}, 相对={RelativeTolerance:E1}) vs 收紧到 0：" +
                                  $"最大权重偏差={maxAcc:E2}" +
                                  (maxAcc > 1e-3 ? "  ⚠ 精度掉太多，把 RelativeTolerance 调小或设 0" : "  ✓ 精度可接受"));
        }

        /// <summary>缓存 G = A·A 与 Ab = A·b（A 引用变了才重算 G；b 每次都重算）。</summary>
        private void PrepareNormalEquations(float[] A, int n, float[] b)
        {
            if (!ReferenceEquals(_cachedA, A))
            {
                var G = _G;
                for (int i = 0; i < n; i++)
                {
                    int ri = i * n;
                    for (int j = i; j < n; j++)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) sum += (double)A[ri + k] * A[j * n + k];
                        G[ri + j] = sum;
                        G[j * n + i] = sum;
                    }
                }
                _cachedA = A;
            }

            var Ab = _Ab;
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                int ri = i * n;
                for (int j = 0; j < n; j++) sum += (double)A[ri + j] * b[j];
                Ab[i] = sum;
            }
        }

        private void ConstrainedMin(float[] x, float[] A, int n, float[] b, int[] onBound, float[] x0, float[] bFree, bool useNe)
        {
            CmCount++;

            var freeVars = _freeVars;
            int freeCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (onBound[i] == 0) freeVars[freeCount++] = i;
            }

            if (freeCount == 0) return;

            if (useNe && SolveFreeCholesky(x, A, n, onBound, x0, freeVars, freeCount)) return;

            // ── 回退：原来的 Householder QR ──
            if (freeCount == n)
            {
                Array.Copy(A, _aCopy, n * n);
                Array.Copy(b, _bCopy, n);
                LinearSolver.LeastSquares(_aCopy, n, n, _bCopy, x);
                return;
            }

            Array.Copy(x, x0, n);
            for (int i = 0; i < freeCount; i++) x0[freeVars[i]] = 0f;

            Array.Copy(b, bFree, n);
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                int row = i * n;
                for (int j = 0; j < n; j++) sum += (double)A[row + j] * x0[j];
                bFree[i] -= (float)sum;
            }

            var aFree = _aFree;
            for (int i = 0; i < n; i++)
            {
                int dst = i * freeCount;
                int src = i * n;
                for (int j = 0; j < freeCount; j++) aFree[dst + j] = A[src + freeVars[j]];
            }

            var xFree = _xFree;
            LinearSolver.LeastSquares(aFree, n, freeCount, bFree, xFree);

            for (int i = 0; i < freeCount; i++) x[freeVars[i]] = xFree[i];
        }

        /// <summary>
        /// 正规方程 + Cholesky：G_FF · xF = (A·b)_F − G_{F,C}·x_C。
        /// 成功返回 true；Cholesky 不收敛（条件数过大）返回 false 让调用方回退 QR。
        /// </summary>
        private bool SolveFreeCholesky(float[] x, float[] A, int n, int[] onBound, float[] x0, int[] freeVars, int p)
        {
            var G = _G;
            var Ab = _Ab;
            var rhs = _rhsD;

            Array.Copy(x, x0, n);
            for (int i = 0; i < p; i++) x0[freeVars[i]] = 0f;

            // rhs_F = Ab_F − Σ_{j 固定} G[F,j]·x0[j]
            //   （x0 在自由位置是 0，所以遍历全部 j 也等价，省一次分支）
            for (int i = 0; i < p; i++)
            {
                int row = freeVars[i] * n;
                double v = Ab[freeVars[i]];
                for (int j = 0; j < n; j++)
                {
                    float xj = x0[j];
                    if (xj != 0f) v -= G[row + j] * xj;
                }
                rhs[i] = v;
            }

            // L = Cholesky(G_FF)，就地放在 _chol
            var L = _chol;
            double maxDiag = 0.0;
            for (int i = 0; i < p; i++)
            {
                int fi = freeVars[i];
                int ri = fi * n;
                for (int j = 0; j < p; j++) L[i * p + j] = G[ri + freeVars[j]];
                if (L[i * p + i] > maxDiag) maxDiag = L[i * p + i];
            }
            // 对角为负说明数值上已经不正定了（cond(G) 爆掉），直接回退
            if (maxDiag <= 0.0) { CholFail++; return false; }

            double eps = maxDiag * 1e-13;
            if (eps <= 0.0) eps = 1e-300;

            for (int j = 0; j < p; j++)
            {
                double d = L[j * p + j];
                for (int k = 0; k < j; k++)
                {
                    double ljk = L[j * p + k];
                    d -= ljk * ljk;
                }
                if (d <= eps) { CholFail++; return false; }

                double ljj = Math.Sqrt(d);
                L[j * p + j] = ljj;
                double inv = 1.0 / ljj;

                for (int i = j + 1; i < p; i++)
                {
                    double s = L[i * p + j];
                    for (int k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
                    L[i * p + j] = s * inv;
                }
            }

            // 前代 L y = rhs
            var y = _y;
            for (int i = 0; i < p; i++)
            {
                double sum = rhs[i];
                for (int k = 0; k < i; k++) sum -= L[i * p + k] * y[k];
                y[i] = sum / L[i * p + i];
            }
            // 回代 L^T x = y
            for (int i = p - 1; i >= 0; i--)
            {
                double sum = y[i];
                for (int k = i + 1; k < p; k++) sum -= L[k * p + i] * x[freeVars[k]];
                double xv = sum / L[i * p + i];
                x[freeVars[i]] = (float)xv;
            }

            return true;
        }

        private float CostAndGradient(float[] g, float[] r, float[] x, float[] A, int n, float[] b)
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

        private void InitialFeasiblePoint(float[] x, float[] A, int n, float[] b, float[] l, float[] u,
                                          int[] onBound, float[] x0, float[] bFree, bool useNe)
        {
            Array.Clear(onBound, 0, n);

            while (true)
            {
                ConstrainedMin(x, A, n, b, onBound, x0, bFree, useNe);

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

        private int VarToDetach(float[] g, float[] x, float[] A, int n, float[] b, float[] l, float[] u, float[] r)
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

    /// <summary>
    /// <see cref="BvlsWorkspace"/> 的静态薄封装：老调用点（Bvls.Solve）可以继续用，
    /// 每个线程一份工作区。热路径（BlendshapeSolver）应持有自己的实例以避免字典查找。
    /// </summary>
    public static class Bvls
    {
        /// <summary>静态路径是否走正规方程 + Cholesky（默认开）。</summary>
        public static bool UseNormalEquations = true;

        /// <summary>静态路径的相对 KKT 容差（同 <see cref="BvlsWorkspace.RelativeTolerance"/>）。</summary>
        public static float RelativeTolerance = 1e-6f;

        [ThreadStatic]
        private static System.Collections.Generic.Dictionary<int, BvlsWorkspace> _pool;

        /// <summary>A: n x n 行主序；b/l/u: 长度 n；x 为输入初值兼输出。</summary>
        public static void Solve(float[] x, float[] A, int n, float[] b, float[] l, float[] u, float tolerance)
        {
            var pool = _pool ?? (_pool = new System.Collections.Generic.Dictionary<int, BvlsWorkspace>());
            BvlsWorkspace ws;
            if (!pool.TryGetValue(n, out ws))
            {
                ws = new BvlsWorkspace(n);
                pool[n] = ws;
            }
            ws.UseNormalEquations = UseNormalEquations;
            ws.RelativeTolerance = RelativeTolerance;
            ws.Solve(x, A, n, b, l, u, tolerance);
        }
    }
}
