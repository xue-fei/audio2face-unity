using System;
using System.Collections.Generic;
using System.Numerics;
using UnityEngine;

namespace Audio2Face
{
    [Serializable]
    internal class BlendshapeConfigFile
    {
        public BlendshapeParams blendshape_params;
    }

    [Serializable]
    internal class BlendshapeParams
    {
        public float strengthL2regularization;
        public float strengthTemporalSmoothing;
        public float strengthL1regularization;
        public float strengthSymmetry;
        public int numPoses;
        public int[] bsSolveActivePoses;
        public int[] bsSolveCancelPoses;
        public int[] bsSolveSymmetryPoses;
        public float[] bsWeightMultipliers;
        public float[] bsWeightOffsets;
        public float templateBBSize;
    }

    /// <summary>
    /// 从几何顶点反解 blendshape 权重。
    /// 移植自 Audio2Face-3D-SDK:
    ///   blendshape_solver_base.cpp (Prepare / ComputeAMat)
    ///   blendshape_solver.cpp      (Solve)
    ///   bvls.cpp                   (BVLS)
    ///
    /// 流程（每帧）：
    ///   1. target = masked(prediction) - masked(neutral)
    ///   2. b = D^T * target + temporalReg * scaleFactor * prevWeights
    ///   3. BVLS: min |AMat x - b|^2, 0 &lt;= x &lt;= 1
    ///   4. cancelPairs 收紧上界后重解一次
    ///   5. 映射回全部 pose，乘 multipliers 加 offsets
    /// </summary>
    public sealed class BlendshapeSolver
    {
        public int NumPoses { get; private set; }
        public string[] PoseNames { get; private set; }
        public bool IsReady { get; private set; }

        /// <summary>
        /// 官方 stylization 对 json 参数的覆盖（Audio2Face-3D-Samples/configs/*_diffusion_stylization_config.yaml）。
        /// 这些覆盖在 Load 时自动应用，确保求解器行为与 NVIDIA 官方完整服务一致。
        /// 关键差异（只此一处）：
        ///   skin: jawLeft/jawRight/mouthLeft/mouthRight 在 json 里 active=0，但官方 yaml 里 active=1。
        ///         不激活它们会导致下巴/嘴角左右动的能量被错误分配到 jawOpen/mouthLeft 等 pose，
        ///         是「嘴歪眼斜」的根因之一。
        ///   multiplier/offset 不做覆盖：官方 yaml 与 SDK json 里都是 1.0 / 0.0。
        /// </summary>
        private static readonly Dictionary<string, bool> SkinActiveOverrides = new Dictionary<string, bool>
        {
            ["jawLeft"] = true,
            ["jawRight"] = true,
            ["mouthLeft"] = true,
            ["mouthRight"] = true,
        };

        private static readonly Dictionary<string, bool> TongueActiveOverrides = new Dictionary<string, bool>();

        // 注意：官方 Audio2Face-3D-Samples/configs/*_diffusion_stylization_config.yaml 里的
        // weight_multipliers 全部是 1.0、weight_offsets 全部是 0.0，与 SDK 库层的
        // bs_skin_config_*.json（bsWeightMultipliers 全 1.0）完全一致。
        //
        // 之前这里硬编码过一套非 1.0 的 multiplier（mouthClose=0.3、jawForward=0.7、
        // mouthStretch=0.05 等），声称来自「微服务文档」，但那套值与官方 Samples 仓库里的
        // yaml 完全矛盾，直接把闭嘴压到 30%、嘴角横向拉伸压到 5%，正是「口型不正常」的根因。
        // 现在清空，让 multiplier/offset 严格按官方配置走（即不覆盖，保持 1.0 / 0.0）。
        private static readonly Dictionary<string, float> SkinMultOverrides = new Dictionary<string, float>();
        private static readonly Dictionary<string, float> SkinOffsetOverrides = new Dictionary<string, float>();

        private static readonly Dictionary<string, float> TongueMultOverrides = new Dictionary<string, float>();
        private static readonly Dictionary<string, float> TongueOffsetOverrides = new Dictionary<string, float>();

        /// <summary>true = 输入几何是相对 neutral 的位移（模型原始输出）；false = 绝对坐标（SDK animator 之后）。</summary>
        public bool TargetIsDelta { get; set; }

        /// <summary>上一次 Solve 的 target 向量（长度 SolvedPositionCount），用于诊断。</summary>
        public float[] LastTarget => _target;

        /// <summary>上一次 Solve 的 target 的 RMS，用来判断量级是否正常。</summary>
        public float LastTargetRms { get; private set; }

        /// <summary>
        /// 可选的静态偏置（长度 SolvedPositionCount），Solve 时先从 target 里减掉。
        /// 用于消除「模型的 neutral」与「npz 里的 neutral」不一致带来的常量偏置：
        /// 这种偏置会让求解器在静音时也硬凑出非零权重，并抬高残差。
        /// </summary>
        public float[] TargetBias
        {
            get { return _targetBias; }
            set { _targetBias = (value != null && value.Length >= _numPositions) ? value : null; }
        }

        /// <summary>掩码后的 neutral（长度 SolvedPositionCount），诊断用。</summary>
        public float[] NeutralMasked => _neutral;

        /// <summary>
        /// 静音基线权重（长度 NumPoses），null = 不启用。
        ///
        /// 在<b>权重空间</b>而不是顶点空间做静音标定，是因为实测顶点空间减偏置无效：
        /// 静音时模型输出的那张「静息脸」有 ~25% 能量落在 blendshape 张成的子空间之外，
        /// 而且这部分还在缓慢漂移（中心 30 帧 RMS 在 0.134~0.244 之间摆动）。
        /// 减掉顶点均值只会削弱这个漂移分量，BVLS 该解出来的权重一个不少
        /// （实测：去偏置后 target RMS 从 0.252 掉到 0.065，jawOpen 反而从 0.251 涨到 0.251→0.62 量级）。
        ///
        /// 权重空间直接减就一定能归零：静音时 w ≈ b，减完就是 0。
        /// </summary>
        public float[] WeightBaseline
        {
            get { return _weightBaseline; }
            set { _weightBaseline = (value != null && value.Length >= NumPoses) ? value : null; }
        }

        /// <summary>
        /// 减掉基线后是否按 (1 - b) 重新归一化，把可用区间从 [b, 1] 拉回 [0, 1]。
        /// 默认 false：官方 SDK 不做静息脸标定、也没有这个重归一化。之前默认 true 时，
        /// 当某 pose 基线 b 接近 1，1-b 趋近 0，会把微小的残差误差放大成巨大权重，
        /// 是「嘴歪眼斜」的放大来源之一。需要标定静息脸时，直接用 w-b（不重归一化）更安全。
        /// </summary>
        public bool BaselineRescale { get; set; } = false;

        /// <summary>
        /// 对同一帧布局的若干帧求「掩码后」的逐分量平均，用来算静态偏置。
        /// </summary>
        /// <param name="geometry">整个 prediction</param>
        /// <param name="offset">起始帧的 skin 段起点</param>
        /// <param name="stride">每帧维度（TotalDim）</param>
        public void ComputeMaskedMean(float[] geometry, int offset, int stride, int frames, float[] dst)
        {
            if (dst == null || dst.Length < _numPositions) return;
            Array.Clear(dst, 0, _numPositions);
            if (geometry == null || frames <= 0) return;

            for (int f = 0; f < frames; f++)
            {
                int baseIdx = offset + f * stride;
                for (int i = 0; i < _numPositions; i++)
                {
                    int src = baseIdx + (_maskPositions != null ? _maskPositions[i] : i);
                    dst[i] += geometry[src];
                }
            }
            float inv = 1f / frames;
            for (int i = 0; i < _numPositions; i++) dst[i] *= inv;
        }

        /// <summary>
        /// 上一次解的相对残差 ||target - D*x|| / ||target||。
        /// 接近 0 = 这帧几何确实能用 blendshape 线性组合表示；
        /// 接近 1 = 输入根本不是人脸形变（量级或语义弄错了，比如多减了一次 neutral）。
        /// </summary>

        /// <summary>
        /// 纯恒等检验开关：true 时 BVLS 用不带任何正则的 A = D^T·D。
        /// 把某个 pose 的 delta 原样喂进去应当解出该 pose = 1.000、残差 ≈ 0。
        /// 这一条只验证「掩码 + D^T·D + BVLS」；默认求解路径一定带正则，不能拿它当生产判据。
        /// </summary>
        public bool DisableRegularization
        {
            get { return _disableReg; }
            set { _disableReg = value && _aMatNoReg != null; }
        }

        /// <summary>该 pose 的 delta 范数 |d|（活跃空间内）。用来挑自检样本：|d| 太小的 pose 会被正则压扁。</summary>
        public float PoseDeltaNorm(int poseIndex)
        {
            if (!IsReady || poseIndex < 0 || poseIndex >= NumPoses) return 0f;
            int ai = _activeIndexMap[poseIndex];
            if (ai < 0) return 0f;
            double s = 0.0;
            int row = ai * _numPositions;
            for (int i = 0; i < _numPositions; i++) s += (double)_deltas[row + i] * _deltas[row + i];
            return (float)Math.Sqrt(s);
        }

        /// <summary>对角正则项之和（L2*10*sf + Temporal*100*sf），即单 pose 自解时被压扁的分母增量。</summary>
        public float DiagonalRegularization => _diagReg;

        public float ComputeRelativeResidual()
        {
            int k = _b != null ? _b.Length : 0;
            if (k == 0 || _numPositions == 0) return 1f;

            double num = 0.0, den = 0.0;
            for (int i = 0; i < _numPositions; i++)
            {
                double s = 0.0;
                for (int j = 0; j < k; j++) s += (double)_deltas[j * _numPositions + i] * _x[j];
                double r = (double)_target[i] - s;
                num += r * r;
                den += (double)_target[i] * _target[i];
            }
            return den > 0.0 ? (float)Math.Sqrt(num / den) : 0f;
        }

        private int _numPositions;              // P：参与求解的顶点分量数
        private int[] _maskPositions;           // P：每个求解分量在全量几何里的下标；null = 不掩码
        private float[] _neutral;               // P（已掩码）
        private float[] _deltas;                // K * P，pose 主序
        private float[] _aMat;                  // K * K（含全部正则）
        private float[] _aMatNoReg;             // K * K（纯 D^T·D，仅供自检）
        private bool _disableReg;
        private float _diagReg;                 // L2*10*sf + Temporal*100*sf
        private float[] _b;                     // K
        private float[] _x;                     // K
        private float[] _prevWeights;           // K
        private float[] _lower;                 // K
        private float[] _upper;                 // K
        private float[] _multipliers;           // NumPoses
        private float[] _offsets;               // NumPoses
        private int[] _activePoses;             // NumPoses
        private int[] _symmetryPoses;           // NumPoses，<0 表示无对称伙伴
        private List<KeyValuePair<int, int>> _cancelPairs;
        private int[] _activeIndexMap;          // NumPoses -> K 内索引，-1 表示非活跃

        private float _scaleFactor;
        private float _temporalReg;
        private float _tolerance;
        private bool _useMask;
        private readonly string _stylingKind;

        private float[] _target;                // P
        private float[] _targetBias;            // P，可选静态偏置
        private float[] _weightBaseline;        // NumPoses，可选静音基线
        private float[] _result;                // NumPoses

        // BVLS 工作区（复用临时缓冲 + 缓存正规方程矩阵）。
        // 每个 solver 一份：A 矩阵在 Load 之后就不变了，G = A·A 可以一直缓存着。
        private BvlsWorkspace _bvls;

    /// <param name="npzPath">bs_skin_Mark.npz 之类的完整路径</param>
    /// <param name="configPath">bs_skin_config_Mark.json 之类的完整路径</param>
    /// <param name="useMask">是否使用 npz 里的 frontalMask（只解正面顶点，快 5 倍以上）</param>
    /// <param name="tolerance">BVLS 收敛容差</param>
    /// <param name="targetIsDelta">
    /// 传进来的几何是「位移」还是「绝对坐标」。
    /// SDK 里 animator 会先做 pose = neutral + prediction（见 animator_cuda.cu 的
    /// AnimatorSkinComposeKernelStep2 / AnimatorTongueOffsetKernel），solver 再减 neutral；
    /// 我们不跑 animator，直接喂模型原始输出，所以它是 delta，不能减 neutral。
    /// </param>
    public BlendshapeSolver(string npzPath, string configPath, bool useMask = true,
                            float tolerance = 1e-10f, bool targetIsDelta = true,
                            string stylingKind = null)
    {
        _tolerance = tolerance;
        _useMask = useMask;
        TargetIsDelta = targetIsDelta;
        _stylingKind = stylingKind;
        IsReady = false;

        try
        {
            Load(npzPath, configPath);
            IsReady = true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"[BlendshapeSolver] 初始化失败: {ex.Message}");
        }
    }

        private void Load(string npzPath, string configPath)
        {
            var configText = System.IO.File.ReadAllText(configPath);
            var cfg = JsonUtility.FromJson<BlendshapeConfigFile>(configText);
            if (cfg == null || cfg.blendshape_params == null)
                throw new Exception($"配置文件解析失败: {configPath}");

            var p = cfg.blendshape_params;
            NumPoses = p.numPoses;
            _temporalReg = p.strengthTemporalSmoothing;

            using (var npz = NpzFile.Load(npzPath))
            {
                var neutralFull = npz.GetFloats("neutral");
                if (neutralFull == null)
                    throw new Exception($"{npzPath} 缺少 neutral.npy");

                int totalPositions = neutralFull.Length;

                // poseNames[0] 是 "neutral"，真正 pose 从 1 开始
                var allNames = npz.GetStrings("poseNames") ?? new string[0];
                PoseNames = new string[NumPoses];
                for (int i = 0; i < NumPoses; i++)
                {
                    int src = allNames.Length > NumPoses ? i + 1 : i;
                    PoseNames[i] = src < allNames.Length ? allNames[src] : $"pose{i}";
                }

                // JSON 里的数组（bsSolveActivePoses / bsSolveSymmetryPoses / bsWeightMultipliers /
                // bsWeightOffsets）本身就是按 npz poseNames 顺序排列的（对照 SDK parse_helper.cpp：
                // 读进来直接按 index 用，无任何 remap；SetBlendshapeConfig → SetActivePoses 里
                // activePoses[i] 直接对应 npz poseNames[i]）。
                // 之前误以为 JSON 是 ARKit 52 顺序加了一层 RemapJsonArrays，结果把正确的顺序
                // 打乱：eyeLook 系列被错误激活（眼睛上下抖动）、嘴部 pose 能量错位（嘴张得过大）。
                _activePoses = p.bsSolveActivePoses;
                _symmetryPoses = p.bsSolveSymmetryPoses;
                _multipliers = p.bsWeightMultipliers;
                _offsets = p.bsWeightOffsets;

                // 官方 stylization 覆盖必须在 PoseNames 确定后、活跃列表使用前执行
                ApplyStylingOverrides(p);

                // 顶点掩码（frontalMask 存的是顶点索引，每个顶点 3 个分量）
                int[] maskPositions = null;
                if (_useMask && npz.Has("frontalMask"))
                {
                    var vertMask = npz.GetInts("frontalMask");
                    if (vertMask != null && vertMask.Length > 0)
                    {
                        int numVertex = totalPositions / 3;
                        for (int i = 0; i < vertMask.Length; i++)
                        {
                            if (vertMask[i] < 0 || vertMask[i] >= numVertex)
                                throw new Exception($"frontalMask 下标越界: {vertMask[i]} (顶点数 {numVertex})");
                        }

                        maskPositions = new int[vertMask.Length * 3];
                        for (int i = 0; i < vertMask.Length; i++)
                        {
                            maskPositions[3 * i + 0] = 3 * vertMask[i] + 0;
                            maskPositions[3 * i + 1] = 3 * vertMask[i] + 1;
                            maskPositions[3 * i + 2] = 3 * vertMask[i] + 2;
                        }
                    }
                }

                _maskPositions = maskPositions;
                _numPositions = maskPositions != null ? maskPositions.Length : totalPositions;

                // neutral（掩码后）
                _neutral = new float[_numPositions];
                if (maskPositions != null)
                {
                    for (int i = 0; i < _numPositions; i++) _neutral[i] = neutralFull[maskPositions[i]];
                }
                else
                {
                    Array.Copy(neutralFull, _neutral, _numPositions);
                }

                // 活跃 pose 列表（配置缺失时视为全部活跃）
                if (_activePoses == null || _activePoses.Length < NumPoses)
                {
                    var all = new int[NumPoses];
                    for (int i = 0; i < NumPoses; i++) all[i] = 1;
                    _activePoses = all;
                }

                var activeIdx = new List<int>();
                _activeIndexMap = new int[NumPoses];
                for (int i = 0; i < NumPoses; i++)
                {
                    bool active = i < _activePoses.Length && _activePoses[i] != 0;
                    _activeIndexMap[i] = active ? activeIdx.Count : -1;
                    if (active) activeIdx.Add(i);
                }
                if (activeIdx.Count == 0)
                    throw new Exception("没有活跃的 blendshape pose");

                int k = activeIdx.Count;

                // delta poses：npz 里的 pose 数组本身就是位移量（不是绝对坐标），
                // 实测 eyeBlinkLeft RMS=0.038 而 neutral RMS=88.9，可确认无需再减 neutral。
                _deltas = new float[k * _numPositions];
                for (int j = 0; j < k; j++)
                {
                    var pose = npz.GetFloats(PoseNames[activeIdx[j]]);
                    if (pose == null)
                        throw new Exception($"{npzPath} 缺少 {PoseNames[activeIdx[j]]}.npy");

                    int dst = j * _numPositions;
                    if (maskPositions != null)
                    {
                        for (int i = 0; i < _numPositions; i++) _deltas[dst + i] = pose[maskPositions[i]];
                    }
                    else
                    {
                        Array.Copy(pose, 0, _deltas, dst, _numPositions);
                    }
                }

                // scaleFactor：基于 neutral 包围盒对角线与模板包围盒之比的平方
                float targetBBSize = BoundingBoxDiagonal(neutralFull);
                float templateBB = p.templateBBSize > 0f ? p.templateBBSize : 1f;
                _scaleFactor = (targetBBSize / templateBB) * (targetBBSize / templateBB);

                // AMat = ATA + L1^2*0.25*sf*Ones + L2*10*sf*I + Temporal*100*sf*I + Symmetry*10*sf*(S^T S)
                var aMat = new float[k * k];
                for (int a = 0; a < k; a++)
                {
                    int ra = a * _numPositions;
                    for (int b2 = a; b2 < k; b2++)
                    {
                        int rb = b2 * _numPositions;
                        double sum = 0.0;
                        for (int i = 0; i < _numPositions; i++) sum += (double)_deltas[ra + i] * _deltas[rb + i];
                        aMat[a * k + b2] = (float)sum;
                        aMat[b2 * k + a] = (float)sum;
                    }
                }
                // 自检用的无正则版本（必须在加正则之前留一份）
                _aMatNoReg = (float[])aMat.Clone();

                float sf = _scaleFactor;
                float l1Term = p.strengthL1regularization * p.strengthL1regularization * 0.25f * sf;
                float l2Term = p.strengthL2regularization * 10.0f * sf;
                float temporalTerm = p.strengthTemporalSmoothing * 100.0f * sf;
                float symmetryTerm = p.strengthSymmetry * 10.0f * sf;

                for (int a = 0; a < k; a++)
                {
                    for (int b2 = 0; b2 < k; b2++) aMat[a * k + b2] += l1Term;
                    aMat[a * k + a] += l2Term + temporalTerm;
                }
                _diagReg = l2Term + temporalTerm;

                // 对称正则：symMat 每行为 (+1 at i, -1 at j)
                var symmetryPairs = BuildPairs(p.bsSolveSymmetryPoses, activeIdx, _activeIndexMap);
                foreach (var pair in symmetryPairs)
                {
                    aMat[pair.Key * k + pair.Key] += symmetryTerm;
                    aMat[pair.Value * k + pair.Value] += symmetryTerm;
                    aMat[pair.Key * k + pair.Value] -= symmetryTerm;
                    aMat[pair.Value * k + pair.Key] -= symmetryTerm;
                }

                _cancelPairs = BuildPairs(p.bsSolveCancelPoses, activeIdx, _activeIndexMap);
                _aMat = aMat;

                _b = new float[k];
                _x = new float[k];
                _prevWeights = new float[k];
                _lower = new float[k];
                _upper = new float[k];
                for (int i = 0; i < k; i++) _lower[i] = 0f;
                _bvls = new BvlsWorkspace(k)
                {
                    UseNormalEquations = UseBvlsNormalEquations,
                    RelativeTolerance = BvlsRelativeTolerance
                };
            }

            _target = new float[_numPositions];
            _result = new float[NumPoses];
        }

        /// <summary>
        /// 在 json 参数之上叠加官方 stylization 覆盖（Audio2Face-3D-Samples/configs/*_diffusion_stylization_config.yaml）。
        /// 覆盖在「活跃列表确定前」执行，这样 active 变化会自然反映到后续的 D 矩阵 / AMat 构建。
        /// </summary>
        private void ApplyStylingOverrides(BlendshapeParams p)
        {
            if (_stylingKind == "skin")
            {
                ApplyActiveOverrides(p, SkinActiveOverrides);
                ApplyMultOverrides(p, SkinMultOverrides);
                ApplyOffsetOverrides(p, SkinOffsetOverrides);
            }
            else if (_stylingKind == "tongue")
            {
                ApplyActiveOverrides(p, TongueActiveOverrides);
                ApplyMultOverrides(p, TongueMultOverrides);
                ApplyOffsetOverrides(p, TongueOffsetOverrides);
            }
        }

        private void ApplyActiveOverrides(BlendshapeParams p, Dictionary<string, bool> ov)
        {
            if (ov == null || ov.Count == 0 || _activePoses == null) return;
            for (int i = 0; i < NumPoses && i < _activePoses.Length; i++)
            {
                if (ov.TryGetValue(PoseNames[i], out var active))
                {
                    int old = _activePoses[i];
                    _activePoses[i] = active ? 1 : 0;
                    if (old != _activePoses[i])
                        Debug.Log($"[BlendshapeSolver] styling override: {PoseNames[i]} active {old} -> {_activePoses[i]}");
                }
            }
        }

        private void ApplyMultOverrides(BlendshapeParams p, Dictionary<string, float> ov)
        {
            if (ov == null || ov.Count == 0 || _multipliers == null) return;
            for (int i = 0; i < NumPoses && i < _multipliers.Length; i++)
            {
                if (ov.TryGetValue(PoseNames[i], out var m))
                {
                    if (Mathf.Abs(_multipliers[i] - m) > 1e-6f)
                        Debug.Log($"[BlendshapeSolver] styling override: {PoseNames[i]} mult {_multipliers[i]} -> {m}");
                    _multipliers[i] = m;
                }
            }
        }

        private void ApplyOffsetOverrides(BlendshapeParams p, Dictionary<string, float> ov)
        {
            if (ov == null || ov.Count == 0 || _offsets == null) return;
            for (int i = 0; i < NumPoses && i < _offsets.Length; i++)
            {
                if (ov.TryGetValue(PoseNames[i], out var o))
                {
                    if (Mathf.Abs(_offsets[i] - o) > 1e-6f)
                        Debug.Log($"[BlendshapeSolver] styling override: {PoseNames[i]} offset {_offsets[i]} -> {o}");
                    _offsets[i] = o;
                }
            }
        }

        private static float BoundingBoxDiagonal(float[] neutral)
        {
            int numVertex = neutral.Length / 3;
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            for (int i = 0; i < numVertex; i++)
            {
                float x = neutral[3 * i + 0], y = neutral[3 * i + 1], z = neutral[3 * i + 2];
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (z < minZ) minZ = z;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                if (z > maxZ) maxZ = z;
            }
            float dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
            return Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        /// <summary>
        /// 把 cancelPoses / symmetryPoses（按全部 pose 索引）转换成活跃 pose 内的索引对。
        /// 同一个 pairId 恰好出现两次才成对。
        /// </summary>
        private static List<KeyValuePair<int, int>> BuildPairs(int[] pairIds, List<int> activeIdx, int[] activeIndexMap)
        {
            var result = new List<KeyValuePair<int, int>>();
            if (pairIds == null) return result;

            var buckets = new Dictionary<int, List<int>>();
            for (int i = 0; i < activeIdx.Count; i++)
            {
                int poseIdx = activeIdx[i];
                if (poseIdx >= pairIds.Length) continue;
                int id = pairIds[poseIdx];
                if (id < 0) continue;
                List<int> list;
                if (!buckets.TryGetValue(id, out list))
                {
                    list = new List<int>();
                    buckets[id] = list;
                }
                list.Add(i);
            }

            foreach (var kv in buckets)
            {
                if (kv.Value.Count == 2)
                    result.Add(new KeyValuePair<int, int>(kv.Value[0], kv.Value[1]));
            }
            return result;
        }

        /// <summary>
        /// 对一帧几何求解 blendshape 权重。
        /// </summary>
        /// <param name="geometry">该帧的几何数据起点（skin 段或 tongue 段）</param>
        /// <param name="offset">geometry 中的起始下标</param>
        /// <returns>长度 NumPoses 的权重数组（复用内部缓冲，调用方需自行拷贝）</returns>
        public float[] Solve(float[] geometry, int offset)
        {
            if (!IsReady) return null;

            // 1. target = masked(geometry)（delta 模式）或 masked(geometry) - masked(neutral)（绝对坐标模式）
            //    注意：_deltas 的每一行是按 frontalMask 挑出来的分量，
            //    target 必须走同一套掩码，否则 b = D^T·target 是错位投影，残差会接近 1。
            _profileOn = ++_profileFrames >= ProfilePeriod;
            // 每帧都测（Stopwatch 两次调用相对 ms 级解算可忽略），只在第 ProfilePeriod 帧汇总打印。
            // 坑：若只在那一帧测却又除以 ProfilePeriod，会打出小 300 倍的假数字。
            _sw.Restart();
            double sq = 0.0;
            for (int i = 0; i < _numPositions; i++)
            {
                int src = offset + (_maskPositions != null ? _maskPositions[i] : i);
                float v = geometry[src];
                if (!TargetIsDelta) v -= _neutral[i];
                if (_targetBias != null) v -= _targetBias[i];
                _target[i] = v;
                sq += (double)v * v;
            }
            _msTarget += _sw.Elapsed.TotalMilliseconds;
            LastTargetRms = _numPositions > 0 ? (float)Math.Sqrt(sq / _numPositions) : 0f;

            return SolveCore();
        }

        /// <summary>
        /// 输入<b>已经是掩码布局</b>（长度 SolvedPositionCount，第 i 个分量就是 _maskPositions[i]）
        /// 时走这条路径，跳过掩码取样。配合 <see cref="Audio2FaceSkinAnimator.AnimateMasked"/> 使用：
        /// animator 只重建 frontalMask 挑中的顶点、输出紧凑数组，solver 直接读，
        /// 省掉「全量重建 24002 个顶点 → 只挑 4262 个」的浪费。
        /// </summary>
        public float[] SolveMasked(float[] maskedGeometry, int offset)
        {
            if (!IsReady || maskedGeometry == null) return null;

            _profileOn = ++_profileFrames >= ProfilePeriod;
            _sw.Restart();

            bool absCoord = !TargetIsDelta;
            var neu = _neutral;
            var bias = _targetBias;
            double sq = 0.0;
            for (int i = 0; i < _numPositions; i++)
            {
                float v = maskedGeometry[offset + i];
                if (absCoord) v -= neu[i];
                if (bias != null) v -= bias[i];
                _target[i] = v;
                sq += (double)v * v;
            }
            _msTarget += _sw.Elapsed.TotalMilliseconds;
            LastTargetRms = _numPositions > 0 ? (float)Math.Sqrt(sq / _numPositions) : 0f;

            return SolveCore();
        }

        /// <summary>
        /// 用已经掩码好的 target 直接求解（长度必须 = SolvedPositionCount）。
        /// 仅供自检：把某个 delta pose 原样喂进去，应该解出该 pose ≈ 1、其余 ≈ 0、残差 ≈ 0。
        /// </summary>
        public float[] SolveMaskedTarget(float[] target)
        {
            if (!IsReady || target == null || target.Length < _numPositions) return null;

            double sq = 0.0;
            for (int i = 0; i < _numPositions; i++)
            {
                _target[i] = target[i];
                sq += (double)target[i] * target[i];
            }
            LastTargetRms = _numPositions > 0 ? (float)Math.Sqrt(sq / _numPositions) : 0f;

            return SolveCore();
        }

        /// <summary>该 pose 是否参与对称正则（有伙伴的 pose 自检时会天然残差偏高，不适合当样本）。</summary>
        public bool HasSymmetryPartner(int poseIndex)
        {
            return _symmetryPoses != null && poseIndex >= 0 && poseIndex < _symmetryPoses.Length &&
                   _symmetryPoses[poseIndex] >= 0;
        }

        /// <summary>取该 pose 的对称伙伴（同一 bsSolveSymmetryPoses id 的另一个活跃 pose）。</summary>
        public bool TryGetSymmetryPartner(int poseIndex, out int partner)
        {
            partner = -1;
            if (_symmetryPoses == null || poseIndex < 0 || poseIndex >= _symmetryPoses.Length) return false;
            int id = _symmetryPoses[poseIndex];
            if (id < 0) return false;
            for (int i = 0; i < _symmetryPoses.Length; i++)
            {
                if (i != poseIndex && _symmetryPoses[i] == id && IsPoseActive(i)) { partner = i; return true; }
            }
            return false;
        }

        /// <summary>bsSolveSymmetryPoses 的长度（0 = JSON 没解析出来，自检会选错 pose）。</summary>
        public int SymmetryPosesCount => _symmetryPoses != null ? _symmetryPoses.Length : 0;

        /// <summary>该 pose 是否在 bsSolveActivePoses 里被激活。</summary>
        public bool IsPoseActive(int poseIndex)
        {
            return _activePoses != null && poseIndex >= 0 && poseIndex < _activePoses.Length &&
                   _activePoses[poseIndex] != 0;
        }

        /// <summary>取某个 pose 的掩码后 delta（长度 SolvedPositionCount），自检用。</summary>
        public bool TryGetMaskedDelta(int poseIndex, float[] dst)
        {
            if (!IsReady || dst == null || dst.Length < _numPositions) return false;
            if (poseIndex < 0 || poseIndex >= NumPoses) return false;

            int ai = _activeIndexMap[poseIndex];
            if (ai < 0)
            {
                Array.Clear(dst, 0, _numPositions);
                return false;
            }
            Array.Copy(_deltas, ai * _numPositions, dst, 0, _numPositions);
            return true;
        }

        /// <summary>
        /// b = D^T·target 是否走 Vector&lt;float&gt; SIMD。设 false 会退回「4 路展开的 float 标量」，
        /// 用于老运行时（Vector 未被硬件加速时）或数值对拍。
        /// </summary>
        public static bool UseSimdDot = true;

        /// <summary>
        /// BVLS 内部解自由变量子问题时是否走「正规方程 + Cholesky」。
        /// 默认 true（比 Householder QR 快约 10 倍）。设 false 退回原来的 QR 路径做数值对拍；
        /// 建议在 Load 之前设置，之后改只影响新建的 solver。
        /// </summary>
        public static bool UseBvlsNormalEquations = true;

        /// <summary>
        /// BVLS 的<b>相对</b> KKT 容差：阈值 = max(solverTolerance, 本值 × 梯度量级)。
        /// 默认 1e-6 —— SDK 的绝对 1e-10 对 1e3~1e5 量级的梯度永远达不到，主循环只能靠
        /// 「没有变量想脱离边界」退出（实测 18.4 次迭代/帧）；改相对判据后降到个位数，
        /// 权重偏差 ~1e-5（见 [BVLS对拍] 第二行）。设 0 = 只认绝对容差（旧行为）。
        /// </summary>
        public static float BvlsRelativeTolerance = 1e-6f;

        // 解算耗时剖析：每 ProfilePeriod 帧打一条，把「点积」和「BVLS」分开，
        // 否则永远不知道 10ms/帧到底花在哪一段。
        private const int ProfilePeriod = 300;
        private readonly System.Diagnostics.Stopwatch _sw = new System.Diagnostics.Stopwatch();
        private int _profileFrames;
        private bool _profileOn;
        private double _msTarget;
        private double _msDot;
        private double _msBvls;
        private long _cmTotal;      // ConstrainedMin 次数（含 cancelPairs 重解）
        private long _iterTotal;    // BVLS 主循环迭代次数
        private int _cholFail;      // Cholesky 回退 QR 次数（>0 说明正规方程条件数撑不住）

        /// <summary>b = D^T·target → BVLS → cancelPairs → 映射回全部 pose。</summary>
        private float[] SolveCore()
        {
            int k = _b.Length;
            bool profile = _profileOn;

            // 2. b = D^T * target + temporalReg * scaleFactor * prevWeights
            //    这是每帧解算最热的一段：k(≈50) × _numPositions(≈6 万) ≈ 300 万次乘加。
            //    原来是「逐元素转 double 再累加」——一旦转 double，JIT 就只能标量化跑（double 只有 2-wide）。
            //    改成 System.Numerics.Vector<float> 显式 SIMD（SSE 4 路 / AVX 8 路），两条累加链拉开 ILP。
            //    精度：float 多路累加，误差约是单路的 1/2~1/2.8，比 double 大约 4 个数量级，
            //    但相对量级 ~1e-5，对 0~1 的权重和 BVLS 容差没有影响。怀疑数值问题时
            //    把 UseSimdDot 设 false 即退回原来的 double 标量路径（两种路径同时保留）。
            _sw.Restart();
            if (UseSimdDot && Vector.IsHardwareAccelerated)
            {
                int n = _numPositions;
                int vw = Vector<float>.Count;
                int limit = n - (n % vw);

                for (int j = 0; j < k; j++)
                {
                    int row = j * n;
                    var acc0 = Vector<float>.Zero;
                    var acc1 = Vector<float>.Zero;
                    int i = 0;
                    for (; i + vw * 2 <= limit; i += vw * 2)
                    {
                        acc0 += new Vector<float>(_deltas, row + i) * new Vector<float>(_target, i);
                        acc1 += new Vector<float>(_deltas, row + i + vw) * new Vector<float>(_target, i + vw);
                    }
                    for (; i < limit; i += vw)
                        acc0 += new Vector<float>(_deltas, row + i) * new Vector<float>(_target, i);

                    var acc = acc0 + acc1;
                    float sum = 0f;
                    for (int c = 0; c < vw; c++) sum += acc[c];   // 不用 Vector.Dot：部分 profile 没有它
                    for (; i < n; i++) sum += _deltas[row + i] * _target[i];

                    _b[j] = sum + _temporalReg * _scaleFactor * _prevWeights[j];
                }
            }
            else
            {
                // 没有硬件 SIMD 时的可移植加速：4 路展开 + 两条累加链（float，不转 double）。
                // 比原来的 double 标量版快一截（float 运算 + 指令级并行），且不依赖 Vector<T> 是否被加速。
                int n = _numPositions;
                int limit = n - (n % 4);
                for (int j = 0; j < k; j++)
                {
                    int row = j * n;
                    float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
                    int i = 0;
                    for (; i < limit; i += 4)
                    {
                        s0 += _deltas[row + i] * _target[i];
                        s1 += _deltas[row + i + 1] * _target[i + 1];
                        s2 += _deltas[row + i + 2] * _target[i + 2];
                        s3 += _deltas[row + i + 3] * _target[i + 3];
                    }
                    float sum = (s0 + s1) + (s2 + s3);
                    for (; i < n; i++) sum += _deltas[row + i] * _target[i];

                    _b[j] = sum + _temporalReg * _scaleFactor * _prevWeights[j];
                }
            }
            _msDot += _sw.Elapsed.TotalMilliseconds;

            // 3. BVLS
            var aMat = _disableReg ? _aMatNoReg : _aMat;
            for (int i = 0; i < k; i++) _upper[i] = 1f;
            Array.Copy(_prevWeights, _x, k);
            int cm0 = _bvls.CmCount, it0 = _bvls.IterCount, cf0 = _bvls.CholFail;
            _sw.Restart();
            _bvls.Solve(_x, aMat, k, _b, _lower, _upper, _tolerance);

            // 4. cancelPairs：把较小的那个上界压到 ~0 后重解
            if (_cancelPairs.Count > 0)
            {
                for (int i = 0; i < k; i++) _upper[i] = 1f;
                foreach (var pair in _cancelPairs)
                {
                    int loser = _x[pair.Key] >= _x[pair.Value] ? pair.Value : pair.Key;
                    _upper[loser] = 1e-10f;
                }
                _bvls.Solve(_x, aMat, k, _b, _lower, _upper, _tolerance);
            }
            _msBvls += _sw.Elapsed.TotalMilliseconds;
            _cmTotal += _bvls.CmCount - cm0;
            _iterTotal += _bvls.IterCount - it0;
            _cholFail += _bvls.CholFail - cf0;

            // 保存供下一帧的时间正则使用
            Array.Copy(_x, _prevWeights, k);

            // 5. 映射回全部 pose + multipliers/offsets + 静音基线
            //    基线只在输出层做，_prevWeights 保留原始解，时间正则才不会被标定打断。
            for (int i = 0; i < NumPoses; i++)
            {
                int ai = _activeIndexMap[i];
                float w = ai >= 0 ? _x[ai] : 0f;
                if (_multipliers != null && i < _multipliers.Length) w *= _multipliers[i];
                if (_offsets != null && i < _offsets.Length) w += _offsets[i];

                if (_weightBaseline != null)
                {
                    float b = _weightBaseline[i];
                    if (b < 0f) b = 0f; else if (b > 0.95f) b = 0.95f;
                    w -= b;
                    if (BaselineRescale) w /= (1f - b);
                }

                _result[i] = w < 0f ? 0f : (w > 1f ? 1f : w);
            }

            if (profile)
            {
                int f = _profileFrames;
                Debug.Log($"[解算耗时] n={_numPositions} k={k}: 掩码取样={_msTarget / f:F3} + D^T·target={_msDot / f:F3} " +
                          $"+ BVLS={_msBvls / f:F3} = {(_msTarget + _msDot + _msBvls) / f:F3}ms/帧（{f} 帧均值，" +
                          (UseSimdDot && Vector.IsHardwareAccelerated
                              ? $"SIMD Vector<float>×{Vector<float>.Count}"
                              : "4 路展开 float") +
                          $"；BVLS：{_cmTotal / (double)f:F1} 次线性解+{_iterTotal / (double)f:F1} 次迭代/帧" +
                          (_bvls != null ? $"，KKT阈值={_bvls.LastThreshold:E1}" : "") +
                          (_cholFail > 0 ? $"，Cholesky 回退 {_cholFail} 次" : "") +
                          $"，路径={(_bvls != null && _bvls.UseNormalEquations ? "Cholesky正规方程" : "Householder QR")}）");
                _profileFrames = 0;
                _profileOn = false;
                _msTarget = 0;
                _msDot = 0;
                _msBvls = 0;
                _cmTotal = 0;
                _iterTotal = 0;
                _cholFail = 0;
            }

            return _result;
        }

        public void Reset()
        {
            if (_prevWeights != null) Array.Clear(_prevWeights, 0, _prevWeights.Length);
        }

        public int SolvedPositionCount => _numPositions;

        /// <summary>
        /// frontalMask 展开成<b>分量级</b>的下标表（长度 SolvedPositionCount，值 = 3v+c）；
        /// 没开掩码时是 null。给 <see cref="Audio2FaceSkinAnimator.SetMask"/> 用，
        /// 让 animator 只重建 solver 真正会读的那部分顶点。
        /// </summary>
        public int[] MaskPositions => _maskPositions;
    }
}
