using System;
using System.IO;
using System.Threading;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// Audio2Face v3.0 完整流式口型管线。
    ///
    ///  麦克风/音频 → 环形缓冲 → 每 8000 样本触发一次扩散推理（后台线程）
    ///             → prediction[60, 88831] 取中心 30 帧
    ///             → BVLS 反解 skin(52) + tongue(16) blendshape 权重
    ///             → 帧队列 → SkinnedMeshRenderer
    /// </summary>
    public sealed class Audio2FaceDiffusionPipeline : IDisposable
    {
        private readonly Audio2FaceDiffusionConfig _config;
        private readonly Audio2FaceDiffusionModel _model;
        private readonly BlendshapeSolver _skinSolver;
        private readonly BlendshapeSolver _tongueSolver;
        private readonly Audio2FaceNetworkInfo _info;

        // SDK 后处理层：prediction(delta) → AnimatorSkin → 绝对顶点 → 求解器
        private readonly Audio2FaceModelData _modelData;
        private readonly Audio2FaceSkinAnimator _skinAnimator;
        private readonly Audio2FaceTongueAnimator _tongueAnimator;

        // SDK 完整管线还有两段不走 blendshape 的输出（sample-a2f-low-level-api-fullface/main.cpp）：
        //   jaw (15 = 5 个标记点位移) → AnimatorTeeth → 4x4 刚体变换 → 下颌骨（牙齿挂骨骼上，跟着动）
        //   eyes (4)                  → AnimatorEyes  → 左右眼欧拉角（度）+ saccade 微跳视
        // 少了这两段，嘴能「张开」但牙齿不动、眼睛死盯前方。
        private readonly Audio2FaceJawAnimator _jawAnimator;
        private readonly Audio2FaceEyesAnimator _eyesAnimator;
        private readonly float[] _jawTransform = new float[16];
        private readonly float[] _rightEyeRot = new float[3];
        private readonly float[] _leftEyeRot = new float[3];

        private readonly float[] _skinVertices;
        private readonly float[] _tongueVertices;
        // animator 的掩码视图缓冲：只装 frontalMask 挑出来的分量（= solver.SolvedPositionCount）。
        // 非空时 SolveSkinAt 走「只重建掩码顶点」的快路径。
        private float[] _skinVerticesMasked;
        private readonly float _dt;

        // 眨眼驱动：SDK 的 blinkOffset 要应用层每帧喂，SDK 不自带生成器
        private readonly A2FBlinkGenerator _blink;
        private readonly bool _blinkEnabled;

        // 音频环形缓冲
        private readonly float[] _ring;
        private int _ringHead;
        private long _ringTotal;
        private long _consumedUntil;

        // 交给后台线程的窗口
        private readonly float[] _window;
        private volatile bool _runRequested;
        private volatile bool _busy;

        private Thread _worker;
        private volatile bool _workerRunning;
        private volatile bool _workerPaused;

        // 帧队列（环形，避免 GC）
        private readonly float[] _frameStore;
        private readonly int _frameStride;
        private readonly int _frameCapacity;
        private int _frameHead;
        private int _frameCount;
        private readonly object _frameLock = new object();

        private volatile int _inferenceCount;
        private float _lastInferenceMs;
        // 单次耗时拆两段：ONNX 推理本身 vs 30 帧的解算/动画后处理。
        // 瓶颈常常不在 GPU 上而在后处理（实测 ONNX≈60ms、后处理≈300ms），不拆开永远看不出来。
        private float _lastRunMs;
        private float _lastPostMs;
        // 解算剖析：animator 顶点重建 vs blendshape 求解（skin / tongue 分开统计）
        private const int SolveProfilePeriod = 300;
        private readonly System.Diagnostics.Stopwatch _swSolve = new System.Diagnostics.Stopwatch();
        private int _solveFrames;
        private double _msAnim;
        private double _msSolve;
        private int _solveFramesT;
        private double _msAnimT;
        private double _msSolveT;

        // ---- 口型开合整形 + 嘴部通道诊断 ----
        // 背景：Mark 模板里几个「闭唇」pose 与 jawOpen 在顶点空间几乎反向共线
        //   jawOpen · mouthShrugLower = -0.921 / mouthPressLeft = -0.614 / mouthRollLower = -0.567
        // BVLS 拟合张嘴几何时会把它们一起解出来（几何残差很小），语义上却互相抵消，
        // 观感就是「下巴掉下去、嘴唇仍合着」。这里在权重层做定向整形。
        private int _iJawOpen = -1;
        private int[] _lipSealIdx = new int[0];
        private int[] _lipOpenIdx = new int[0];

        // Maya-ACE 官方场景 A2FAnimationPlayer 节点的 .bstm / .bsto —— 16 个 tongue pose 的
        // 乘法器与偏移。按 pose 名匹配，不依赖 npz 下标顺序。
        // 取值来自 Maya-ACE/sample_project/scenes/mark_v3_fullface.ma 第 320210~320211 行：
        //   setAttr -s 16 ".bstm[0:15]"  2 1 1 1 3 1 1 1 2 1 1 1 1 2 1 1;
        //   setAttr -s 16 ".bsto[9:15]"  0.2 0 0 0 0 0 0;
        private static readonly string[] TongueGainPoses =
        {
            "tongueTipUp", "tongueTipDown", "tongueTipLeft", "tongueTipRight",
            "tongueRollUp", "tongueRollDown", "tongueRollLeft", "tongueRollRight",
            "tongueUp", "tongueDown", "tongueLeft", "tongueRight",
            "tongueIn", "tongueStretch", "tongueWide", "tongueNarrow"
        };
        private static readonly float[] TongueGainMul =
        {
            2f, 1f, 1f, 1f, 3f, 1f, 1f, 1f, 2f, 1f, 1f, 1f, 1f, 2f, 1f, 1f
        };
        private static readonly float[] TongueGainOff =
        {
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.2f, 0f, 0f, 0f, 0f, 0f, 0f
        };
        private int[] _tongueGainIdx = new int[0];
        private int[] _mouthDbgIdx = new int[0];
        private readonly string[] _mouthDbgNames =
        {
            "jawOpen", "mouthClose", "mouthFunnel", "mouthLowerDownLeft", "mouthUpperUpLeft",
            "mouthPressLeft", "mouthShrugLower", "mouthRollLower", "mouthPucker"
        };
        private float[] _mouthDbgMax = new float[0];
        private int _mouthDbgFrames;
        private float _tongueDbgMax;
        private const int MouthDbgPeriod = 60;
        /// <summary>当前这次推理的窗口右端（样本号），SolveCenterFrames 用它判断 warmup。</summary>
        private long _windowEnd;

        public bool IsRunning => _workerRunning;
        public bool IsBusy => _busy;
        public float LastInferenceMs => _lastInferenceMs;
        /// <summary>最近一次推理里 ONNX Session.Run 本身的耗时（不含解算）。</summary>
        public float LastRunMs => _lastRunMs;
        /// <summary>最近一次推理里 30 帧 skin/tongue/jaw 解算 + 动画的耗时。</summary>
        public float LastPostProcessMs => _lastPostMs;
        public int PendingFrames { get { lock (_frameLock) return _frameCount; } }
        /// <summary>环形缓冲里尚未被推理消费的音频样本数（背压用）。</summary>
        public long UnconsumedAudioSamples => _ringTotal - _consumedUntil;
        /// <summary>背压上限 = 窗口大小。未消费样本数达到一个窗口时暂停推流，
        /// 保证游标式窗口的左端始终还在 ring 里（ring 物理容量是它的 2 倍）。</summary>
        public int RingCapacity => _info != null ? _info.BufferLength : (_ring != null ? _ring.Length / 2 : 0);
        /// <summary>消费游标（= 已推进到的窗口右端样本号）。</summary>
        public long ConsumedSamples => _consumedUntil;
        public int SkinPoseCount { get; private set; }
        public int TonguePoseCount { get; private set; }
        public int FrameStride => _frameStride;

        /// <summary>
        /// jaw 通道数：旋转角(度) + 旋转轴 xyz + 下颌质心真实位移 xyz。
        ///
        /// 为什么不存 4x4 的平移：Kabsch 解出的 t = aMean − R·bMean 是「绕 A2F 模型原点旋转」
        /// 的结果，而模型原点离下颌 150 个单位 —— 4° 的旋转就能产生 10 个单位的 t（实测
        /// t.z 最大 10.7，而真实质心位移只有 −0.7）。再叠加一次绕骨骼轴的旋转就是重复计算。
        /// </summary>
        public const int JawChannelCount = 7;
        /// <summary>eyes 通道数：rightX、rightY、leftX、leftY（度）。</summary>
        public const int EyesChannelCount = 4;
        /// <summary>帧里 jaw 通道的起始下标。</summary>
        public int JawChannelOffset => SkinPoseCount + TonguePoseCount;
        /// <summary>帧里 eyes 通道的起始下标。</summary>
        public int EyesChannelOffset => JawChannelOffset + JawChannelCount;
        public Audio2FaceNetworkInfo Info => _info;
        public BlendshapeSolver SkinSolver => _skinSolver;
        public BlendshapeSolver TongueSolver => _tongueSolver;
        public Audio2FaceSkinAnimator SkinAnimator => _skinAnimator;
        public bool UsingSkinAnimator => _skinAnimator != null && _skinAnimator.IsReady;
        public float[] SkinVertices => _skinVertices;
        public float[] Prediction => _model.Prediction;
        public int InferenceCount => _inferenceCount;

        public Audio2FaceDiffusionPipeline(Audio2FaceDiffusionConfig config)
        {
            _config = config;
            string folder = config.ModelFolderFullPath;

            _model = new Audio2FaceDiffusionModel();
            _model.Initialize(config);
            _info = _model.Info;

            // 眨眼调度器（blinkOffset 由应用层每帧喂给 animator，SDK 不自带）
            _blinkEnabled = config.enableBlink;
            _blink = new A2FBlinkGenerator(config.blinkSeed);
            _blink.enabled = _blinkEnabled;
            _blink.minInterval = config.blinkIntervalMin;
            _blink.maxInterval = config.blinkIntervalMax;
            _blink.duration = config.blinkDuration;
            _blink.doubleBlinkChance = config.blinkDoubleChance;
            _blink.Reset(config.blinkSeed);

            // 输入音频增益（官方 stylization 的 input_strength，Mark=1.3），独立于 animator 是否开启。
            // 读不到配置就保持默认 1.0（与 SDK 库的 model_config 基线一致）。
            var animCfg = Audio2FaceAnimatorConfig.Load(folder, config.identity);
            if (animCfg != null) _model.InputStrength = animCfg.InputStrength;

            // 目标网格不是 MetaHuman 时官方增益会过头，允许在 Inspector 手动压低
            if (config.overrideInputStrength)
            {
                _model.InputStrength = config.inputStrength;
                if (config.debugMode)
                    Debug.Log($"[Audio2Face] 音频增益已被 overrideInputStrength 改为 {config.inputStrength:F2}" +
                              $"（官方值={animCfg?.InputStrength ?? 1f:F2}）");
            }

            // SDK 的 AnimatorSkin 把 prediction 变成绝对顶点，开了它求解器就不能再按 delta 处理。
            // 必须先决定这个再建求解器——TargetIsDelta 是构造参数，事后改容易漏。
            bool delta = config.predictionIsDelta;
            if (config.useSkinAnimator || config.useTongueAnimator ||
                config.useJawAnimator || config.useEyesAnimator)
            {
                _modelData = Audio2FaceModelData.Load(folder, config.identity);
                if (_modelData != null)
                {
                    // animator 参数已在上面统一加载（animCfg，含官方 stylization face_params 叠加）。

                    if (config.useSkinAnimator)
                    {
                        if (_modelData.IsReady && _modelData.NeutralSkin.Length == _info.SkinSize)
                        {
                            bool userOverride = config.overrideSkinAnimator && config.skinAnimator != null;
                            var skinParams = userOverride
                                ? config.skinAnimator
                                : (animCfg != null ? animCfg.ToSkinParams() : Audio2FaceSkinAnimator.Params.SdkDefault);

                            // 强度覆盖：只有上下半脸 / skin 三个增益，其余（lipOpenOffset、eyelidOffset、
                            // 平滑系数、mask 参数）继续用官方值。这样不用在 Inspector 里手填整个 Params。
                            if (config.overrideFaceStrength)
                            {
                                skinParams.lowerFaceStrength = config.lowerFaceStrength;
                                skinParams.upperFaceStrength = config.upperFaceStrength;
                                skinParams.skinStrength = config.skinStrength;
                            }

                            _skinAnimator = new Audio2FaceSkinAnimator(_modelData, skinParams);
                            _skinVertices = new float[_info.SkinSize];
                            delta = false;              // animator 已经把 neutral 加回去了
                            if (config.debugMode)
                            {
                                string src = userOverride ? "用户覆盖(overrideSkinAnimator)"
                                           : (animCfg != null ? $"model_config+官方stylization({config.identity})" : "SDK代码默认");
                                if (config.overrideFaceStrength)
                                    src += $"+overrideFaceStrength(lower={config.lowerFaceStrength:F2},skin={config.skinStrength:F2})";
                                Debug.Log($"[Audio2Face] skin animator 已启用: 顶点={_modelData.SkinVertexCount} " +
                                          $"下半脸×{_skinAnimator.CurrentParams.lowerFaceStrength} " +
                                          $"上半脸×{_skinAnimator.CurrentParams.upperFaceStrength} " +
                                          $"lipOpenOffset={_skinAnimator.CurrentParams.lipOpenOffset} (参数源={src})");
                            }
                        }
                        else
                        {
                            Debug.LogWarning($"[Audio2Face] model_data 不可用或与 skin_size 不匹配 " +
                                             $"({_modelData.SkinVertexCount * 3} vs {_info.SkinSize})，skin animator 关闭");
                        }
                    }

                    if (config.useTongueAnimator && _modelData.NeutralTongue != null &&
                        _modelData.NeutralTongue.Length == _info.TongueSize)
                    {
                        _tongueAnimator = new Audio2FaceTongueAnimator(_modelData);
                        if (animCfg != null)
                            _tongueAnimator.SetParams(animCfg.config.tongue_strength,
                                                      animCfg.config.tongue_height_offset,
                                                      animCfg.config.tongue_depth_offset);
                        _tongueVertices = new float[_info.TongueSize];
                    }

                    // 下颌/牙齿：jaw 段 → 刚体变换。牙齿挂在骨骼上，只有这一步能让它动。
                    if (config.useJawAnimator && _modelData.NeutralJaw != null &&
                        _modelData.NeutralJaw.Length == _info.JawSize)
                    {
                        _jawAnimator = new Audio2FaceJawAnimator(
                            _modelData.NeutralJaw,
                            animCfg != null ? animCfg.ToTeethParams() : Audio2FaceJawAnimator.Params.Default);
                        if (config.debugMode)
                            Debug.Log($"[Audio2Face] jaw animator 已启用: 标记点={_modelData.NeutralJaw.Length / 3} " +
                                      $"(jaw_size={_info.JawSize})");
                    }
                    else if (config.useJawAnimator && config.debugMode)
                    {
                        Debug.LogWarning($"[Audio2Face] neutral_jaw 缺失或与 jaw_size 不匹配 " +
                                         $"({_modelData.NeutralJaw?.Length ?? 0} vs {_info.JawSize})，下颌不会动");
                    }

                    // 眼球：eyes 段 + saccade_rot_matrix → 左右眼欧拉角
                    if (config.useEyesAnimator && _modelData.SaccadeRotMatrix != null &&
                        _modelData.SaccadeRotMatrix.Length >= 2)
                    {
                        _eyesAnimator = new Audio2FaceEyesAnimator(
                            _modelData.SaccadeRotMatrix,
                            animCfg != null ? animCfg.ToEyesParams() : Audio2FaceEyesAnimator.Params.Default);
                        if (config.debugMode)
                            Debug.Log($"[Audio2Face] eyes animator 已启用: saccade 帧数={_modelData.SaccadeRotMatrix.Length / 2} " +
                                      $"(eyes_size={_info.EyesSize})");
                    }
                    else if (config.useEyesAnimator && config.debugMode)
                    {
                        Debug.LogWarning("[Audio2Face] saccade_rot_matrix 缺失，眼球不会动");
                    }
                }
            }

            _skinSolver = new BlendshapeSolver(
                Path.Combine(folder, $"bs_skin_{config.identity}.npz"),
                Path.Combine(folder, $"bs_skin_config_{config.identity}.json"),
                config.useFrontalMask, config.solverTolerance, delta, "skin");

            if (_tongueAnimator != null && _tongueAnimator.IsReady)
            {
                // tongue animator 输出的是绝对顶点，所以这一段不走 delta 路径
                _tongueSolver = new BlendshapeSolver(
                    Path.Combine(folder, $"bs_tongue_{config.identity}.npz"),
                    Path.Combine(folder, $"bs_tongue_config_{config.identity}.json"),
                    false, config.solverTolerance, false, "tongue");
            }
            else
            {
                _tongueSolver = new BlendshapeSolver(
                    Path.Combine(folder, $"bs_tongue_{config.identity}.npz"),
                    Path.Combine(folder, $"bs_tongue_config_{config.identity}.json"),
                    false, config.solverTolerance, delta, "tongue");
            }

            _dt = _info.FrameRate > 0f ? 1f / _info.FrameRate : 1f / 60f;

            if (!_skinSolver.IsReady)
                throw new Exception("skin blendshape 求解器初始化失败");

            // ── animator 掩码视图 ──────────────────────────────────────────────
            // solver 只按 frontalMask 读 ~18% 的顶点，而插值是逐分量独立的，
            // 所以让 animator 只重建这些顶点即可，输出紧凑布局直接喂 SolveMasked。
            // 数值上与「重建全部顶点再挑」逐位一致（faceMask 沿用全量 neutral 的 Y 归一化基准）。
            if (_skinAnimator != null && _skinAnimator.IsReady)
            {
                _skinAnimator.SetMask(_skinSolver.MaskPositions);
                if (_skinAnimator.HasMaskView)
                {
                    _skinVerticesMasked = new float[_skinSolver.SolvedPositionCount];
                    if (config.debugMode)
                        Debug.Log($"[Audio2Face] skin animator 掩码视图已启用: 顶点 {_info.SkinSize / 3} → " +
                                  $"{_skinSolver.SolvedPositionCount / 3}（{_skinSolver.SolvedPositionCount * 100f / _info.SkinSize:F0}%），" +
                                  $"分量 {_info.SkinSize} → {_skinSolver.SolvedPositionCount}");
                }
            }

            SkinPoseCount = _skinSolver.NumPoses;
            TonguePoseCount = _tongueSolver.IsReady ? _tongueSolver.NumPoses : 0;
            // 尾部再挂 8 个「非 blendshape」通道：jaw 4 + eyes 4
            _frameStride = SkinPoseCount + TonguePoseCount + JawChannelCount + EyesChannelCount;

            // 环形缓冲容量 = 窗口大小的 2 倍。窗口 = [consumedUntil - BufferLength, consumedUntil)
            // 游标式窗口会「落后」于最新推送（推流快于推理时），窗口左端需要的历史样本
            // 不能被新数据覆盖，所以 ring 必须能同时容纳一个窗口 + 最多一个窗口的未消费余量。
            _ring = new float[_info.BufferLength * 2];
            _window = new float[_info.BufferLength];
            _frameCapacity = 512;
            _frameStore = new float[_frameCapacity * _frameStride];

            BuildMouthTables();

            StartWorker();
        }

        /// <summary>
        /// 按 pose 名建口型整形 / 诊断用的索引表。名字取自 npz 的 poseNames，
        /// 所以不依赖任何硬编码下标（Mark/Claire/James 三套 npz 顺序一致）。
        /// </summary>
        private void BuildMouthTables()
        {
            var n = _skinSolver != null ? _skinSolver.PoseNames : null;
            if (n == null) return;

            _iJawOpen = PoseIndex(n, "jawOpen");

            _lipSealIdx = Collect(n, new[]
            {
                "mouthClose", "mouthPressLeft", "mouthPressRight",
                "mouthShrugLower", "mouthRollLower",
            });

            _lipOpenIdx = Collect(n, new[]
            {
                "mouthLowerDownLeft", "mouthLowerDownRight",
                "mouthUpperUpLeft", "mouthUpperUpRight",
            });

            _mouthDbgIdx = new int[_mouthDbgNames.Length];
            for (int i = 0; i < _mouthDbgNames.Length; i++)
                _mouthDbgIdx[i] = PoseIndex(n, _mouthDbgNames[i]);
            _mouthDbgMax = new float[_mouthDbgNames.Length];

            if (_config.applyOfficialTongueGains && _tongueSolver != null && _tongueSolver.IsReady)
            {
                var tn = _tongueSolver.PoseNames;
                _tongueGainIdx = new int[TongueGainPoses.Length];
                for (int i = 0; i < TongueGainPoses.Length; i++)
                    _tongueGainIdx[i] = PoseIndex(tn, TongueGainPoses[i]);
            }
        }

        private static int PoseIndex(string[] names, string pose)
        {
            if (names == null) return -1;
            for (int i = 0; i < names.Length; i++)
                if (names[i] == pose) return i;
            return -1;
        }

        private static int[] Collect(string[] names, string[] poses)
        {
            var tmp = new int[poses.Length];
            int c = 0;
            foreach (var p in poses)
            {
                int i = PoseIndex(names, p);
                if (i >= 0) tmp[c++] = i;
            }
            var res = new int[c];
            for (int i = 0; i < c; i++) res[i] = tmp[i];
            return res;
        }

        /// <summary>
        /// 口型开合整形（就地修改 skin 权重，必须在基线减法之后、入队之前调用）。
        ///
        /// 两步：
        ///   1. 闭唇组 × lipSealScale × (1 - k × jawOpen)：张得越开，闭唇通道压得越狠。
        ///      按 jawOpen 动态压制而不是一刀切，/m/ /b/ 这类闭口音（jawOpen≈0）不受影响。
        ///   2. 唇分开组 += jawOpenToLipOpen × jawOpen：jawOpen 只带下颌，
        ///      补一点 mouthLowerDown/mouthUpperUp 才能真正把唇缝拉开、露出牙齿。
        /// </summary>
        private void ShapeMouthOpen(float[] w)
        {
            if (w == null || _iJawOpen < 0 || _iJawOpen >= w.Length) return;
            if (!_config.enableMouthOpenAssist) return;

            float open = w[_iJawOpen];
            if (open < 0f) open = 0f; else if (open > 1f) open = 1f;

            // jawOpen 单独放大（它与其它 pose 几乎正交，放大它不会把闭唇通道一起放大）。
            // 必须在压制闭唇之前做：suppress 用的是放大后的 jawOpen。
            float gain = _config.jawOpenGain;
            if (!Mathf.Approximately(gain, 1f))
            {
                float g = open * gain;
                w[_iJawOpen] = g > 1f ? 1f : g;
                open = w[_iJawOpen];
            }

            float seal = _config.lipSealScale;
            float suppress = 1f - _config.lipSealSuppressByJawOpen * open;
            if (suppress < 0f) suppress = 0f;
            float sealGain = seal * suppress;

            for (int i = 0; i < _lipSealIdx.Length; i++)
            {
                int k = _lipSealIdx[i];
                if (k < 0 || k >= w.Length) continue;
                float v = w[k] * sealGain;
                w[k] = v < 0f ? 0f : (v > 1f ? 1f : v);
            }

            float add = _config.jawOpenToLipOpen * open;
            if (add > 0f)
            {
                for (int i = 0; i < _lipOpenIdx.Length; i++)
                {
                    int k = _lipOpenIdx[i];
                    if (k < 0 || k >= w.Length) continue;
                    float v = w[k] + add;
                    w[k] = v > 1f ? 1f : v;
                }
            }
        }

        /// <summary>
        /// 舌头权重整形：套 Maya 官方场景的 .bstm / .bsto（16 路乘法器 + 偏移）。
        /// w = w × mul + off，clamp 到 [0,1]。
        /// </summary>
        private void ShapeTongue(float[] w)
        {
            if (w == null || _tongueGainIdx == null || _tongueGainIdx.Length == 0) return;

            for (int i = 0; i < _tongueGainIdx.Length; i++)
            {
                int k = _tongueGainIdx[i];
                if (k < 0 || k >= w.Length) continue;
                float v = w[k] * TongueGainMul[i] + TongueGainOff[i];
                w[k] = v < 0f ? 0f : (v > 1f ? 1f : v);
            }
        }

        /// <summary>
        /// 嘴部通道峰值诊断。逐帧记最大值，攒够 60 帧（1 秒）打一次。
        /// 看这条就能判断「嘴没张开」到底是 jawOpen 不够、还是被闭唇通道抵消。
        /// </summary>
        private void TraceMouthChannels(float[] skin, float[] tongue)
        {
            if (!_config.debugMode || _mouthDbgIdx == null || _mouthDbgIdx.Length == 0) return;

            for (int k = 0; k < _mouthDbgIdx.Length; k++)
            {
                int i = _mouthDbgIdx[k];
                float v = (i >= 0 && skin != null && i < skin.Length) ? skin[i] : 0f;
                if (v > _mouthDbgMax[k]) _mouthDbgMax[k] = v;
            }
            if (tongue != null)
            {
                for (int i = 0; i < tongue.Length; i++)
                    if (tongue[i] > _tongueDbgMax) _tongueDbgMax = tongue[i];
            }

            if (++_mouthDbgFrames < MouthDbgPeriod) return;

            var sb = new System.Text.StringBuilder();
            sb.Append("[A2F嘴部] 近 60 帧峰值: ");
            for (int k = 0; k < _mouthDbgNames.Length; k++)
                sb.Append($"{_mouthDbgNames[k]}={_mouthDbgMax[k]:F2} ");
            sb.Append($"| tongueMax={_tongueDbgMax:F2}");
            if (_iJawOpen >= 0)
            {
                string mode = _config.enableMouthOpenAssist ? "开" : "关";
                sb.Append($" | 整形={mode}");
                if (_config.enableMouthOpenAssist)
                    sb.Append($"(seal×{_config.lipSealScale:F2}, k={_config.lipSealSuppressByJawOpen:F2}, +lip={_config.jawOpenToLipOpen:F2})");
            }
            if (_jawAnimator != null && _jawAnimator.IsReady)
                sb.Append($" | jaw角={_jawAnimator.LastAngleDeg:F2}° 下颌下降={-_jawAnimator.LastCentroidDelta.y:F3}" +
                          $"(杠杆t.z={_jawAnimator.LastTranslation.z:F2}，比它大一个量级是正常的)");
            if (_eyesAnimator != null && _eyesAnimator.IsReady)
                sb.Append($" | 眼 R=({_rightEyeRot[0]:F1},{_rightEyeRot[1]:F1}) L=({_leftEyeRot[0]:F1},{_leftEyeRot[1]:F1})");

            Debug.Log(sb.ToString());

            for (int k = 0; k < _mouthDbgMax.Length; k++) _mouthDbgMax[k] = 0f;
            _tongueDbgMax = 0f;
            _mouthDbgFrames = 0;
        }

        private void StartWorker()
        {
            _workerRunning = true;
            _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "A2F-Diffusion" };
            _worker.Start();
        }

        /// <summary>主线程调用：把 16kHz 单声道 PCM 推进缓冲。</summary>
        public void PushAudio(float[] data, int offset, int count)
        {
            int len = _ring.Length;
            for (int i = 0; i < count; i++)
            {
                _ring[_ringHead] = data[offset + i];
                _ringHead = (_ringHead + 1) % len;
                _ringTotal++;
            }
        }

        /// <summary>主线程调用：判断是否需要触发一次推理。</summary>
        public void Tick()
        {
            if (_runRequested || _busy) return;

            // 对齐 SDK WindowProgress（executor_diffusion_core.cpp GetProgressParameters）：
            //   窗口右端 = 消费游标 consumedUntil（每次推理后 +stride=8000），
            //   窗口 = [consumedUntil - BufferLength, consumedUntil)，负的部分（音频未到）补 0。
            // 触发条件：ringTotal >= consumedUntil（音频已覆盖到窗口右端）。
            //
            // 之前用「最近 ringTotal 个样本右对齐」，在推流快于推理时窗口会超前（右端跑到
            // 最新推送位置），口型比听到的声音提前约 1 秒，是「张嘴/眨眼不自然」的根因。
            if (_ringTotal < _consumedUntil) return;

            int ringLen = _ring.Length;                 // 物理容量（2 × 窗口）
            int windowLen = _info.BufferLength;         // 窗口大小
            int available = (int)Math.Min(_consumedUntil, windowLen);
            int zeroCount = windowLen - available;

            // 窗口起点样本号 = consumedUntil - available，其在环形缓冲里的位置
            int start = (int)((_consumedUntil - available) % ringLen);

            for (int i = 0; i < zeroCount; i++) _window[i] = 0f;
            for (int i = 0; i < available; i++) _window[zeroCount + i] = _ring[(start + i) % ringLen];

            // 记录本次推理的窗口右端（递增前），供 warmup 判断
            _windowEnd = _consumedUntil;
            _consumedUntil += _info.StrideSamples;
            _runRequested = true;

            // 诊断：打印窗口滑动证据（前 3 样本 + 头部位置 + 累计计数），判断窗口是否冻结
            if (_inferenceCount <= 12)
            {
                Debug.Log($"[A2F窗口] Tick 触发: ringTotal={_ringTotal} consumed={_consumedUntil} ringHead={_ringHead} " +
                          $"start={start} win[0..2]={_window[0]:F4},{_window[1]:F4},{_window[2]:F4} win[8000]={_window[8000]:F4}");
            }
        }

        private void WorkerLoop()
        {
            while (_workerRunning)
            {
                if (!_runRequested || _workerPaused)
                {
                    Thread.Sleep(1);
                    continue;
                }

                _busy = true;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                long msAfterRun = 0;
                try
                {
                    _model.Run(_window);
                    msAfterRun = sw.ElapsedMilliseconds;   // ← ONNX 推理到此为止
                    SolveCenterFrames();                   // ← 剩下全是 CPU 后处理（30 帧解算）
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[Audio2Face] 推理失败: {ex}");
                }
                sw.Stop();
                _lastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;
                _lastRunMs = (float)msAfterRun;
                _lastPostMs = Mathf.Max(0f, _lastInferenceMs - _lastRunMs);
                // 拆成两段是为了定位瓶颈：ONNX 段是 GPU/推理后端的事，
                // 后处理段是 30 帧×（skin animator + blendshape 解算）的纯 CPU 活。
                // 前 3 次每次都打（第 1 次含 CUDA kernel autotune，偏慢是正常的），之后每 20 次一条防刷屏。
                if (_inferenceCount <= 3 || _inferenceCount % 20 == 0)
                    Debug.Log($"[A2F推理] #{_inferenceCount} ONNX={_lastRunMs:F0}ms 后处理={_lastPostMs:F0}ms " +
                              $"合计={_lastInferenceMs:F0}ms 后端={_model.ActiveProvider}");
                _runRequested = false;
                _busy = false;
            }
        }

        private void SolveCenterFrames()
        {
            var pred = _model.Prediction;
            int totalDim = _model.TotalDim;
            int left = _info.FramesLeftTruncate;
            int center = _info.FramesCenter;

            // 后处理（解算）单独计时：_lastInferenceMs 要到 WorkerLoop 结束才更新，
            // 在这里读它是上一次的值（以前这条日志一直打的是上一次的耗时）。
            var postSw = _config != null && _config.debugMode ? System.Diagnostics.Stopwatch.StartNew() : null;

            _inferenceCount++;
            // warmup = 窗口右端还没覆盖满一个窗口（窗口仍含 padding_left）。
            // 之前用 _inferenceCount <= WarmupInferences 判断，但 _inferenceCount 会被
            // CalibrateRestPose 的静音推理推进（日志里标定后已经到 4），导致正式推理阶段
            // 前几个含 padding 的窗口被当成正式帧输出。改用窗口右端游标判断更可靠。
            bool warmup = _windowEnd < _info.BufferLength;

            if (!warmup)
            {
                // 逐帧诊断：只看每次推理的第一帧（f==left）会漏掉 29/30 的数据，
                // 「一抽一抽」的单帧尖刺必须看连续序列才能发现。
                bool trace = _config.debugMode && _windowEnd < _info.BufferLength * 4;
                var sb = trace ? new System.Text.StringBuilder() : null;
                int iJaw = -1, iClose = -1, iBlinkL = -1, iBlinkR = -1;
                if (_skinSolver != null)
                {
                    var names = _skinSolver.PoseNames;
                    if (names != null)
                    {
                        iJaw = System.Array.IndexOf(names, "jawOpen");
                        iClose = System.Array.IndexOf(names, "mouthClose");
                        iBlinkL = System.Array.IndexOf(names, "eyeBlinkLeft");
                        iBlinkR = System.Array.IndexOf(names, "eyeBlinkRight");
                    }
                    if (sb != null)
                        sb.AppendLine($"[A2F逐帧] 窗口右端={_windowEnd} → 30 帧 jawOpen / mouthClose / eyeBlinkL / eyeBlinkR");
                }

                // 眨眼驱动：算出帧 f 对应的音频时刻。
                // 本次窗口覆盖音频 [windowEnd - BufferLength, windowEnd)，模型输出的 60 帧在这段音频上
                // 均匀展开，所以帧 f ↔ 样本 winStart + f * (BufferLength / FramesPerInference)。
                // 用帧时间戳（而不是墙钟）驱动，眨眼才会固定落在动画时间轴上、并且和音频对齐。
                long winStart = _windowEnd - _info.BufferLength;
                float samplesPerFrame = _info.FramesPerInference > 0
                    ? (float)_info.BufferLength / _info.FramesPerInference : 1f;
                float invRate = _info.SampleRate > 0 ? 1f / _info.SampleRate : 1f / 16000f;
                bool driveBlink = _blinkEnabled && _skinAnimator != null && _skinAnimator.IsReady;

                for (int f = left; f < left + center; f++)
                {
                    int row = f * totalDim;

                    float blinkOff = 0f;
                    if (driveBlink)
                    {
                        blinkOff = _blink.Evaluate((winStart + f * samplesPerFrame) * invRate);
                        _skinAnimator.SetBlinkOffset(blinkOff);
                    }

                    var skin = SolveSkinAt(row);
                    int tongueCount = 0;
                    float[] tongue = null;
                    if (_tongueSolver.IsReady)
                    {
                        tongue = SolveTongueAt(row);
                        tongueCount = _tongueSolver.NumPoses;
                    }

                    // jaw / eyes：SDK 完整管线里这两段不走 blendshape
                    // （jaw → 4x4 刚体变换驱动下颌骨；eyes → 左右眼欧拉角 + saccade）
                    if (_jawAnimator != null && _jawAnimator.IsReady)
                        _jawAnimator.ComputeJawTransform(_jawTransform, pred, row + _info.JawOffset);
                    if (_eyesAnimator != null && _eyesAnimator.IsReady)
                    {
                        _eyesAnimator.IncrementLiveTime(_dt);
                        _eyesAnimator.ComputeEyesRotation(_rightEyeRot, _leftEyeRot, pred, row + _info.EyesOffset);
                    }

                    // 开合整形 + 嘴部诊断（必须放在基线减法之后、入队之前）
                    ShapeMouthOpen(skin);
                    ShapeTongue(tongue);
                    TraceMouthChannels(skin, tongue);

                    // 诊断：打印前 20 次推理的基线减法后权重
                    if (_inferenceCount <= 30 && f == left)
                    {
                        float sum = 0f;
                        if (skin != null) { for (int i = 0; i < skin.Length; i++) sum += Mathf.Abs(skin[i]); }
                        int jaw = skin != null ? System.Array.IndexOf(_skinSolver.PoseNames, "jawOpen") : -1;
                        int mouth = skin != null ? System.Array.IndexOf(_skinSolver.PoseNames, "mouthClose") : -1;
                        int smileL = skin != null ? System.Array.IndexOf(_skinSolver.PoseNames, "mouthSmileLeft") : -1;
                        Debug.Log($"[A2F权重] #{_inferenceCount} audio={_model.AudioRMS:F4} sum={sum:F4} jawOpen={(jaw >= 0 && skin != null ? skin[jaw].ToString("F4") : "-")} mouthClose={(mouth >= 0 && skin != null ? skin[mouth].ToString("F4") : "-")} smileL={(smileL >= 0 && skin != null ? skin[smileL].ToString("F4") : "-")}");
                    }

                    if (sb != null && skin != null)
                    {
                        float gj = (iJaw >= 0 && iJaw < skin.Length) ? skin[iJaw] : 0f;
                        float gc = (iClose >= 0 && iClose < skin.Length) ? skin[iClose] : 0f;
                        float gl = (iBlinkL >= 0 && iBlinkL < skin.Length) ? skin[iBlinkL] : 0f;
                        float gr = (iBlinkR >= 0 && iBlinkR < skin.Length) ? skin[iBlinkR] : 0f;
                        sb.AppendLine($"   f={f:D2} t={(winStart + f * samplesPerFrame) * invRate:F2}s  jaw={gj:F3}  close={gc:F3}  blinkL={gl:F3}  blinkR={gr:F3}  blinkOff={blinkOff:F2}");

                    }

                    // 眨眼只在少数帧上非零，单独打一条。不能放进上面的 sb 块 ——
                    // trace 只覆盖前 2 秒音频，而眨眼最早也要 2.5s 后才出现，放进去永远看不到。
                    if (_config.debugMode && blinkOff > 0.01f && skin != null)
                        Debug.Log($"[A2F眨眼] t={(winStart + f * samplesPerFrame) * invRate:F2}s " +
                                  $"blinkOffset={blinkOff:F2} → eyeBlink L=" +
                                  (iBlinkL >= 0 && iBlinkL < skin.Length ? skin[iBlinkL].ToString("F3") : "-") +
                                  " R=" + (iBlinkR >= 0 && iBlinkR < skin.Length ? skin[iBlinkR].ToString("F3") : "-"));

                    lock (_frameLock)
                    {
                        int slot = (_frameHead + _frameCount) % _frameCapacity;
                        int dst = slot * _frameStride;

                        for (int i = 0; i < SkinPoseCount; i++) _frameStore[dst + i] = skin != null ? skin[i] : 0f;
                        for (int i = 0; i < tongueCount; i++) _frameStore[dst + SkinPoseCount + i] = tongue[i];

                        // jaw 7 + eyes 4
                        int jb = SkinPoseCount + TonguePoseCount;
                        bool jawOk = _jawAnimator != null && _jawAnimator.IsReady;
                        var jd = jawOk ? _jawAnimator.LastCentroidDelta : Vector3.zero;
                        var ja = jawOk ? _jawAnimator.LastAxis : new Vector3(1f, 0f, 0f);
                        _frameStore[dst + jb + 0] = jawOk ? _jawAnimator.LastAngleDeg : 0f;
                        _frameStore[dst + jb + 1] = ja.x;
                        _frameStore[dst + jb + 2] = ja.y;
                        _frameStore[dst + jb + 3] = ja.z;
                        _frameStore[dst + jb + 4] = jd.x;
                        _frameStore[dst + jb + 5] = jd.y;
                        _frameStore[dst + jb + 6] = jd.z;
                        _frameStore[dst + jb + 7] = _rightEyeRot[0];
                        _frameStore[dst + jb + 8] = _rightEyeRot[1];
                        _frameStore[dst + jb + 9] = _leftEyeRot[0];
                        _frameStore[dst + jb + 10] = _leftEyeRot[1];

                        if (_frameCount < _frameCapacity) _frameCount++;
                        else _frameHead = (_frameHead + 1) % _frameCapacity;
                    }
                }

                if (sb != null) Debug.Log(sb.ToString());
            }

            if (postSw != null && !warmup)
            {
                postSw.Stop();
                Debug.Log($"[Audio2Face] 第 {_inferenceCount} 次推理完成 ONNX={_lastRunMs:F1}ms " +
                          $"解算30帧={postSw.Elapsed.TotalMilliseconds:F1}ms，队列 {PendingFrames} 帧");
            }
        }

        /// <summary>
        /// 解一帧 skin 权重。animator 开着时先过 animator（它输出绝对顶点），
        /// 关着时直接喂 prediction（delta）。返回的是求解器内部缓冲，用完即失效。
        /// </summary>
        public float[] SolveSkinAt(int row)
        {
            if (_skinAnimator != null && _skinAnimator.IsReady)
            {
                // 解算侧剖析：把「animator 顶点重建」和「blendshape 求解」分开计时。
                // 没有这个拆分就不知道每帧那 ~10ms 到底是谁花的
                // （实测 solver 只占 ~5.7ms，剩下的全在 animator 里）。
                _swSolve.Restart();
                float[] r;
                double tAnim;
                if (_skinVerticesMasked != null)
                {
                    // 快路径：只重建 solver 真正会读的掩码顶点（数值与全量路径一致）
                    _skinAnimator.AnimateMasked(_model.Prediction, row + _info.SkinOffset, _skinVerticesMasked, 0, _dt);
                    tAnim = _swSolve.Elapsed.TotalMilliseconds;
                    r = _skinSolver.SolveMasked(_skinVerticesMasked, 0);
                }
                else
                {
                    _skinAnimator.Animate(_model.Prediction, row + _info.SkinOffset, _skinVertices, 0, _dt);
                    tAnim = _swSolve.Elapsed.TotalMilliseconds;
                    r = _skinSolver.Solve(_skinVertices, 0);
                }
                _msAnim += tAnim;
                _msSolve += _swSolve.Elapsed.TotalMilliseconds - tAnim;
                if (++_solveFrames >= SolveProfilePeriod)
                {
                    Debug.Log($"[解算剖析] skin: animator={_msAnim / _solveFrames:F3} + solver={_msSolve / _solveFrames:F3} " +
                              $"= {(_msAnim + _msSolve) / _solveFrames:F3}ms/帧（{_solveFrames} 帧均值，顶点=" +
                              (_skinVerticesMasked != null
                                  ? $"{_skinVerticesMasked.Length / 3}(掩码)/{_skinVertices.Length / 3}(全量)"
                                  : $"{_skinVertices.Length / 3}") + "）");
                    _solveFrames = 0;
                    _msAnim = 0;
                    _msSolve = 0;
                }
                return r;
            }
            return _skinSolver.Solve(_model.Prediction, row + _info.SkinOffset);
        }

        private float[] SolveTongueAt(int row)
        {
            if (_tongueAnimator != null && _tongueAnimator.IsReady)
            {
                _swSolve.Restart();
                _tongueAnimator.Animate(_model.Prediction, row + _info.TongueOffset, _tongueVertices, 0);
                double tAnim = _swSolve.Elapsed.TotalMilliseconds;
                var r = _tongueSolver.Solve(_tongueVertices, 0);
                _msAnimT += tAnim;
                _msSolveT += _swSolve.Elapsed.TotalMilliseconds - tAnim;
                if (++_solveFramesT >= SolveProfilePeriod)
                {
                    Debug.Log($"[解算剖析] tongue: animator={_msAnimT / _solveFramesT:F3} + solver={_msSolveT / _solveFramesT:F3} " +
                              $"= {(_msAnimT + _msSolveT) / _solveFramesT:F3}ms/帧（{_solveFramesT} 帧均值）");
                    _solveFramesT = 0;
                    _msAnimT = 0;
                    _msSolveT = 0;
                }
                return r;
            }
            return _tongueSolver.Solve(_model.Prediction, row + _info.TongueOffset);
        }

        /// <summary>
        /// 静息脸标定：用<b>当前 prediction</b> 的中心帧求平均权重，存为静音基线。
        /// 之后每帧输出 = clamp(w - b)，静音 → 全 0，说话时只体现相对静息脸的形变。
        /// （不做 (w-b)/(1-b) 重归一化：那会在基线接近 1 时把残差放大成巨大权重，
        ///   是「嘴歪眼斜」的放大来源。官方 SDK 本就不做静息脸标定。）
        ///
        /// 用法：启动后先喂 1~2 秒静音、跑几次推理，再调一次本方法。
        /// 换身份 / 换情绪要重新标定。
        /// </summary>
        /// <returns>skin 基线权重（长度 SkinPoseCount）</returns>
        public float[] CalibrateRestPose()
        {
            int left = _info.FramesLeftTruncate;
            int center = _info.FramesCenter;
            int stride = _model.TotalDim;

            // 标定期间必须关掉旧基线，否则解出来的已经是减过基线的结果，会二次相减
            _skinSolver.WeightBaseline = null;
            if (_tongueSolver.IsReady) _tongueSolver.WeightBaseline = null;

            _skinAnimator?.Reset();
            // 标定静息脸时眼睛必须完全睁开，否则基线里会混进闭眼量
            _skinAnimator?.SetBlinkOffset(0f);
            _skinSolver.Reset();
            if (_tongueSolver.IsReady) _tongueSolver.Reset();

            var acc = new float[SkinPoseCount];
            for (int f = 0; f < center; f++)
            {
                var w = SolveSkinAt((left + f) * stride);
                if (w == null) break;
                for (int i = 0; i < SkinPoseCount; i++) acc[i] += w[i];
            }
            float inv = 1f / Mathf.Max(1, center);
            for (int i = 0; i < SkinPoseCount; i++) acc[i] *= inv;
            _skinSolver.WeightBaseline = acc;

            if (_tongueSolver.IsReady)
            {
                var tacc = new float[TonguePoseCount];
                for (int f = 0; f < center; f++)
                {
                    var w = SolveTongueAt((left + f) * stride);
                    if (w == null) break;
                    for (int i = 0; i < TonguePoseCount; i++) tacc[i] += w[i];
                }
                for (int i = 0; i < TonguePoseCount; i++) tacc[i] *= inv;
                _tongueSolver.WeightBaseline = tacc;
            }

            // 清掉时间正则状态：标定帧的 _prevWeights 不该带进正式播放
            _skinAnimator?.Reset();
            _skinSolver.Reset();
            if (_tongueSolver.IsReady) _tongueSolver.Reset();

            return acc;
        }

        /// <summary>清掉静息脸标定，回到原始权重。</summary>
        public void ClearRestPose()
        {
            _skinSolver.WeightBaseline = null;
            if (_tongueSolver.IsReady) _tongueSolver.WeightBaseline = null;
        }

        /// <summary>取出一帧（长度 FrameStride：前 SkinPoseCount 个是 skin，后面是 tongue）。</summary>
        public bool TryDequeueFrame(float[] dst)
        {
            if (dst == null || dst.Length < _frameStride) return false;

            lock (_frameLock)
            {
                if (_frameCount == 0) return false;
                Array.Copy(_frameStore, _frameHead * _frameStride, dst, 0, _frameStride);
                _frameHead = (_frameHead + 1) % _frameCapacity;
                _frameCount--;
                return true;
            }
        }

        /// <summary>
        /// 暂停后台推理。自检脚本必须先调这个再用 SkinSolver：
        /// 求解器内部复用 _target/_x/_result 缓冲，后台线程同时在解帧会把自检结果冲掉
        /// （表现是「自身权重对、残差乱跳」）。
        /// </summary>
        public void SetPaused(bool paused)
        {
            _workerPaused = paused;
            if (paused)
            {
                // 等当前这次跑完再返回
                for (int i = 0; i < 200 && _busy; i++) Thread.Sleep(10);
            }
        }

        /// <summary>
        /// 顶点空间的静音偏置标定。
        ///
        /// <b>实测这条路压不掉静音权重，保留它只是为了对照。</b>
        /// 静音输出有约 25% 能量落在 blendshape 张成的子空间之外，而且这部分还在缓慢漂移，
        /// 减掉顶点均值只削掉了漂移分量，BVLS 该解出的权重一个不少
        /// （jawOpen 0.080 → 0.075）。要归零请用 CalibrateRestPose（权重空间）。
        /// </summary>
        /// <returns>skin 段偏置的 RMS</returns>
        public float CalibrateSilenceBaseline(int frames)
        {
            if (frames <= 0)
            {
                _skinSolver.TargetBias = null;
                if (_tongueSolver.IsReady) _tongueSolver.TargetBias = null;
                return 0f;
            }

            int left = _info.FramesLeftTruncate;
            int stride = _model.TotalDim;

            var skinBias = new float[_skinSolver.SolvedPositionCount];
            ComputeMaskedBias(_skinSolver, _skinAnimator, _info.SkinOffset, left, stride, frames, skinBias);
            _skinSolver.TargetBias = skinBias;

            if (_tongueSolver.IsReady)
            {
                var tongueBias = new float[_tongueSolver.SolvedPositionCount];
                ComputeMaskedBias(_tongueSolver, _tongueAnimator, _info.TongueOffset, left, stride, frames, tongueBias);
                _tongueSolver.TargetBias = tongueBias;
            }

            double s = 0.0;
            for (int i = 0; i < skinBias.Length; i++) s += (double)skinBias[i] * skinBias[i];
            return (float)Math.Sqrt(s / Mathf.Max(1, skinBias.Length));
        }

        /// <summary>
        /// 算「求解器 target 空间」里的静态偏置。
        /// animator 开着时必须在 animator 输出上取均值再减 neutral ——
        /// 求解器是先把输入减 neutral、再减 bias，直接在 prediction 上取均值会差一个 neutral。
        /// </summary>
        private void ComputeMaskedBias(BlendshapeSolver solver, IA2FGeometryAnimator animator,
                                       int segOffset, int leftFrame, int stride, int frames, float[] bias)
        {
            if (animator != null && animator.IsReady)
            {
                int size = animator.Size;
                var all = new float[frames * size];
                for (int f = 0; f < frames; f++)
                    animator.Animate(_model.Prediction, (leftFrame + f) * stride + segOffset,
                                     all, f * size, _dt);
                solver.ComputeMaskedMean(all, 0, size, frames, bias);

                var n = solver.NeutralMasked;
                for (int i = 0; i < bias.Length && i < n.Length; i++) bias[i] -= n[i];
            }
            else
            {
                solver.ComputeMaskedMean(_model.Prediction, leftFrame * stride + segOffset,
                                         stride, frames, bias);
            }
        }

        /// <summary>丢掉已排队但未消费的帧（标定时会产生一批静音帧，不能留给播放）。</summary>
        public void ClearFrameQueue()
        {
            lock (_frameLock)
            {
                _frameHead = 0;
                _frameCount = 0;
            }
        }

        /// <summary>在调用线程上同步跑一个窗口（自检用）。调用前必须先 SetPaused(true)，
        /// 否则会和后台线程抢 _window / 求解器缓冲。
        /// </summary>
        public void RunNow(float[] window)
        {
            Array.Copy(window, _window, Mathf.Min(window.Length, _window.Length));
            var sw = System.Diagnostics.Stopwatch.StartNew();
            _model.Run(_window);
            _lastRunMs = (float)sw.Elapsed.TotalMilliseconds;
            SolveCenterFrames();
            sw.Stop();
            _lastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;
            _lastPostMs = Mathf.Max(0f, _lastInferenceMs - _lastRunMs);
        }

        /// <summary>换噪声种子（自检用）。</summary>
        public void SetNoiseSeed(int seed) => _model.SetNoiseSeed(seed);

        /// <summary>设置情绪（10 维），来自 Audio2Emotion 的 FinalOutput。</summary>
        public void SetEmotion(float[] emotions) => _model.SetEmotion(emotions);

        public void Reset()
        {
            _model.ResetState();
            _skinSolver.Reset();
            if (_tongueSolver.IsReady) _tongueSolver.Reset();
            _skinAnimator?.Reset();
            _eyesAnimator?.Reset();
            // 眨眼排期也要回到原点，并把 animator 的 blinkOffset 清掉 ——
            // 否则标定静息脸时会带着上一次残留的闭眼量，静息基线就偏了。
            _blink?.Reset(_config != null ? _config.blinkSeed : 0);
            _skinAnimator?.SetBlinkOffset(0f);
            _ringHead = 0;
            _ringTotal = 0;
            _consumedUntil = 0;
            _inferenceCount = 0;
            _windowEnd = 0;
            lock (_frameLock)
            {
                _frameHead = 0;
                _frameCount = 0;
            }
        }

        public void Dispose()
        {
            _workerRunning = false;
            if (_worker != null && _worker.IsAlive)
            {
                _worker.Join(500);
                _worker = null;
            }
            _model?.Dispose();
        }
    }
}
