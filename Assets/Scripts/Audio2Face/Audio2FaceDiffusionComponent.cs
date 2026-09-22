using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// 挂载到数字人 GameObject 上，自动把 Audio2Face v3.0 扩散模型的输出
    /// 写到 SkinnedMeshRenderer 的 blendshape 上。
    /// </summary>
    public class Audio2FaceDiffusionComponent : MonoBehaviour
    {
        [Header("配置")]
        public Audio2FaceDiffusionConfig Config = new Audio2FaceDiffusionConfig();

        [Header("音频输入")]
        [Tooltip("勾选后 Start 时自动开麦克风；否则需要外部调用 PushAudio")]
        public bool AutoStartMicrophone = false;
        public string MicrophoneDevice = "";

        [Header("播放")]
        [Tooltip("按模型帧率消费帧队列，一般保持 true")]
        public bool RealtimePlayback = true;
        [Tooltip("整体强度缩放，1 = 原样")]
        public float Strength = 1.0f;
        [Tooltip("处理 AudioClip 时是否同时播放音频")]
        public bool PlayAudioWhileProcessing = true;

        [Header("静息脸标定")]
        [Tooltip("默认 true：constantNoise 噪声恒定，Start 时安静推理得到固定噪声地板，\n" +
                 "标定可精确减去它，音频权重从零开始；stylization lip_close_offset 负责自然闭合。\n" +
                 "关闭会让静音时仍有恒定偏置权重，容易表现为无语音关联的挤眉微张。")]
        public bool CalibrateRestPoseOnStart = true;

        [Header("骨骼驱动（SDK 的 jaw / eyes 输出，不走 blendshape）")]
        [Tooltip("下颌骨。jaw 段（15 维）解出的 4x4 刚体变换会转成「绕轴角度 + 位移」。\n" +
                 "牙齿挂在骨骼上，只有这一步能让它跟着嘴动 —— 纯 blendshape 是带不动牙齿的。\n" +
                 "纯 blendshape 网格（没有骨骼）不用填。")]
        public Transform JawBone;
        [Tooltip("下颌旋转轴（骨骼局部空间）。多数角色是 X 轴；转反了把 JawAngleSign 取 -1")]
        public Vector3 JawRotationAxis = new Vector3(1f, 0f, 0f);
        public float JawAngleSign = 1f;
        [Tooltip("下颌旋转的放大倍数。A2F 解出的下颌角是给「下牙床」用的，实测峰值只有 3~4°，\n" +
                 "直接套到下颌骨上几乎看不出来。设 3 左右（≈10~12°）比较接近真人口语开合。")]
        public float JawAngleGain = 3f;
        [Tooltip("是否用 A2F 解出的旋转轴（而不是上面的固定轴）。旋转看着怪就关掉，用固定轴。")]
        public bool UseFittedJawAxis = true;
        [Tooltip("下颌位移方向（Unity 局部空间）。A2F 的 Y 是高度轴，默认同向。\n" +
                 "这里用的是「下颌标记点质心的真实位移」，不是 4x4 里的平移（那个含绕模型原点的杠杆臂，\n" +
                 "量级大十倍，会把下颌整个推出去）。")]
        public Vector3 JawTranslationAxis = new Vector3(0f, 1f, 0f);
        [Tooltip("A2F 位移单位 → Unity 世界单位的缩放。A2F 用厘米量级，Unity 角色按米建模时填 0.01")]
        public float JawTranslationScale = 0.01f;

        [Tooltip("左右眼球骨骼。eyes 段（4 维）+ saccade 微跳视会转成欧拉角（度）")]
        public Transform LeftEyeBone;
        public Transform RightEyeBone;
        public float EyeRotationScale = 1f;

        [Header("事件")]
        public UnityEngine.Events.UnityEvent<float[]> OnFrameApplied;

        private Audio2FaceDiffusionPipeline _pipeline;
        private SkinnedMeshRenderer _skinnedMesh;

        /// <summary>一张网格的 pose → blendshape 下标映射。</summary>
        private sealed class MeshBinding
        {
            public SkinnedMeshRenderer renderer;
            public int[] skin;
            public int[] tongue;
        }

        /// <summary>
        /// 所有要驱动的网格。数字人通常不止一张：脸 / 舌头 / 牙齿 / 眼球各自带 blendshape，
        /// 只驱动第一张的话舌头和牙齿永远不动 —— 这是「嘴张开了却看不到牙和舌头」的常见原因。
        /// </summary>
        private readonly List<MeshBinding> _bindings = new List<MeshBinding>();

        private readonly float[] _frame = new float[512];

        private AudioClip _micClip;
        private int _micReadPos;
        private readonly float[] _micBuffer = new float[4096];

        // 骨骼驱动：jaw 刚体变换 + 左右眼欧拉角（都来自 SDK 的非 blendshape 输出）
        private Quaternion _jawInitRot = Quaternion.identity;
        private Vector3 _jawInitPos = Vector3.zero;
        private Quaternion _leftEyeInitRot = Quaternion.identity;
        private Quaternion _rightEyeInitRot = Quaternion.identity;
        private float _lastJawDrop;

        // ---- 唇缝开合度量 ----
        // 「嘴巴到底张没张」不能靠 jawOpen 权重判断：权重是求解器在 Mark 模板上的分解结果，
        // 落到不同网格上效果差很多。这里直接量唇部顶点在当前权重下的实际位移。
        //
        // 做法（v3，前两版都测错了，详见 BuildLipProbe 的注释）：
        //   1. 上唇组 = mouthUpperUpLeft 形变最集中的顶点；下唇组 = mouthLowerDownLeft 形变最集中的顶点。
        //      用 ARKit 语义最明确的两个形状当定位器，不猜任何方向。
        //   2. 每个 pose 的系数 = mean(上唇组 Δy) − mean(下唇组 Δy)。
        //      下唇下降 / 上唇上抬 → 正（张开）；闭唇 → 负。符号由构造保证。
        //
        // v2 用 mouthClose 的 Δy 符号分组，Sloth_Head2 上 mouthClose 的显著顶点 100% 是 Δy<0
        // （上唇 963 / 下唇 0），分组退化，jawOpen 与 mouthClose 算出同号，完全没法用。
        private float[] _lipCoeff;         // 每个 skin pose 对唇缝开合的系数（网格本地单位）
        private int[] _lipUpperIdx;        // 上唇顶点（mouthUpperUpLeft 形变集中区）
        private int[] _lipLowerIdx;        // 下唇顶点（mouthLowerDownLeft 形变集中区）
        private const int LipProbeVertices = 60;   // 每组最多取多少个顶点
        private Vector3 _lipNormal;        // 唇缝张开方向（静息上唇质心 − 下唇质心，已归一化）
        private float _lipRestGap;         // 静息时上下唇质心距离（= 唇厚，aperture 的零点）
        /// <summary>自检时把「最大张口应该到多少」作为目标（% 网格高）。正常说话 2~6%。</summary>
        private const float LipOpenTargetPercent = 3.0f;
        private float _lipRefHeight = 1f;  // 归一化基准：网格包围盒高度
        private bool _lipProbeReady;
        private int _lipRegionCount;
        private int _iJawOpenCoeff = -1;
        private byte[] _lipGroup;          // 0=其它 1=jawOpen 2=唇分开组 3=闭唇组
        private static readonly string[] LipOpenPoses =
            { "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight" };
        private static readonly string[] LipSealPoses =
            { "mouthClose", "mouthPressLeft", "mouthPressRight", "mouthShrugLower", "mouthRollLower" };

        private float _apertureMin = float.MaxValue;
        private float _apertureMax = float.MinValue;
        private int _apertureFrames;
        // 开合最大那一帧的成因分解，用来判断「到底是谁把嘴撑开的」
        private float _apJaw, _apLipOpen, _apSeal, _apOther;
        private float _apJawW;             // 开合最大那一帧的 jawOpen 权重本身

        // 真值校验：不看权重、不看系数，直接烘网格量唇缝（见 SampleRealAperture）
        private Mesh _bakeMesh;            // BakeMesh 用的临时网格（探针 + 实测共用）
        private float _realMin = float.MaxValue, _realMax = float.MinValue;
        private int _realSamples;
        private float _realJawMax;
        private int _appliedSinceSample;
        // ApplyFrame 心跳：确认「Component 是否在把帧写进 _bindings 网格」（与 Tester 的 [A2F驱动] 对照定位）
        private int _applyFrames;
        private float _applyJawMax;

        private float _frameAccumulator;
        private bool _started;

        // ---- 音频时钟同步 ----
        // SDK 语义（executor_diffusion_core.cpp GetProgressParameters）：
        //   第 k 个窗口 start = 8000k - 16000，target = start + targetOffset(4000)。
        //   中心帧 = [left=15, 45)，帧 f 的音频时刻 = start + f * (bufferLength / 60)。
        //   所以第一个正式输出帧（f=15）对应的音频样本 = targetOffset = 4000（0.25 秒），
        //   而不是 0。播放必须按「音频时刻 → 帧号」反查，不能自由按 60fps 累加，
        //   否则队列一积压/一耗尽就会永久漂移。
        private AudioSource _audioSource;
        private bool _audioPlaying;
        private float _targetOffsetSec;
        private int _displayedFrames;
        private int _lastSyncLogFrame;
        // 预缓冲 / 预热期间挂起帧消费：见 Update 里累加器分支的说明
        private bool _holdFrames;

        public bool IsReady => _pipeline != null && _started;
        public float LastInferenceMs => _pipeline != null ? _pipeline.LastInferenceMs : 0f;
        public int PendingFrames => _pipeline != null ? _pipeline.PendingFrames : 0;

        /// <summary>求解器输出的 pose 名（标准 ARKit 52），供测试/映射工具复用。</summary>
        public string[] PoseNames => _pipeline != null && _pipeline.SkinSolver != null
            ? _pipeline.SkinSolver.PoseNames : null;

        /// <summary>实际驱动 blendshape 的网格。</summary>
        public SkinnedMeshRenderer TargetMesh => _skinnedMesh;

        void Start()
        {
            if (Config != null && Config.Migrate())
                Debug.Log($"[Audio2Face] 配置已升级到 v{Audio2FaceDiffusionConfig.LatestConfigVersion}：" +
                          "清掉了旧版本残留在场景里的 lowerFaceStrength / jawOpenToLipOpen 等值 —— " +
                          "Unity 会一直用「字段首次出现时」的序列化值，改代码默认值不会生效。");

            if (Config.autoApplyToSkinnedMesh)
            {
                _skinnedMesh = GetComponent<SkinnedMeshRenderer>();
                if (_skinnedMesh == null) _skinnedMesh = GetComponentInChildren<SkinnedMeshRenderer>();
                if (_skinnedMesh == null) _skinnedMesh = TryUpgradeMeshRenderer();
                if (_skinnedMesh == null)
                    Debug.LogWarning("[Audio2Face] 没有找到 SkinnedMeshRenderer（也找不到可升级的 MeshRenderer），" +
                                     "blendshape 不会生效");
            }

            try
            {
                _pipeline = new Audio2FaceDiffusionPipeline(Config);
                _started = true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Audio2Face] 管线初始化失败: {ex.Message}\n{ex.StackTrace}");
                enabled = false;
                return;
            }

            if (_skinnedMesh != null)
            {
                // 收集全部带 blendshape 的网格（脸 / 舌头 / 牙齿 / 眼球）。
                // fallbackToIndexMapping 只对主网格生效：给舌头网格按 skin pose 下标硬套会直接错位。
                var all = GetComponentsInChildren<SkinnedMeshRenderer>(true);
                foreach (var r in all)
                {
                    if (r == null || r.sharedMesh == null || r.sharedMesh.blendShapeCount == 0) continue;
                    AddBinding(r, r == _skinnedMesh);
                }

                int totalSkin = 0;
                int totalTongue = 0;
                foreach (var b in _bindings)
                {
                    totalSkin += CountHit(b.skin);
                    totalTongue += CountHit(b.tongue);
                }
                if (totalSkin == 0)
                    Debug.LogError("[Audio2Face] 0 个 pose 能对应上网格 blendshape，口型一定不会动。" +
                                   "看上面的映射表检查命名（前缀 / _L _R / Left Right）。");
                else if (Config.debugMode)
                    Debug.Log($"[Audio2Face] 共绑定 {_bindings.Count} 张网格，skin 命中 {totalSkin} 通道，tongue 命中 {totalTongue} 通道");

                LogMouthBinding();
                if (JawBone == null)
                    Debug.LogWarning("[A2F骨骼] 没填 Jaw Bone —— 下颌刚体变换没有落点，挂在骨骼上的牙齿不会动。" +
                                     "A2F 的 jaw 段是 5 个标记点拟合出的刚体变换（角度 + 平移），只能作用在骨骼上， " +
                                     "blendshape 带不动它。如果 Sloth_Head2 是纯 blendshape 网格就忽略这条。");

                BuildLipProbe();
                SelfTestMouthOpen();
            }

            if (JawBone != null)
            {
                _jawInitRot = JawBone.localRotation;
                _jawInitPos = JawBone.localPosition;
            }
            if (LeftEyeBone != null) _leftEyeInitRot = LeftEyeBone.localRotation;
            if (RightEyeBone != null) _rightEyeInitRot = RightEyeBone.localRotation;

            if (CalibrateRestPoseOnStart) CalibrateRestPose();

            if (AutoStartMicrophone) StartMicrophone();

            // RealtimePlayback 是「消费帧队列」的总闸（Update 里 if(!RealtimePlayback) return）。
            // 关着时 ApplyFrame 从不会被调用 → OnFrameApplied 不触发 → 嘴永远静止（但手动扫掠仍会动）。
            // 这是「扫掠能张嘴、播放并驱动不张嘴」的头号嫌疑，先明确报出来。
            if (!RealtimePlayback)
                Debug.LogWarning("[Audio2Face] RealtimePlayback = false：Update 不会消费帧队列，" +
                                 "「播放并驱动」/麦克风都不会动嘴（只有测试台的手动扫掠会动）。" +
                                 "这就是「扫掠能张嘴、播放并驱动不张嘴」的典型原因 —— 请把它勾回 true。");
        }

        /// <summary>给一张网格建 pose → blendshape 映射并登记。</summary>
        private void AddBinding(SkinnedMeshRenderer r, bool allowFallback)
        {
            foreach (var b in _bindings)
                if (b.renderer == r) return;

            var binding = new MeshBinding { renderer = r };
            binding.skin = ArkItBlendshapeNames.MapPoses(
                r.sharedMesh, _pipeline.SkinSolver.PoseNames,
                allowFallback && Config.fallbackToIndexMapping, out int matched, out string report);

            if (Config.debugMode)
                Debug.Log($"[A2F Map] <{r.name}> skin 匹配 {matched}/{_pipeline.SkinSolver.PoseNames.Length}\n{report}");

            if (_pipeline.TongueSolver != null && _pipeline.TongueSolver.IsReady)
            {
                binding.tongue = ArkItBlendshapeNames.MapPoses(
                    r.sharedMesh, _pipeline.TongueSolver.PoseNames, false,
                    out int tmatched, out string treport);
                if (Config.debugMode && tmatched > 0)
                    Debug.Log($"[A2F Map] <{r.name}> tongue 匹配 {tmatched}/{_pipeline.TongueSolver.PoseNames.Length}\n{treport}");
            }

            _bindings.Add(binding);
        }

        private static int CountHit(int[] map)
        {
            if (map == null) return 0;
            int c = 0;
            for (int i = 0; i < map.Length; i++)
                if (map[i] >= 0) c++;
            return c;
        }

        /// <summary>
        /// 打印「嘴部 / 舌头」关键通道在所有网格上的绑定情况。
        ///
        /// 「下巴动了但嘴没张开 / 看不到舌头」有相当一部分根本不是求解问题，而是网格上没有
        /// 对应的 blendshape —— 权重算得再对也驱动不了。先确认这一层，再谈调参。
        /// </summary>
        private void LogMouthBinding()
        {
            if (_bindings.Count == 0 || _pipeline == null) return;

            var skinNames = _pipeline.SkinSolver != null ? _pipeline.SkinSolver.PoseNames : null;
            var tongueNames = _pipeline.TongueSolver != null ? _pipeline.TongueSolver.PoseNames : null;
            string[] keys =
            {
                "jawOpen", "mouthClose", "mouthFunnel",
                "mouthLowerDownLeft", "mouthLowerDownRight",
                "mouthUpperUpLeft", "mouthUpperUpRight",
                "mouthPressLeft", "mouthPressRight",
                "mouthShrugLower", "mouthRollLower"
            };

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"[A2F绑定] 共 {_bindings.Count} 张网格参与驱动");
            foreach (var b in _bindings)
            {
                var m = b.renderer.sharedMesh;
                sb.AppendLine($"  <{b.renderer.name}> mesh={m.name} 形状={m.blendShapeCount} " +
                              $"skin命中={CountHit(b.skin)} tongue命中={CountHit(b.tongue)}");
            }

            sb.AppendLine("  嘴部关键通道 → 实际生效的网格形状");
            int miss = 0;
            foreach (var key in keys)
            {
                int p = skinNames != null ? Array.IndexOf(skinNames, key) : -1;
                string hit = "✗ 所有网格都没有（该通道不会动）";
                bool any = false;
                if (p >= 0)
                {
                    foreach (var b in _bindings)
                    {
                        if (b.skin == null || p >= b.skin.Length || b.skin[p] < 0) continue;
                        hit = $"→ <{b.renderer.name}> #{b.skin[p]:D2} {b.renderer.sharedMesh.GetBlendShapeName(b.skin[p])}";
                        any = true;
                        break;
                    }
                }
                if (!any) miss++;
                sb.AppendLine($"    {key,-20} {hit}");
            }

            var hits = new List<string>();
            foreach (var b in _bindings)
            {
                if (b.tongue == null || tongueNames == null) continue;
                for (int i = 0; i < b.tongue.Length && i < tongueNames.Length; i++)
                    if (b.tongue[i] >= 0 && !hits.Contains(tongueNames[i])) hits.Add(tongueNames[i]);
            }
            sb.AppendLine($"  舌头通道命中 {hits.Count}/{(_pipeline.TonguePoseCount)}: " +
                          (hits.Count > 0 ? string.Join(", ", hits) : "（无）"));
            Debug.Log(sb.ToString());

            if (hits.Count == 0)
                Debug.LogWarning("[A2F绑定] 没有任何网格带舌头 blendshape —— 舌头不可能动。" +
                                 "Audio2Face 的 16 个 tongue pose（tongueTipUp / tongueRollUp / tongueIn ...）" +
                                 "需要在网格上有同名或同义形状；没有的话要么补形状，要么放弃露舌。");
            if (miss > 0)
                Debug.LogWarning($"[A2F绑定] {miss} 个嘴部关键通道在所有网格上都缺失，张嘴幅度会打折。" +
                                 "看上面的表补形状或改命名（前缀 / _L _R / Left Right）。");
        }

        /// <summary>
        /// 静息脸标定。喂静音跑几轮推理（跨过预热），取中心帧的平均权重当基线，
        /// 之后每帧输出 = clamp(w - b)，静音 → 全 0。
        ///
        /// 用的是模型自己的静音输出，不需要麦克风，所以随时可以重标
        /// （换身份、换情绪、改 animator 参数之后都应该重标一次）。
        /// </summary>
        public void CalibrateRestPose()
        {
            if (_pipeline == null) return;

            var info = _pipeline.Info;
            var silence = new float[info.BufferLength];

            _pipeline.SetPaused(true);
            try
            {
                _pipeline.Reset();
                for (int i = 0; i < info.WarmupInferences + 1; i++) _pipeline.RunNow(silence);

                var baseline = _pipeline.CalibrateRestPose();
                _pipeline.ClearFrameQueue();      // 标定产生的静音帧不能留给播放

                int jaw = Array.IndexOf(_pipeline.SkinSolver.PoseNames, "jawOpen");
                if (Config.debugMode)
                    Debug.Log($"[Audio2Face] 静息脸标定完成: " +
                              (jaw >= 0 ? $"jawOpen 基线={baseline[jaw]:F3} " : "") +
                              $"权重合计={Sum(baseline):F3}");
                // 标定后立即跑一次静音推理，验证基线是否真正抵消
                if (Config.debugMode)
                {
                    var testSilence = new float[info.BufferLength];
                    _pipeline.RunNow(testSilence);
                    var testW = _pipeline.SolveSkinAt((info.FramesLeftTruncate + info.FramesCenter / 2) * _pipeline.Info.TotalDim);
                    float testSum = 0f;
                    for (int i = 0; i < testW.Length; i++) testSum += testW[i];
                    Debug.Log($"[Audio2Face] 标定后静音验证: 权重合计={testSum:F3} (应接近0)");
                }
            }
            finally
            {
                _pipeline.SetPaused(false);
            }
        }

        private static float Sum(float[] a)
        {
            float s = 0f;
            for (int i = 0; i < a.Length; i++) s += a[i];
            return s;
        }

        void OnDestroy()
        {
            StopMicrophone();
            _pipeline?.Dispose();
            _pipeline = null;
        }

        void Update()
        {
            if (_pipeline == null) return;

            if (_micClip != null) PollMicrophone();

            _pipeline.Tick();

            if (!RealtimePlayback) return;

            float fps = _pipeline.Info != null ? _pipeline.Info.FrameRate : 60f;

            // 优先用音频时钟驱动：帧 j 的音频时刻 = targetOffsetSec + j / fps。
            // 这样队列积压（推理快于实时）或短暂耗尽（推理慢于实时）都不会造成永久漂移。
            if (_audioPlaying && _audioSource != null && _audioSource.isPlaying)
            {
                double targetFrame = (_audioSource.time - _targetOffsetSec) * fps;
                int want = (int)Math.Floor(targetFrame) - _displayedFrames;
                if (want <= 0) return;

                // 卡顿/掉帧落后过多 → 丢帧追赶，避免越拖越久。
                // ⚠ 只能按「真正取出来」的帧数推进 _displayedFrames：队列里帧不够时原来的写法
                // 仍然把 drop 全加进计数，_displayedFrames 就超前于实际消费，动画相对音频
                // 永久滞后（drop - 实际取到）帧，而且再也追不回来。
                const int maxCatchUp = 8;
                if (want > maxCatchUp)
                {
                    int drop = want - maxCatchUp;
                    int dropped = 0;
                    for (int i = 0; i < drop; i++)
                    {
                        if (!_pipeline.TryDequeueFrame(_frame)) break;
                        dropped++;
                    }
                    _displayedFrames += dropped;
                    want -= dropped;
                    if (want <= 0) return;
                }

                for (int i = 0; i < want; i++)
                {
                    // 队列空：本帧不消费、也不推进 _displayedFrames，下一帧继续补 —— 不会永久漂移
                    if (!_pipeline.TryDequeueFrame(_frame)) break;
                    ApplyFrame(_frame);
                    _displayedFrames++;
                }

                // 同步诊断：最后一帧「自带的音频时刻」应该紧跟 audioSource.time。
                // 两者差 = 动画相对声音的偏移（毫秒）。持续为正 = 动画超前，为负 = 滞后。
                if (Config.debugMode && _displayedFrames - _lastSyncLogFrame >= 30)
                {
                    _lastSyncLogFrame = _displayedFrames;
                    float frameT = _targetOffsetSec + (_displayedFrames - 1) / fps;
                    Debug.Log($"[A2F同步] 音频={_audioSource.time:F3}s 动画帧时刻={frameT:F3}s " +
                              $"差={(frameT - _audioSource.time) * 1000f:F0}ms 已显示={_displayedFrames} 队列={_pipeline.PendingFrames}");
                }
                return;
            }

            // 无音频时钟（麦克风 / 未播放音频）：退回固定帧率累加器。
            // ⚠ 预缓冲期间必须挡住：这段路径不看音频时钟，会以 60fps 白消费帧，
            // 等真正开播时队列头的帧已经对应后面几秒的音频，动画整体超前（实测超前 2.7s）。
            if (_holdFrames) return;

            _frameAccumulator += Time.deltaTime * fps;
            int budget = (int)_frameAccumulator;
            if (budget <= 0) return;

            int consumed = 0;
            for (int i = 0; i < budget; i++)
            {
                if (!_pipeline.TryDequeueFrame(_frame)) break;
                ApplyFrame(_frame);
                consumed++;
            }
            // 只扣掉真正消费掉的帧数；没消费的留到下一帧重试，避免队列空时吞掉预算造成漂移
            _frameAccumulator -= consumed;
        }

        private void ApplyFrame(float[] frame)
        {
            int tongueBase = _pipeline != null ? _pipeline.SkinPoseCount : 0;
            for (int t = 0; t < _bindings.Count; t++)
            {
                var b = _bindings[t];
                if (b.renderer == null) continue;

                if (b.skin != null)
                {
                    for (int i = 0; i < b.skin.Length; i++)
                    {
                        int bi = b.skin[i];
                        if (bi < 0) continue;
                        b.renderer.SetBlendShapeWeight(bi, Mathf.Clamp(frame[i] * Strength, 0f, 1f) * 100f);
                    }
                }

                if (b.tongue != null)
                {
                    for (int i = 0; i < b.tongue.Length; i++)
                    {
                        int bi = b.tongue[i];
                        if (bi < 0) continue;
                        b.renderer.SetBlendShapeWeight(bi, Mathf.Clamp(frame[tongueBase + i] * Strength, 0f, 1f) * 100f);
                    }
                }
            }

            // jaw / eyes：骨骼驱动。这两段在 SDK 里就不走 blendshape，
            // 缺了它们牙齿不会动、眼睛会死盯前方。
            if (_pipeline != null)
            {
                int jb = _pipeline.JawChannelOffset;

                if (JawBone != null && jb + Audio2FaceDiffusionPipeline.JawChannelCount <= frame.Length)
                {
                    // 角度 × 增益 × 方向。轴默认用 A2F 拟合出来的（更贴合它的坐标系），
                    // 看着别扭就关掉 UseFittedJawAxis 用固定轴。
                    Vector3 axis = JawRotationAxis;
                    if (UseFittedJawAxis)
                    {
                        var fa = new Vector3(frame[jb + 1], frame[jb + 2], frame[jb + 3]);
                        if (fa.sqrMagnitude > 1e-6f) axis = fa.normalized;
                    }
                    JawBone.localRotation = _jawInitRot *
                        Quaternion.AngleAxis(frame[jb] * JawAngleGain * JawAngleSign, axis);

                    // 质心真实位移（jb+4..6），不是 4x4 的平移
                    float drop = -frame[jb + 5];              // A2F 的 Y 是高度轴，张嘴是负方向
                    JawBone.localPosition = _jawInitPos + JawTranslationAxis * (drop * JawTranslationScale);
                    _lastJawDrop = drop;
                }

                int eb = jb + Audio2FaceDiffusionPipeline.JawChannelCount;
                if (RightEyeBone != null && eb + 2 <= frame.Length)
                    RightEyeBone.localRotation = _rightEyeInitRot *
                        Quaternion.Euler(frame[eb] * EyeRotationScale, frame[eb + 1] * EyeRotationScale, 0f);

                if (LeftEyeBone != null && eb + 4 <= frame.Length)
                    LeftEyeBone.localRotation = _leftEyeInitRot *
                        Quaternion.Euler(frame[eb + 2] * EyeRotationScale, frame[eb + 3] * EyeRotationScale, 0f);
            }

            MeasureMouthAperture(frame);
            SampleRealAperture(frame);

            // 心跳（无论 debugMode）：这条出现 = Component.ApplyFrame 在跑、帧在写 _bindings 网格。
            // 与 Tester 的 [A2F驱动] 心跳对照就能定位「播放并驱动」：
            //   两条都有 → 帧在流，去 [A2F实测] 看嘴真开没开（没开 = 幅度/基线问题，调 jawOpenGain）。
            //   只有 [A2F写入]、没有 [A2F驱动] → OnFrameApplied 没连上 OnDriverFrame（查 Awake 的 AddListener）。
            //   两条都有但数值对不上（写入≈1.00、驱动≈0.x）→ Tester 的 pose 序跟 frame 不同序，
            //   权重被写串了（Tester.EnsureMap 应已自动修复；再看到这句就查它的 ResolvePoseNames）。
            //   两条都没有 → Update 根本没消费帧（查 RealtimePlayback / 音频时钟 / _holdFrames）。
            {
                float jw = (_iJawOpenCoeff >= 0 && _iJawOpenCoeff < frame.Length)
                    ? Mathf.Clamp(frame[_iJawOpenCoeff] * Strength, 0f, 1f) : -1f;
                if (jw > _applyJawMax) _applyJawMax = jw;
                if (++_applyFrames % 60 == 0)
                {
                    Debug.Log($"[A2F写入] ApplyFrame 已写 {_applyFrames} 帧 | 本秒写入 _bindings 的 jawOpen 峰值={_applyJawMax:F2}" +
                              $" (Strength={Strength:F2})");
                    _applyJawMax = 0f;
                }
            }

            OnFrameApplied?.Invoke(frame);
        }

        /// <summary>
        /// 建「唇缝开合」探针（v4）。
        ///
        /// 前三版都测错了，别再退回去：
        ///   v1 单点顶点 + 反号 → mouthClose 恒正、jawOpen 恒 ~0（单点很可能落在嘴角）。
        ///   v2 用 mouthClose 的 Δy <b>符号</b> 分上下唇 → Sloth_Head2 上 mouthClose 的显著顶点
        ///      100% 都是 Δy&lt;0（日志：上唇 963 / 下唇 0），分组退化成一坨，jawOpen 与 mouthClose
        ///      算出同号（−9.42% / −9.64%），张开和闭合根本分不出来。
        ///   v3 靠形状语义定位上下唇（对），但系数是「Δ 位移投影到某个方向」—— 方向取自
        ///      <c>mesh.vertices</c>，而 Sloth_Head2 没开 Read/Write，Unity 返回空数组，
        ///      探针在建的那一步就 IndexOutOfRange 死了（整轮 [A2F唇缝] 数据作废）。
        ///
        /// v4：<b>不推算，直接量</b>。每个 pose 打满权重烘一次 BakeMesh，用上下唇质心距离的变化
        ///     当系数。BakeMesh 不要求 Read/Write，也不依赖网格轴向，Unity 的 blendshape 又本来
        ///     就是线性叠加，所以 Σ w_i · coeff_i 是精确解而不是近似。
        ///     代价是启动时烘 52 次（几十毫秒，只在 Start 里做一次）。
        /// </summary>
        private void BuildLipProbe()
        {
            if (_bindings.Count == 0 || _pipeline == null) return;
            var b = _bindings[0];
            var smr = b.renderer;
            var mesh = smr != null ? smr.sharedMesh : null;
            if (mesh == null || b.skin == null) return;

            var poseNames = _pipeline.SkinSolver != null ? _pipeline.SkinSolver.PoseNames : null;
            if (poseNames == null) return;

            try
            {
                int vc = mesh.vertexCount;
                var dv = new Vector3[vc];
                var dn = new Vector3[vc];
                var dt = new Vector3[vc];

                int upPose = Array.IndexOf(poseNames, "mouthUpperUpLeft");
                int loPose = Array.IndexOf(poseNames, "mouthLowerDownLeft");
                int upShape = (upPose >= 0 && upPose < b.skin.Length) ? b.skin[upPose] : -1;
                int loShape = (loPose >= 0 && loPose < b.skin.Length) ? b.skin[loPose] : -1;
                if (upShape < 0 || loShape < 0 ||
                    upShape >= mesh.blendShapeCount || loShape >= mesh.blendShapeCount ||
                    mesh.GetBlendShapeFrameCount(upShape) <= 0 || mesh.GetBlendShapeFrameCount(loShape) <= 0)
                {
                    Debug.LogWarning("[A2F唇缝] 网格上没有 mouthUpperUpLeft / mouthLowerDownLeft 形状，无法建开合探针");
                    return;
                }

                // 1) 用这两个形状各自的形变集中区定位上唇 / 下唇顶点
                mesh.GetBlendShapeFrameVertices(upShape, 0, dv, dn, dt);
                _lipUpperIdx = PickStrongestVertices(dv, vc, LipProbeVertices);
                mesh.GetBlendShapeFrameVertices(loShape, 0, dv, dn, dt);
                _lipLowerIdx = PickStrongestVertices(dv, vc, LipProbeVertices);
                if (_lipUpperIdx == null || _lipLowerIdx == null ||
                    _lipUpperIdx.Length == 0 || _lipLowerIdx.Length == 0)
                {
                    Debug.LogWarning("[A2F唇缝] 上/下唇顶点定位失败，无法建开合探针");
                    return;
                }

                // 1b) 静息唇缝基准。
                //     ⚠ 不能用 mesh.vertices：Sloth_Head2 没开 Read/Write，Unity 只打一条
                //     "Not allowed to access vertices" 然后返回空数组（不抛异常），下一行索引
                //     直接越界 —— 上一轮日志里探针就是这么死的。
                //     BakeMesh 不要求 Read/Write，而且拿到的是「当前权重下的真实姿态」，
                //     下面量系数也用同一个 Gap()，空间与量纲完全自洽。
                if (_bakeMesh != null) Destroy(_bakeMesh);
                _bakeMesh = new Mesh();

                var savedW = new float[mesh.blendShapeCount];
                for (int i = 0; i < savedW.Length; i++) savedW[i] = smr.GetBlendShapeWeight(i);

                void ClearAll()
                {
                    for (int i = 0; i < mesh.blendShapeCount; i++) smr.SetBlendShapeWeight(i, 0f);
                }
                // 上下唇质心距离：静息≈唇厚，张开就变大。用距离而不是 Y 差 → 与网格轴向无关。
                float Gap()
                {
                    smr.BakeMesh(_bakeMesh);
                    var vs = _bakeMesh.vertices;
                    if (vs == null || vs.Length == 0) return 0f;
                    var u = Vector3.zero;
                    var l = Vector3.zero;
                    for (int k = 0; k < _lipUpperIdx.Length; k++)
                        u += vs[Mathf.Min(_lipUpperIdx[k], vs.Length - 1)];
                    for (int k = 0; k < _lipLowerIdx.Length; k++)
                        l += vs[Mathf.Min(_lipLowerIdx[k], vs.Length - 1)];
                    return (u / _lipUpperIdx.Length - l / _lipLowerIdx.Length).magnitude;
                }

                ClearAll();
                float g0 = Gap();
                _lipRestGap = g0;
                _lipRefHeight = _bakeMesh.bounds.size.y > 1e-6f ? _bakeMesh.bounds.size.y : 1f;
                _lipNormal = Vector3.up;      // v4 起系数是实测标量，不再需要投影方向
                _lipRegionCount = _lipUpperIdx.Length + _lipLowerIdx.Length;

                int jp = Array.IndexOf(poseNames, "jawOpen");
                int js = (jp >= 0 && jp < b.skin.Length) ? b.skin[jp] : -1;
                float jawMaxDisp = 0f;
                if (js >= 0 && js < mesh.blendShapeCount && mesh.GetBlendShapeFrameCount(js) > 0)
                {
                    mesh.GetBlendShapeFrameVertices(js, 0, dv, dn, dt);
                    for (int v = 0; v < vc; v++)
                    {
                        float l = dv[v].magnitude;
                        if (l > jawMaxDisp) jawMaxDisp = l;
                    }
                }

                // 2) 每个 pose 的唇缝分组（成因分解用）
                int n = poseNames.Length;
                _lipCoeff = new float[n];
                _lipGroup = new byte[n];
                _iJawOpenCoeff = Array.IndexOf(poseNames, "jawOpen");
                for (int i = 0; i < n; i++)
                {
                    string nm = poseNames[i];
                    if (i == _iJawOpenCoeff) _lipGroup[i] = 1;
                    else if (Array.IndexOf(LipOpenPoses, nm) >= 0) _lipGroup[i] = 2;
                    else if (Array.IndexOf(LipSealPoses, nm) >= 0) _lipGroup[i] = 3;
                }

                // 3) 每个 pose 的系数 = 「这个 pose 打满时的唇缝」−「静息唇缝」，实测得到。
                //    不再做任何线性化 / 投影假设：把形状真的摆上去烘一次，量质心距离。
                //    Unity 的 blendshape 本身是线性叠加，所以 Σ w_i · coeff_i 就是精确解。
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < n; i++)
                {
                    int bi = (i < b.skin.Length) ? b.skin[i] : -1;
                    if (bi < 0 || bi >= mesh.blendShapeCount) continue;
                    if (mesh.GetBlendShapeFrameCount(bi) <= 0) continue;

                    ClearAll();
                    smr.SetBlendShapeWeight(bi, 100f);
                    _lipCoeff[i] = Gap() - g0;
                }
                ClearAll();
                for (int i = 0; i < savedW.Length; i++) smr.SetBlendShapeWeight(i, savedW[i]);
                sw.Stop();
                _lipProbeReady = true;

                // 4) 量纲参照 + 关键通道系数。没有量纲参照的话「0.03」这种数字没法判断大小。
                var sb = new System.Text.StringBuilder();
                sb.AppendLine($"[A2F唇缝] 探针已建：上唇 {_lipUpperIdx.Length} 顶点 / 下唇 {_lipLowerIdx.Length} 顶点" +
                              $"（网格={mesh.name}，姿态数={n}）");
                sb.AppendLine($"[A2F唇缝] 量纲：网格包围盒高={_lipRefHeight:F3}，静息唇缝={_lipRestGap / _lipRefHeight * 100f:F2}%，" +
                              $"jawOpen 形状最大顶点位移={jawMaxDisp:F3}（{n} 个 pose 逐个打满 BakeMesh 实测，{sw.ElapsedMilliseconds}ms）");
                sb.Append("[A2F唇缝] 满权重(1.0)时的唇缝变化（+开 / −闭，% 网格高）：");
                string[] keys = { "jawOpen", "mouthUpperUpLeft", "mouthLowerDownLeft", "mouthClose",
                                  "mouthFunnel", "mouthPucker", "mouthStretchLeft", "mouthSmileLeft",
                                  "mouthPressLeft", "mouthShrugLower", "mouthRollLower", "jawForward" };
                for (int i = 0; i < keys.Length; i++)
                {
                    int p = Array.IndexOf(poseNames, keys[i]);
                    float c = (p >= 0 && p < _lipCoeff.Length) ? _lipCoeff[p] : 0f;
                    sb.Append($" {keys[i]}={c / _lipRefHeight * 100f:F2}%");
                }
                Debug.Log(sb.ToString());

                float jawC = (_iJawOpenCoeff >= 0 && _iJawOpenCoeff < _lipCoeff.Length)
                    ? _lipCoeff[_iJawOpenCoeff] : 0f;
                if (Mathf.Abs(jawC) < _lipRefHeight * 0.002f)
                    Debug.LogWarning("[A2F唇缝] jawOpen 在这张网格上几乎不拉唇缝（系数≈0）—— " +
                                     "它的形变只带下颌/下巴，不带嘴唇。这就是「下巴动了但嘴没张开」。 " +
                                     "只能靠 jawOpenToLipOpen 把量补到 mouthLowerDown / mouthUpperUp 上， " +
                                     "或者给网格补一个真正分开嘴唇的形状。");
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[A2F唇缝] 建探针失败（网格可能未开启 Read/Write）: {e.Message}");
            }
        }

        /// <summary>
        /// 按形变量级从大到小取顶点；形变掉到峰值 20% 以下就停，避免把下巴 / 脸颊混进「唇」。
        /// </summary>
        private static int[] PickStrongestVertices(Vector3[] dv, int vc, int take)
        {
            if (vc <= 0) return null;
            var mag = new float[vc];
            for (int i = 0; i < vc; i++) mag[i] = dv[i].sqrMagnitude;
            var order = new int[vc];
            for (int i = 0; i < vc; i++) order[i] = i;
            System.Array.Sort(order, (a, b) => mag[b].CompareTo(mag[a]));
            if (mag[order[0]] < 1e-12f) return null;

            int n = Mathf.Min(take, vc);
            float floor = mag[order[0]] * 0.04f;      // (20%)^2：位移掉到峰值 20% 以下就不算唇了
            for (int i = n; i < vc; i++)
            {
                if (mag[order[i]] < floor) break;
                n++;
            }
            var res = new int[n];
            for (int i = 0; i < n; i++) res[i] = order[i];
            return res;
        }

        /// <summary>
        /// 启动自检：把网格摆成几个极端姿势，用 <c>BakeMesh</c> 量出上下唇质心的<b>真实</b>间距。
        ///
        /// 这是唯一不依赖任何假设的证据 —— 权重、系数、量纲推导全部绕开，直接看几何。
        /// 只回答一个问题：<b>jawOpen 打满时嘴唇到底分不分开。</b>
        ///   · 净开口 ≈ 0 → 这张网格的 jawOpen 只带下颌整体下移、嘴唇跟着一起走（「下巴动了嘴没张开」
        ///     的字面意思），只能靠 jawOpenToLipOpen 把量转到 mouthUpperUp / mouthLowerDown 上。
        ///   · 净开口 明显 &gt; 0 → 网格没问题，那是权重幅度或抵消的问题。
        /// </summary>
        private void SelfTestMouthOpen()
        {
            if (_bindings.Count == 0 || _pipeline == null) return;
            var b = _bindings[0];
            var smr = b.renderer;
            var mesh = smr != null ? smr.sharedMesh : null;
            var poseNames = _pipeline.SkinSolver != null ? _pipeline.SkinSolver.PoseNames : null;
            if (mesh == null || poseNames == null || _lipUpperIdx == null || _lipLowerIdx == null) return;

            var baked = new Mesh();

            // 上下唇质心的距离：静息时约等于唇厚，张开就变大。用距离而不是 Y 差，轴无关。
            float Gap()
            {
                smr.BakeMesh(baked);
                var vs = baked.vertices;
                if (vs == null || vs.Length <= 1) return 0f;
                var u = Vector3.zero;
                var l = Vector3.zero;
                for (int k = 0; k < _lipUpperIdx.Length; k++)
                    u += vs[Mathf.Min(_lipUpperIdx[k], vs.Length - 1)];
                for (int k = 0; k < _lipLowerIdx.Length; k++)
                    l += vs[Mathf.Min(_lipLowerIdx[k], vs.Length - 1)];
                return (u / _lipUpperIdx.Length - l / _lipLowerIdx.Length).magnitude;
            }
            void SetPose(string name, float w)
            {
                int p = Array.IndexOf(poseNames, name);
                if (p < 0 || p >= b.skin.Length) return;
                int bi = b.skin[p];
                if (bi >= 0 && bi < mesh.blendShapeCount) smr.SetBlendShapeWeight(bi, w * 100f);
            }
            void ClearAll()
            {
                for (int i = 0; i < mesh.blendShapeCount; i++) smr.SetBlendShapeWeight(i, 0f);
            }

            try
            {
                ClearAll();
                float g0 = Gap();

                ClearAll();
                SetPose("jawOpen", 1f);
                float g1 = Gap();

                ClearAll();
                SetPose("jawOpen", 1f);
                SetPose("mouthUpperUpLeft", 1f);
                SetPose("mouthUpperUpRight", 1f);
                SetPose("mouthLowerDownLeft", 1f);
                SetPose("mouthLowerDownRight", 1f);
                float g2 = Gap();

                ClearAll();

                float pct = 100f / _lipRefHeight;
                float jawOnly = (g1 - g0) * pct;
                float lipOnly = (g2 - g1) * pct;     // 4 个唇分开通道从 0 打到 1 的额外开度

                // 线性外推要多少 jawOpenToLipOpen 才能把最大张口顶到目标开度。
                // ShapeMouthOpen 是 w += k * open，开度 ≈ k * open * lipOnly，所以 k ≈ 目标 / lipOnly。
                float need = lipOnly > 0.05f ? (LipOpenTargetPercent - jawOnly) / lipOnly : 0f;

                Debug.Log($"[A2F自检] 上下唇质心间距（%网格高；静息唇厚基准={_lipRestGap * pct:F2}）：" +
                          $"静息={g0 * pct:F2} jawOpen(1.0)={g1 * pct:F2} jawOpen+唇分开={g2 * pct:F2} " +
                          $"| 净开口：仅jawOpen={jawOnly:F2}% 唇分开组(0→1)={lipOnly:F2}%" +
                          (Mathf.Abs(jawOnly) < 0.5f
                              ? "  ← jawOpen 拉不开唇缝：这形状只带下颌，嘴唇跟着整体走"
                              : "  ← jawOpen 本身能拉开唇缝，是权重/幅度的问题"));
                if (need > 0.05f)
                    Debug.Log($"[A2F自检] 想让张口峰值到 {LipOpenTargetPercent:F1}% 网格高，" +
                              $"jawOpenToLipOpen 需要 ≈ {Mathf.Min(need, 3f):F2}" +
                              $"（当前 {(Config != null ? Config.jawOpenToLipOpen : 0f):F2}）" +
                              "—— 直接在 Inspector 里改，改代码默认值对已序列化的字段无效。");
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[A2F自检] 张口自检失败: {e.Message}");
            }
            finally
            {
                if (baked != null) Destroy(baked);
            }
        }

        /// <summary>
        /// 量当前帧的唇缝开合（网格本地单位）。这是判断「嘴到底张没张」的客观数字，
        /// 比看 jawOpen 权重靠谱 —— 权重是 Mark 模板上的分解，落到别的目标网格上效果差很多。
        /// </summary>
        private void MeasureMouthAperture(float[] frame)
        {
            if (!_lipProbeReady || _lipCoeff == null) return;

            float jaw = 0f, lip = 0f, seal = 0f, other = 0f;
            int n = Mathf.Min(_lipCoeff.Length, frame.Length);
            for (int i = 0; i < n; i++)
            {
                float w = Mathf.Clamp(frame[i] * Strength, 0f, 1f);
                float c = w * _lipCoeff[i];
                switch (_lipGroup != null && i < _lipGroup.Length ? _lipGroup[i] : (byte)0)
                {
                    case 1: jaw += c; break;
                    case 2: lip += c; break;
                    case 3: seal += c; break;
                    default: other += c; break;
                }
            }
            float s = jaw + lip + seal + other;

            if (s < _apertureMin) _apertureMin = s;
            if (s > _apertureMax)
            {
                _apertureMax = s;
                _apJaw = jaw; _apLipOpen = lip; _apSeal = seal; _apOther = other;
                _apJawW = (_iJawOpenCoeff >= 0 && _iJawOpenCoeff < frame.Length)
                    ? Mathf.Clamp(frame[_iJawOpenCoeff] * Strength, 0f, 1f) : 0f;
            }

            if (++_apertureFrames < 60) return;
            if (Config != null && Config.debugMode)
            {
                float pct = 100f / _lipRefHeight;
                Debug.Log($"[A2F唇缝] 近 60 帧开合区间 = [{_apertureMin * pct:F2}% .. {_apertureMax * pct:F2}%] 网格高" +
                          $"（静息≈0；说话时 max 应到 2~6%，低于 1% 就是看不出张嘴）");
                Debug.Log($"[A2F唇缝] 最大张口那一帧的成因：jawOpen={_apJaw * pct:F2}% " +
                          $"唇分开(UpperUp/LowerDown)={_apLipOpen * pct:F2}% " +
                          $"闭唇组={_apSeal * pct:F2}% 其它={_apOther * pct:F2}% " +
                          $"(该帧 jawOpen 权重={_apJawW:F2}) " +
                          $"| 下颌={(JawBone != null ? $"下降峰值={_lastJawDrop:F3}" : "未绑定骨骼")}");
            }
            _apertureMin = float.MaxValue;
            _apertureMax = float.MinValue;
            _apertureFrames = 0;
            _apJaw = _apLipOpen = _apSeal = _apOther = 0f;
        }

        /// <summary>
        /// <b>真值校验</b>：把当前姿态烘出来，直接量上下唇质心距离。
        ///
        /// 前面那些数字（jawOpen 权重、pose 系数、量纲换算）全是推算，中间任何一环错了结论就错。
        /// 这条不看权重，只看几何事实 —— 用来终结「日志上权重够大，但屏幕上嘴没张开」这类争论：
        ///   · 实测区间上限 ≈ 静息基准（比如 1.2% → 1.3%）→ 权重根本没落到网格上，去查 Strength / 绑定；
        ///   · 实测上限 明显高于静息（比如 1.2% → 3.5%）→ 网格确实张开了，那是观感 / 牙齿 / 幅度问题。
        /// </summary>
        private void SampleRealAperture(float[] frame)
        {
            if (!_lipProbeReady || _bakeMesh == null) return;
            if (_bindings.Count == 0 || _lipUpperIdx == null || _lipLowerIdx == null) return;
            // 这条是「嘴到底张没张」的<b>真值</b>（BakeMesh 直接量唇缝），是所有权重/系数推算的最终裁判，
            // 所以无论 debugMode 都打（每 30 帧烘一次，4855 顶点约 2 次/秒，开销可接受）。
            // 排查「嘴没张开」时必须看它：上限≈静息基准=权重没落到网格上；上限明显抬高=嘴开了、只是幅度/牙齿问题。
            // 若在意这 2 次/秒的 BakeMesh，确认问题后可在此改回 debugMode 门控。
            if (Config == null) return;

            if (_iJawOpenCoeff >= 0 && _iJawOpenCoeff < frame.Length)
            {
                float w = Mathf.Clamp(frame[_iJawOpenCoeff] * Strength, 0f, 1f);
                if (w > _realJawMax) _realJawMax = w;
            }

            if (++_appliedSinceSample < 30) return;     // 每 30 帧烘一次（0.5s）
            _appliedSinceSample = 0;

            try
            {
                var smr = _bindings[0].renderer;
                smr.BakeMesh(_bakeMesh);
                var vs = _bakeMesh.vertices;
                if (vs == null || vs.Length == 0) return;

                var u = Vector3.zero;
                var l = Vector3.zero;
                for (int k = 0; k < _lipUpperIdx.Length; k++)
                    u += vs[Mathf.Min(_lipUpperIdx[k], vs.Length - 1)];
                for (int k = 0; k < _lipLowerIdx.Length; k++)
                    l += vs[Mathf.Min(_lipLowerIdx[k], vs.Length - 1)];
                float gap = (u / _lipUpperIdx.Length - l / _lipLowerIdx.Length).magnitude;

                if (gap < _realMin) _realMin = gap;
                if (gap > _realMax) _realMax = gap;
                if (++_realSamples < 8) return;         // 8 次采样 ≈ 4 秒汇总一条

                float pct = 100f / _lipRefHeight;
                float rest = _lipRestGap * pct;
                Debug.Log($"[A2F实测] 唇缝真实开度 = [{_realMin * pct:F2}% .. {_realMax * pct:F2}%] 网格高" +
                          $"（静息基准={rest:F2}%，张嘴峰值应是它的 2.5 倍以上）" +
                          $"| 本段写入网格的 jawOpen 峰值={_realJawMax:F2} (Strength={Strength:F2})");

                _realMin = float.MaxValue;
                _realMax = float.MinValue;
                _realSamples = 0;
                _realJawMax = 0f;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[A2F实测] 烘网格失败: {e.Message}");
            }
        }

        /// <summary>
        /// Unity 把「无骨骼的静态网格」导入成 MeshFilter + MeshRenderer，而 blendshape 权重只能写在
        /// SkinnedMeshRenderer 上（MeshRenderer 没有 SetBlendShapeWeight）。就地加一个 bones 为空的
        /// SkinnedMeshRenderer 顶替它：纯 blendshape 网格不做蒙皮变换，渲染结果与原来一致。
        /// </summary>
        private SkinnedMeshRenderer TryUpgradeMeshRenderer()
        {
            var mr = GetComponent<MeshRenderer>();
            if (mr == null) mr = GetComponentInChildren<MeshRenderer>();
            if (mr == null) return null;

            var mf = mr.GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null || mf.sharedMesh.blendShapeCount == 0) return null;

            var smr = mr.gameObject.AddComponent<SkinnedMeshRenderer>();
            smr.sharedMesh = mf.sharedMesh;
            smr.sharedMaterials = mr.sharedMaterials;
            smr.bones = new Transform[0];
            smr.rootBone = null;
            smr.quality = SkinQuality.Bone1;
            smr.updateWhenOffscreen = true;
            smr.localBounds = mf.sharedMesh.bounds;
            mr.enabled = false;

            Debug.Log($"[Audio2Face] 网格没有蒙皮，已自动加 SkinnedMeshRenderer 顶替 MeshRenderer：" +
                      $"blendshape={mf.sharedMesh.blendShapeCount} 顶点={mf.sharedMesh.vertexCount}");
            return smr;
        }

        #region 音频输入

        public void StartMicrophone()
        {
            if (Microphone.devices.Length == 0)
            {
                Debug.LogWarning("[Audio2Face] 没有麦克风设备");
                return;
            }

            var device = string.IsNullOrEmpty(MicrophoneDevice) ? Microphone.devices[0] : MicrophoneDevice;
            int rate = _pipeline != null && _pipeline.Info != null ? _pipeline.Info.SampleRate : 16000;
            _micClip = Microphone.Start(device, true, 1, rate);
            _micReadPos = 0;
            if (_micClip != null) Debug.Log($"[Audio2Face] 麦克风启动: {device} @ {rate}Hz");
        }

        public void StopMicrophone()
        {
            if (_micClip != null && !string.IsNullOrEmpty(MicrophoneDevice))
            {
                Microphone.End(MicrophoneDevice);
            }
            _micClip = null;
            _micReadPos = 0;
        }

        private void PollMicrophone()
        {
            string device = string.IsNullOrEmpty(MicrophoneDevice) ? Microphone.devices[0] : MicrophoneDevice;
            int pos = Microphone.GetPosition(device);
            if (pos == _micReadPos || _micClip == null) return;

            int total = _micClip.samples;
            int available = pos > _micReadPos ? pos - _micReadPos : (total - _micReadPos) + pos;

            int remaining = available;
            while (remaining > 0)
            {
                int chunk = Mathf.Min(remaining, _micBuffer.Length);
                _micClip.GetData(_micBuffer, _micReadPos);
                _pipeline.PushAudio(_micBuffer, 0, chunk);
                _micReadPos = (_micReadPos + chunk) % total;
                remaining -= chunk;
            }
        }

        /// <summary>外部推送 16kHz 单声道 PCM。</summary>
        public void PushAudio(float[] data, int count)
        {
            if (_pipeline == null) return;
            _pipeline.PushAudio(data, 0, count);
        }

        /// <summary>离线处理整段 AudioClip（自动重采样到 16kHz，边推边等推理跟上）。</summary>
        public void ProcessAudioClip(AudioClip clip)
        {
            if (_pipeline == null) return;
            StartCoroutine(ProcessClipRoutine(clip));
        }

        private IEnumerator ProcessClipRoutine(AudioClip clip)
        {
            var raw = new float[clip.samples * clip.channels];
            clip.GetData(raw, 0);

            var mono = new float[clip.samples];
            for (int i = 0; i < clip.samples; i++) mono[i] = raw[i * clip.channels];

            int target = _pipeline.Info.SampleRate;
            var pcm = clip.frequency == target ? mono : Resample(mono, clip.frequency, target);

            // 新的一段音频：把音频游标 / 环形缓冲 / GRU 全部归零，保证窗口序号从 0 开始
            // （窗口 0、1 是 padding 预热窗口，不输出帧）。不重置的话 CalibrateRestPose 期间
            // Update→Tick 已经把消费游标推走，后续 Tick 的触发条件永远满足不了，会直接卡死。
            _pipeline.Reset();
            _displayedFrames = 0;
            _lastSyncLogFrame = 0;
            _frameAccumulator = 0f;
            _holdFrames = true;          // 预热 + 预缓冲期间不许 Update 偷跑帧，开播时才解除
            _audioPlaying = false;

            int cursor = 0;
            int pushCount = 0;
            int warmupSamples = _pipeline.Info.WarmupInferences * _pipeline.Info.StrideSamples;

            // 阶段 1：推预热音频（不播放），让模型跨过 padding_left 的静音区
            // SDK 的 WarmupInferences = ceil(paddingLeft / StrideSamples) = ceil(16000/8000) = 2
            // 这 2 次推理不输出帧，但必须先把音频推够，否则正式推理的窗口里混着旧数据
            // 关键：预热期间不播放音频，否则音频会比动画快 1 秒
            if (Config.debugMode)
                Debug.Log($"[A2F推流] 阶段1: 预热 {warmupSamples} 样本 ({_pipeline.Info.WarmupInferences} 次推理)");

            while (cursor < warmupSamples && cursor < pcm.Length)
            {
                while (_pipeline.UnconsumedAudioSamples >= _pipeline.RingCapacity)
                    yield return null;

                int chunk = Mathf.Min(4096, warmupSamples - cursor);
                _pipeline.PushAudio(pcm, cursor, chunk);
                cursor += chunk;
                // 不在这里手动 Tick——Update 每帧都会调 Tick，让推理自然串行推进，
                // 避免 _runRequested/_busy 阻塞导致只触发一次预热推理。
                yield return null;
            }

            // 阶段 2：等窗口右端游标推进到 warmupSamples（预热推理全部完成），再开始播放音频。
            // 关键：必须等「消费游标」而不是「推理次数」——CalibrateRestPose 已经跑过几次静音
            // 推理（会把 InferenceCount 推到 4），用它判断预热会立即通过，导致预热窗口还没
            // 推完就开播，前面的口型仍基于 padding 窗口，且音频比动画快。
            // ⚠ 这里不能无限等：如果推理一直不消费（模型没跑起来 / worker 卡住），coroutine 会永远卡在
            // 这一行，_holdFrames 一直是 true、_audioPlaying 一直是 false，Update 两条消费路径全被挡住，
            // 表现就是「播放并驱动」嘴一动不动。加个超时兜底，哪怕预热没跑满也往下走并报警。
            float warmupDeadline = Time.realtimeSinceStartup + 8f;
            while (_pipeline.ConsumedSamples < warmupSamples && Time.realtimeSinceStartup < warmupDeadline)
                yield return null;
            if (_pipeline.ConsumedSamples < warmupSamples)
                Debug.LogWarning($"[A2F推流] 预热等待超时（消费 {_pipeline.ConsumedSamples}/{warmupSamples}），强行继续。" +
                                 "若嘴不动，多半是推理没跑起来——看上面有没有 [A2F窗口]/[Audio2Face] 推理日志。");

            if (Config.debugMode)
                Debug.Log($"[A2F推流] 预热完成 (消费游标={_pipeline.ConsumedSamples})，开始预缓冲动画帧");

            // 阶段 2.5：预缓冲。开播那一刻帧队列是空的，推理要 ~350ms 才吐第一批 30 帧，
            // 期间脸是僵的，然后 Update 一次补 6~20 帧 —— 观感就是「口型比声音慢半拍」。
            // 先多推一点音频、等到攒够 prebufferFrames 帧再开播，开头就能对上。
            // ⚠ 第一版写错了：循环条件只判 PendingFrames，推理一次要 350~600ms，
            // 于是每帧 Update 又推 4096 样本，等到攒够帧时音频已经推到 6.4s（102016/128820）。
            // 音频本身没问题（还是从 0 播），但帧被 Update 的累加器分支白消费掉，
            // 开播那一刻队列头的帧已经对应 2.7s 之后的音频 → 动画整体超前。
            // 正确做法：先按「要 N 帧需要多少音频」算一个推送上限，推够就只等推理，不再推。
            int prebuffer = Mathf.Max(0, Config.prebufferFrames);
            if (prebuffer > 0)
            {
                float samplesPerFrame = _pipeline.Info.FramesPerInference > 0
                    ? (float)_pipeline.Info.BufferLength / _pipeline.Info.FramesPerInference
                    : 266.67f;
                int pushTarget = Mathf.Min(pcm.Length,
                    warmupSamples + (int)Mathf.Ceil(prebuffer * samplesPerFrame));

                float deadline = Time.realtimeSinceStartup + 15f;
                int idleWaits = 0;
                while (cursor < pcm.Length && Time.realtimeSinceStartup < deadline)
                {
                    bool needAudio = cursor < pushTarget;
                    bool needFrames = _pipeline.PendingFrames < prebuffer;
                    if (!needAudio && !needFrames) break;

                    if (needAudio)
                    {
                        while (_pipeline.UnconsumedAudioSamples >= _pipeline.RingCapacity)
                            yield return null;

                        int chunk = Mathf.Min(4096, pcm.Length - cursor);
                        _pipeline.PushAudio(pcm, cursor, chunk);
                        cursor += chunk;
                        pushCount++;
                        idleWaits = 0;
                    }
                    else
                    {
                        // 音频已推够，只剩等推理收尾。单次推理最慢 ~650ms，等 3 秒还没新帧
                        // 说明出不来（比 IsBusy 可靠 —— _runRequested 置位和 _busy 置位之间
                        // 有窗口期，直接看 IsBusy 会误判成"没在跑"而提前放弃）。
                        if (++idleWaits > 180) break;
                    }
                    _pipeline.Tick();
                    yield return null;
                }

                if (Config.debugMode)
                    Debug.Log($"[A2F推流] 预缓冲完成：队列 {_pipeline.PendingFrames}/{prebuffer} 帧，" +
                              $"音频已推到 {cursor}/{pcm.Length}（上限 {pushTarget}）");
            }

            _holdFrames = false;         // 解除挂起：后面不管播不播音频，都允许正常消费帧

            // 现在开始播放音频，与动画对齐。
            // 跳过 target offset 对应的音频：模型第一个正式帧对应音频的 targetOffset 位置
            // （bufferLength * numFramesLeftTruncate / nbFramesPerInference = 4000 样本 = 0.25 秒）。
            // 之前错误地跳过 warmupSamples（1 秒），导致音频比动画滞后 0.75 秒。
            // 音频从 0 完整播放。动画侧通过 targetOffset 对齐（第一个正式帧对应音频
            // targetOffset=4000 样本 = 0.25s），而不是截断音频开头 —— 截断会丢掉开头 0.25s 语音。
            if (PlayAudioWhileProcessing)
            {
                _audioSource = GetComponent<AudioSource>();
                if (_audioSource == null) _audioSource = gameObject.AddComponent<AudioSource>();
                _audioSource.clip = clip;
                _audioSource.time = 0f;
                _audioSource.Play();

                _targetOffsetSec = (float)_pipeline.Info.TargetOffsetSamples / _pipeline.Info.SampleRate;
                _displayedFrames = 0;
                _lastSyncLogFrame = 0;
                _audioPlaying = true;

                if (Config.debugMode)
                    Debug.Log($"[A2F推流] 播放开始，动画对齐偏移={_targetOffsetSec:F3}s ({_pipeline.Info.TargetOffsetSamples} 样本)");
            }

            // 阶段 3：继续推剩余音频，边推边播放
            while (cursor < pcm.Length)
            {
                // 背压：等推理消费掉环形缓冲里的数据
                // ring 容量 = BufferLength = 16000，每次推理消费 StrideSamples = 8000
                // 保留至少 1 次推理的空间（8000），避免覆盖未消费数据
                while (_pipeline.UnconsumedAudioSamples >= _pipeline.RingCapacity)
                    yield return null;

                int chunk = Mathf.Min(4096, pcm.Length - cursor);
                _pipeline.PushAudio(pcm, cursor, chunk);
                _pipeline.Tick();
                cursor += chunk;
                if (pushCount < 12)
                {
                    float s = 0f; for (int i = 0; i < chunk; i++) s += pcm[cursor - chunk + i] * pcm[cursor - chunk + i];
                    float rms = Mathf.Sqrt(s / chunk);
                    Debug.Log($"[A2F推流] push#{pushCount} cursor={cursor}/{pcm.Length} chunkRMS={rms:F4} 未消费={_pipeline.UnconsumedAudioSamples}/{_pipeline.RingCapacity}");
                }
                pushCount++;
                yield return null;
            }
            Debug.Log($"[A2F推流] 音频推流结束，共 push {pushCount} 次，总样本={pcm.Length}");
        }

        private static float[] Resample(float[] data, int srcRate, int dstRate)
        {
            if (srcRate == dstRate) return data;
            double ratio = (double)dstRate / srcRate;
            int newLen = (int)(data.Length * ratio);
            var result = new float[newLen];
            for (int i = 0; i < newLen; i++)
            {
                int src = (int)(i / ratio);
                if (src >= data.Length) src = data.Length - 1;
                result[i] = data[src];
            }
            return result;
        }

        #endregion

        /// <summary>从 Audio2Emotion 拿 10 维情绪喂给模型。</summary>
        public void SetEmotion(float[] emotions)
        {
            _pipeline?.SetEmotion(emotions);
        }

        public void ResetPipeline() => _pipeline?.Reset();
    }
}
