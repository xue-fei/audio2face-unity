using UnityEngine;

namespace Audio2Face
{
    public enum Audio2FaceExecutionProvider
    {
        CPU = 0,
        CUDA = 1,
        TensorRT = 2
    }

    [System.Serializable]
    public class Audio2FaceDiffusionConfig
    {
        /// <summary>
        /// 配置版本号。<b>改这个类里任何默认值都必须同时把它 +1，并在 Migrate() 里写迁移逻辑。</b>
        ///
        /// 为什么需要：Unity 的序列化规则是「字段第一次出现时，把当时的默认值写进场景 / 预制体，
        /// 之后就一直用序列化值」。所以后来在代码里改默认值 <b>完全不会生效</b> —— 场景里存的还是
        /// 旧值，Inspector 显示的也是旧值。实测踩过：lowerFaceStrength 从 0.8 改成 1.0、jawOpenToLipOpen
        /// 从 0.35 改成 0.5，日志里打出来的仍然是 0.80 / 0.35（jawOpen 峰值被 0.8 卡在 0.40）。
        /// 新增字段反而没问题（之前没序列化过，用代码默认值）。
        /// </summary>
        [HideInInspector]
        public int configVersion = 0;
        public const int LatestConfigVersion = 3;

        /// <summary>
        /// 把旧版本残留的序列化值升到当前版本。返回 true 表示改动过（调用方应打一条日志）。
        /// </summary>
        public bool Migrate()
        {
            if (configVersion >= LatestConfigVersion) return false;

            if (configVersion < 1)
            {
                // v0 → v1：0.8 是为了压 mouthClose 饱和临时压的，现在饱和由「口型开合」段接管。
                // 留着 0.8 会把 jawOpen 一起砍掉，表现就是「嘴巴张不开」。
                if (lowerFaceStrength > 0f && lowerFaceStrength < 1f) lowerFaceStrength = 1.0f;
                if (upperFaceStrength <= 0f) upperFaceStrength = 1.0f;
                if (skinStrength <= 0f) skinStrength = 1.0f;
                if (jawOpenToLipOpen <= 0f) jawOpenToLipOpen = 0.5f;
                if (jawOpenGain <= 0f) jawOpenGain = 1.5f;
                if (lipSealScale <= 0f) lipSealScale = 0.5f;
                if (!enableMouthOpenAssist) enableMouthOpenAssist = true;
            }

            if (configVersion < 2)
            {
                // v1 → v2：v1 的写法是「只在旧值非法时才重置」，结果场景里存的 0.35（一个完全
                // 合法的值）被原样留了下来 —— 日志里一直是 +lip=0.35，改代码默认值毫无作用。
                // 只对 v1 遗留场景整体覆盖一次「口型开合」这一组，之后用户在 Inspector 里调的
                // 值不会再被覆盖（configVersion 已经是 2 了）。
                jawOpenToLipOpen = 0.5f;
                // 1.5 → 2.0：自检实测 jawOpen 打满(1.0)能拉开 3.14% 网格高（正常），
                // 但驱动时峰值权重只有 0.72（jawOpenGain=1.5），实际只开到 ~2.3% —— 看不出来。
                // 提到 2.0 让峰值顶到 ~0.95，接近手动扫掠 jawOpen=1.0 的效果（上限由 clamp 1.0 兜住）。
                jawOpenGain = 2.0f;
                lipSealScale = 0.5f;
                lipSealSuppressByJawOpen = 1.0f;
                enableMouthOpenAssist = true;
            }

            configVersion = LatestConfigVersion;
            return true;
        }

        [Header("模型目录")]
        [Tooltip("StreamingAssets 下的相对路径，里面要有 network.onnx / network_info.json / bs_skin_* / bs_tongue_*")]
        public string modelFolder = "nvidiaAudio2Face-3D-v3.0";

        [Header("身份")]
        [Tooltip("Claire / James / Mark，对应 identity one-hot 的哪一档")]
        public string identity = "Mark";

        [Header("推理后端")]
        public Audio2FaceExecutionProvider executionProvider = Audio2FaceExecutionProvider.CUDA;
        public int deviceId = 0;

        [Header("扩散噪声")]
        [Tooltip("true = 只在初始化时生成一次噪声并复用（输出确定、省 64MB/次的生成开销，等同 SDK 的 constantNoise）")]
        public bool constantNoise = true;
        public int noiseSeed = 1234;

        [Header("Skin Animator（SDK 后处理）")]
        [Tooltip("模型输出 prediction 先过一遍 SDK 的 AnimatorSkin，再喂求解器。\n" +
                 "它做三件事：叠 eyeClose/lipOpen 静态 pose、上下脸按强度分区、时域平滑。\n" +
                 "参数按身份读 model_config_{identity}.json 并叠加官方 stylization face_params\n" +
                 "（Audio2Face-3D Samples 的 *_diffusion_stylization_config.yaml）：\n" +
                 "Mark lower_face_strength=1.4 / input_strength=1.3 / skin_strength=1.1。\n" +
                 "开启动画后求解器的 TargetIsDelta 会自动改成 false（animator 已把 neutral 加回去）。\n" +
                 "关掉 = 老行为，直接把 prediction 当 delta 喂求解器。")]
        public bool useSkinAnimator = true;
        public bool useTongueAnimator = true;

        [Tooltip("jaw 段（15 维 = 5 个下颌标记点位移）→ 4x4 刚体变换，驱动下颌骨。\n" +
                 "牙齿挂在骨骼上，不走 blendshape —— 关掉它嘴能张开但牙齿不会动。")]
        public bool useJawAnimator = true;
        [Tooltip("eyes 段（4 维）→ 左右眼欧拉角 + saccade 微跳视。关掉眼睛会死盯前方。")]
        public bool useEyesAnimator = true;

        [Tooltip("默认 false：animator 参数一律按身份读 model_config + 官方 stylization（与 SDK 服务同源）。\n" +
                 "勾上才用下面 skinAnimator 里手填的值覆盖。\n" +
                 "注意：手填时记得 model_config 基线是 lower_face_strength=1.0，官方运行时是 1.4。")]
        public bool overrideSkinAnimator = false;
        public Audio2FaceSkinAnimator.Params skinAnimator = null;

        [Tooltip("默认 false：音频输入增益按官方 stylization 走（Mark=1.3 / Claire,James=1.0）。\n" +
                 "官方值是为 MetaHuman 量级的网格调的。目标网格不是 MetaHuman（比如 Sloth_Head2）时，\n" +
                 "1.3 × 1.4(lower_face) × 1.1(skin) 叠起来约 2 倍，jawOpen 会顶到 0.7+ 显得夸张。\n" +
                 "勾上并填 1.0 可以把这一层拿掉，是最直接的「张嘴幅度」旋钮。")]
        public bool overrideInputStrength = false;
        [Tooltip("喂给网络的音频增益（SDK audio_accumulator 里对窗口整体相乘）")]
        public float inputStrength = 1.0f;

        [Header("眨眼")]
        [Tooltip("SDK 的 AnimatorSkin 不做眨眼 —— blinkOffset 是应用层每帧喂进去的参数\n" +
                 "（animator.h: SetBlinkOffset，默认 0，取值 0~1）。不喂就永远不眨眼，\n" +
                 "只剩 eyelidOpenOffset 的静态睁眼偏置。这里用一个随机调度器按帧的音频时间戳驱动它。")]
        public bool enableBlink = true;
        [Tooltip("两次眨眼间隔范围（秒）。成人平均约 4s 一次")]
        public float blinkIntervalMin = 2.5f;
        public float blinkIntervalMax = 6.0f;
        [Tooltip("单次眨眼时长（秒）")]
        public float blinkDuration = 0.16f;
        [Tooltip("连眨两下的概率")]
        public float blinkDoubleChance = 0.15f;
        public int blinkSeed = 20260720;

        [Header("幅度（非 MetaHuman 网格必看）")]
        [Tooltip("勾上后用下面的值覆盖官方 stylization 的上下半脸强度。\n" +
                 "核对过三处权威来源，官方值就是 lower=1.0 / upper=1.0 / skin=1.0：\n" +
                 "  · model_config_Mark.json 的 lower_face_strength = 1.0\n" +
                 "  · Maya-ACE 官方场景 mark_v3_fullface.ma 里 A2FAnimationPlayer 节点 .lfst = 1.0\n" +
                 "  · 同节点 .skst = 1.0\n" +
                 "（animator.h 里那段 REFL 默认 1.3/1.1 是「读不到配置」时的兜底，不是官方值。）\n" +
                 "别整体调小：那会把 jawOpen 一起砍掉 —— 就是「嘴巴张不开」。\n" +
                 "要放大张嘴请用下面「口型开合」段的 jawOpenGain，它只放大 jawOpen。")]
        public bool overrideFaceStrength = true;
        [Tooltip("官方 Mark 值 = 1.0（model_config_Mark.json 的 lower_face_strength，Maya 官方场景\n" +
                 "A2FAnimationPlayer 节点的 .lfst 也是 1.0）。之前压到 0.8 是为了压 mouthClose 饱和，\n" +
                 "现在饱和由下面「口型开合」段接管，所以回到官方值，别再整体调小 —— 整体调小会连\n" +
                 "jawOpen 一起砍掉，就是「嘴巴张不开」。")]
        public float lowerFaceStrength = 1.0f;
        public float upperFaceStrength = 1.0f;
        public float skinStrength = 1.0f;

        [Header("口型开合（下巴动了但嘴不张 → 调这一段）")]
        [Tooltip("开合整形：压低「闭唇组」权重、并把 jawOpen 的一部分补给「唇分开」通道。\n" +
                 "为什么需要：Mark 模板里 mouthShrugLower 与 jawOpen 的顶点位移余弦 −0.92、\n" +
                 "mouthPressLeft −0.61、mouthRollLower −0.57（几乎反向共线）。BVLS 拟合张嘴几何时\n" +
                 "会把能量拆成 jawOpen 正向 + 这几个闭唇通道反向，几何残差很小，但语义互相抵消 ——\n" +
                 "表现就是「下巴掉下去了，嘴唇却还合着，看不到牙和舌头」。\n" +
                 "关掉 = 直接用求解器原始权重。")]
        public bool enableMouthOpenAssist = true;
        [Tooltip("闭唇组（mouthClose / mouthPressL·R / mouthShrugLower / mouthRollLower）整体缩放。\n" +
                 "1 = 不动，0.5 = 闭唇作用减半。设 0 会连 /m/ /b/ 这类闭口音一起丢掉，不建议。")]
        public float lipSealScale = 0.5f;
        [Tooltip("按 jawOpen 动态压制闭唇组的强度：w *= (1 - k × jawOpen)。\n" +
                 "1 = 嘴张到最大时闭唇通道基本清零；0 = 只做上面的固定缩放。\n" +
                 "按张嘴量压制而不是一刀切，闭口音（jawOpen≈0）不受影响。")]
        public float lipSealSuppressByJawOpen = 1.0f;
        [Tooltip("把 jawOpen 的一部分补给唇分开通道（mouthLowerDownL·R / mouthUpperUpL·R）。\n" +
                 "jawOpen 单独驱动时只带下颌，唇缝未必张开；补一点才能真正露牙。\n" +
                 "0 = 不补，0.5~0.7 是常用区间，超过 0.8 容易变成「咧嘴」。")]
        public float jawOpenToLipOpen = 0.5f;
        [Tooltip("求解后单独放大 jawOpen（乘完再 clamp 到 1）。\n" +
                 "为什么可以安全地单独放大它：把 jawOpen 的顶点位移用全部 52 个 pose 做线性分解，\n" +
                 "它自身系数是 1.000、其余全部 ~0.000 —— jawOpen 与其它 pose 几乎正交，\n" +
                 "放大它不会像放大 lowerFaceStrength 那样把闭唇通道一起放大、重新饱和。\n" +
                 "实测 lower=1.0 时 jawOpen 峰值 0.40~0.50，×1.5 → 0.60~0.75。")]
        public float jawOpenGain = 1.5f;

        [Header("舌头（Maya 官方场景的 bstm / bsto）")]
        [Tooltip("Maya-ACE 的 sample_project/scenes/mark_v3_fullface.ma 里 A2FAnimationPlayer 节点带两组\n" +
                 "16 路参数：.bstm（乘法器）= 2 1 1 1 3 1 1 1 2 1 1 1 1 2 1 1，.bsto（偏移）第 9 路 = 0.2，\n" +
                 "正好对上 16 个 tongue pose。官方这是在放大「舌头抬起 / 前伸」那几路（TipUp×2、RollUp×3、\n" +
                 "Up×2、Stretch×2、Down+0.2），让舌头更容易露出来。我们的 tongue 求解器没有这一段，\n" +
                 "勾上后按 pose 名套用（不依赖下标顺序）。\n" +
                 "注意：Sloth_Head2 上舌头通道命中 0/16，勾了也看不到变化 —— 先在网格上补形状。")]
        public bool applyOfficialTongueGains = true;

        [Header("播放")]
        [Tooltip("开始播放音频前先算好多少帧动画（60 帧 = 1 秒）。\n" +
                 "推理约 350ms/30 帧，比实时快 1.4 倍，但开播那一刻队列是空的：不预缓冲的话\n" +
                 "前 0.3~0.5 秒脸是僵的，然后一次性补 6~20 帧，看起来就是「口型比声音慢半拍」。\n" +
                 "设 0 = 不预缓冲（麦克风实时场景用 0）。")]
        public int prebufferFrames = 60;

        [Tooltip("播放期间「已推入 pipeline 的音频」最多领先播放头多少秒。\n" +
                 "原来唯一的背压是环形缓冲容量：每帧推 4096 样本 = 0.256s 音频 = 15 倍实时，\n" +
                 "于是整段音频被一次性提前算完，帧队列 30→60→…→385 一路涨（22 秒音频最终\n" +
                 "在内存里堆 1300+ 帧）。离线播一段 clip 只是「提前算完」，实时麦克风输入则会\n" +
                 "持续累积延迟。按播放时钟节流后生产速率 = 消费速率（都是 60 帧/秒），队列稳定。\n" +
                 "实际领先量还会取「开播那一刻已有的领先量」的下限，避免刚开播就干等。")]
        public float maxPushAheadSec = 1.0f;

        [Header("求解器")]
        [Tooltip("使用 npz 里的 frontalMask 只解正面顶点，速度约快 5 倍")]
        public bool useFrontalMask = true;
        [Tooltip("BVLS 收敛容差，SDK 默认 1e-10")]
        public float solverTolerance = 1e-10f;
        [Tooltip("模型输出的 geometry 是相对 neutral 的位移（delta）。\n" +
                 "关掉 useSkinAnimator 时必须为 true：animator 没跑，prediction 本身就是 delta，\n" +
                 "再减一次 neutral 会让权重全部顶到上界 1.0。\n" +
                 "开启 useSkinAnimator 时本值会被自动改成 false。")]
        public bool predictionIsDelta = true;

        [Header("驱动")]
        [Tooltip("blendshape 权重 0..1 映射到 0..100 后写入 SkinnedMeshRenderer")]
        public bool autoApplyToSkinnedMesh = true;
        [Tooltip("找不到同名 blendshape 时是否逐下标回退（有错位风险）")]
        public bool fallbackToIndexMapping = false;

        [Header("调试")]
        public bool debugMode = false;

        public string ModelFolderFullPath => System.IO.Path.Combine(Application.streamingAssetsPath, modelFolder);
    }
}
