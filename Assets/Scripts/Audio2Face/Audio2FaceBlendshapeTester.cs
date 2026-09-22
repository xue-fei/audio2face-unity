using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// 3D 模型 blendshape 效果的「可视化测试台」。
    ///
    /// 它回答两个彼此独立的问题，别混在一起测：
    ///   1. <b>网格本身的 blendshape 能不能动</b>（rig 的问题）→ 用手动滑条。
    ///      某根滑条拉满、脸相应部位不动 → 是资产/blendshape 命名/朝向的问题，跟 Audio2Face 无关。
    ///   2. <b>Audio2Face 输出的权重准不准</b>（管线的问题）→ 用「实时驱动」接 Driver，
    ///      放音频看嘴动不动、幅度够不够。前提是第 1 步已经通过。
    ///
    /// 用法：把本组件挂到场景里那个模型对象上（或它的父节点），运行即可。
    /// </summary>
    [DisallowMultipleComponent]
    public class Audio2FaceBlendshapeTester : MonoBehaviour
    {
        [Header("目标网格（留空自动在本对象/子对象里找）")]
        [Tooltip("必须是 SkinnedMeshRenderer，blendshape 权重只能写在它上面。")]
        public SkinnedMeshRenderer Target;

        [Tooltip("Unity 把「无骨骼的静态网格」导入成 MeshRenderer，这种网格没法写 blendshape 权重。\n" +
                 "勾选后会自动加一个不挂骨骼的 SkinnedMeshRenderer 顶替它——纯 blendshape 网格可以这么用。")]
        public bool AutoUpgradeRenderer = true;

        [Header("面板")]
        public bool ShowPanel = true;
        public float PanelScale = 1f;
        public Vector2 Scroll;

        [Header("实时驱动（可选：验证 Audio2Face 权重）")]
        [Tooltip("留空则本组件只做手动测试，不依赖模型。")]
        public Audio2FaceDiffusionComponent Driver;

        [Tooltip("拖一个 16bit PCM 的 wav 进来，点「播放并驱动」可以边听边看口型。")]
        public AudioClip Clip;
        public bool PlayThroughSpeakers = true;

        [Header("口型增益（默认 1：不要再用它放大，交给 stylization）")]
        [Tooltip("⚠ 这不是「口型不明显」的解法。真正的幅度增益来自官方 stylization 层，\n" +
                 "已经在 Audio2FaceAnimatorConfig 里按身份正确叠加（Mark lower_face_strength=1.4、\n" +
                 "input_strength=1.3）。之前用它 ×4 放大，会把本应 10%~30% 的微妙口型（微笑/嘴角/脸颊）\n" +
                 "无差别顶到 100%，左右脸微小不对称被放大成「嘴歪眼斜」。\n" +
                 "正确做法：保持 1。若真需要微调，只在 0.8~1.5 之间小幅调。")]
        public float DriveGain = 1f;

        // ---- 运行时 ----
        private string[] _poses;
        private int[] _map;
        private float[] _w;
        private int _matched;
        private string _report;
        private bool _ready;
        private bool _liveDriving;
        private float _rawPeak;
        // 心跳：记录「实时驱动到底有没有写进网格」。手动扫掠走 ApplyManual、实时驱动走 OnDriverFrame，
        // 两者都写 Target —— 前者嘴动能证明网格没问题，所以「实时不动」只可能是 OnDriverFrame 没跑到
        // 或 frame 权重≈0。下面的 [A2F驱动] 心跳就是为了把这两种情况分开（配合 [A2F实测] 看嘴真开没开）。
        private int _driveFrames;
        private float _driveJawPeak;

        private AudioSource _audio;
        private float _sweepT;
        private int _sweepPose = -1;
        private string _status = "";
        private readonly List<KeyValuePair<string, float>> _live = new List<KeyValuePair<string, float>>();

        void Awake()
        {
            _audio = GetComponent<AudioSource>();
            if (_audio == null) _audio = gameObject.AddComponent<AudioSource>();
            _audio.playOnAwake = false;

            Target = ResolveTarget();
            if (Target == null)
            {
                Debug.LogError("[A2F Test] 找不到 SkinnedMeshRenderer，也找不到可升级的 MeshRenderer。" +
                               "请把本组件挂到模型对象上。");
                return;
            }

            if (Driver == null) Driver = GetComponentInParent<Audio2FaceDiffusionComponent>();
            if (Driver != null) Driver.OnFrameApplied.AddListener(OnDriverFrame);

            EnsureMap();
            _ready = true;
        }

        private void OnDestroy()
        {
            if (Driver != null) Driver.OnFrameApplied.RemoveListener(OnDriverFrame);
        }

        /// <summary>
        /// 拿网格：优先 SkinnedMeshRenderer；只有 MeshRenderer 时自动升级。
        /// 无骨骼的静态网格在 Unity 里是 MeshFilter+MeshRenderer，而 blendshape 权重只能写在
        /// SkinnedMeshRenderer 上（MeshRenderer 没有 SetBlendShapeWeight）。加一个 bones 为空的
        /// SkinnedMeshRenderer 就能同时拿到 blendshape 和原本的渲染结果。
        /// </summary>
        private SkinnedMeshRenderer ResolveTarget()
        {
            var smr = GetComponent<SkinnedMeshRenderer>();
            if (smr == null) smr = GetComponentInChildren<SkinnedMeshRenderer>(true);
            if (smr != null) return smr;

            if (!AutoUpgradeRenderer) return null;

            var mr = GetComponent<MeshRenderer>();
            if (mr == null) mr = GetComponentInChildren<MeshRenderer>(true);
            if (mr == null) return null;

            var mf = mr.GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null) return null;

            var mesh = mf.sharedMesh;
            if (mesh.blendShapeCount == 0)
            {
                Debug.LogWarning($"[A2F Test] 网格 {mesh.name} 没有任何 blendshape，无法测试。");
                return null;
            }

            smr = mr.gameObject.AddComponent<SkinnedMeshRenderer>();
            smr.sharedMesh = mesh;
            smr.sharedMaterials = mr.sharedMaterials;
            smr.bones = new Transform[0];       // 无骨骼：顶点不做蒙皮变换，退化为普通网格
            smr.rootBone = null;
            smr.quality = SkinQuality.Bone1;
            smr.updateWhenOffscreen = true;
            smr.localBounds = mesh.bounds;
            mr.enabled = false;                 // 关掉原 MeshRenderer，避免双重绘制

            Debug.Log($"[A2F Test] 已把 MeshRenderer 升级为 SkinnedMeshRenderer：" +
                      $"{mesh.name}，blendshape={mesh.blendShapeCount}，顶点={mesh.vertexCount}");
            return smr;
        }

        /// <summary>
        /// frame[] 的下标语义 = 写 frame 的那个 pose 名数组。Driver 写 frame 用
        /// SkinSolver.PoseNames（NVIDIA 自定义序），所以实时驱动的 _poses 必须是同一个数组。
        /// ArkItBlendshapeNames.All52 是字母序，只配「没有 Driver」的纯手动 rig 测试兜底。
        /// </summary>
        private string[] ResolvePoseNames()
        {
            var pn = Driver != null ? Driver.PoseNames : null;
            return (pn != null && pn.Length > 0) ? pn : ArkItBlendshapeNames.All52;
        }

        /// <summary>
        /// 保证 _poses/_map 与 frame 下标同序；pose 源一变就重建。
        ///
        /// ⚠ 别退回「只在 Awake 建一次」：Driver 的 pipeline 到 Start 才 new 出来，
        ///   而 Unity 所有 Awake 都先于所有 Start —— Awake 里 Driver.PoseNames 恒为 null，
        ///   _poses 会退化成 All52（字母序），frame 却是 SkinSolver.PoseNames（自定义序）。
        ///   两个序不一致时，OnDriverFrame 把 frame[i] 写给 All52[i] 对应的 blendshape，
        ///   每个权重都落错形状：嘴张不开、上下唇一起被拽下去。
        ///   手动扫掠不受影响（_w/_map 在同一套序里自洽），所以
        ///   「扫掠正常、播放并驱动乱」就是这个 bug 的指纹。
        /// </summary>
        private bool EnsureMap()
        {
            if (Target == null) return false;
            var want = ResolvePoseNames();
            if (_map != null && ReferenceEquals(_poses, want)) return true;
            BuildMap(want);
            return _map != null;
        }

        private void BuildMap(string[] poseNames)
        {
            _poses = poseNames;

            _map = ArkItBlendshapeNames.MapPoses(Target.sharedMesh, _poses, false, out _matched, out _report);
            _w = new float[_poses.Length];

            bool fallback = ReferenceEquals(_poses, ArkItBlendshapeNames.All52);
            if (fallback && Driver != null)
                Debug.LogWarning("[A2F Test] ⚠ Driver.PoseNames 还没就绪（pipeline 未创建），暂用 ARKit 52 字母序建映射。" +
                                 "实时驱动前会自动换成 Driver 的 pose 序重建——两套序不同，不重建就会把权重写串。");
            Debug.Log($"[A2F Test] 目标网格 = {Target.sharedMesh.name}" +
                      $"，顶点={Target.sharedMesh.vertexCount}" +
                      $"，blendshape={Target.sharedMesh.blendShapeCount}" +
                      $" | pose 序={(fallback ? "ARKit 52 字母序(兜底)" : "Driver.PoseNames(与 frame 同序)")}" +
                      $"×{_poses.Length}");
            Debug.Log(_report);

            if (_matched == 0)
                Debug.LogError("[A2F Test] 0 个 pose 能对应上网格 blendshape，口型一定不会动。" +
                               "请按上面的报告检查命名（前缀 / _L _R / Left Right）。");
        }

        void Update()
        {
            if (!_ready) return;
            EnsureMap();    // pose 源就绪后自动重建（见 EnsureMap 注释）

            if (_sweepPose >= 0)
            {
                _sweepT += Time.deltaTime;
                _w[_sweepPose] = 0.5f * (1f - Mathf.Cos(Mathf.PI * _sweepT)); // 0→1→0，周期 2s
            }

            if (!_liveDriving) ApplyManual();
        }

        private void ApplyManual()
        {
            if (!EnsureMap()) return;
            for (int i = 0; i < _map.Length; i++)
            {
                if (_map[i] < 0) continue;
                Target.SetBlendShapeWeight(_map[i], Mathf.Clamp01(_w[i]) * 100f);
            }
        }

        private void OnDriverFrame(float[] frame)
        {
            // EnsureMap 不能省：pose 源（Driver.PoseNames）就绪后要重建 _map，
            // 否则 frame 下标和 _map 的 pose 序不一致，权重会写串（见 EnsureMap 注释）。
            if (frame == null || !EnsureMap()) return;

            // 增益只保留一个温和的 DriveGain（默认 1，可在面板 0.8~1.5 微调），不再叠加 Driver.Strength：
            // frame 是解算出的 0..1 原始权重（Strength 是 ApplyFrame 写自己 _bindings 时才乘的），
            // 这里按 DriveGain 直接写，避免双重计数。真正的幅度来自 stylization 层（lower_face_strength=1.4 等）。
            // 注意：本回调在 Component.ApplyFrame 末尾（OnFrameApplied）触发，晚于 Component 自己的
            // SetBlendShapeWeight，所以这里是最终写入者。默认 DriveGain=1 即「不额外放大」。
            float gain = DriveGain;

            _live.Clear();
            float rawPeak = 0f;
            for (int i = 0; i < _map.Length && i < frame.Length; i++)
            {
                if (_map[i] < 0) continue;
                float raw = frame[i];
                float scaled = Mathf.Clamp01(raw * gain);
                Target.SetBlendShapeWeight(_map[i], scaled * 100f);
                if (raw > rawPeak) rawPeak = raw;
                if (scaled > 0.01f) _live.Add(new KeyValuePair<string, float>(_poses[i], scaled));
            }
            _live.Sort((a, b) => b.Value.CompareTo(a.Value));
            _rawPeak = rawPeak;

            // 心跳（无论 debugMode）：这条出现 = OnFrameApplied 在触发、帧在往 Target 写。
            // 完全没有这条 = Component.ApplyFrame 根本没跑到（查它的 RealtimePlayback / 音频时钟 / _holdFrames）。
            // 有这条但 jawOpen≈0 = 帧在流、只是权重被基线/幅度压扁（去调 jawOpenGain / stylization）。
            // 若与 [A2F写入] 的 jawOpen 峰值对不上（写入≈1、这里≈0.x）→ pose 序又不同步了，查 EnsureMap。
            _driveFrames++;
            float jawW = -1f;
            int jawIdx = IndexOf("jawOpen");
            if (jawIdx >= 0 && jawIdx < frame.Length) jawW = Mathf.Clamp01(frame[jawIdx] * gain);
            if (jawW > _driveJawPeak) _driveJawPeak = jawW;
            if (_driveFrames % 30 == 0)
            {
                Debug.Log($"[A2F驱动] OnDriverFrame 已写 {_driveFrames} 帧 | 本秒 jawOpen 写入峰值={_driveJawPeak:F2} (DriveGain×{gain:F2})" +
                          $" | 最近帧 jawOpen={jawW:F2} | 本帧非零 pose 数={_live.Count}" +
                          (_driveJawPeak < 0.03f ? "  ← 权重≈0：帧在流但幅度太小（调 jawOpenGain / stylization）" : ""));
                _driveJawPeak = 0f;
            }
        }

        // ================= 测试动作 =================

        public void ZeroAll()
        {
            if (!EnsureMap()) return;
            for (int i = 0; i < _w.Length; i++) _w[i] = 0f;
            _sweepPose = -1;
            _liveDriving = false;
            ApplyManual();
            _status = "全部归零";
        }

        public void SetPose(string poseName, float value)
        {
            if (!EnsureMap()) { _status = "目标网格未就绪"; return; }
            int i = IndexOf(poseName);
            if (i < 0) { _status = $"没有 {poseName} 这个 pose"; return; }
            if (_map[i] < 0) { _status = $"{poseName} 在网格里找不到对应 blendshape"; return; }
            _w[i] = value;
            _status = $"{poseName} = {value:F2}（网格 #{_map[i]}）";
        }

        public void StartSweep(string poseName)
        {
            if (!EnsureMap()) { _status = "目标网格未就绪"; return; }
            int i = IndexOf(poseName);
            if (i < 0) { _status = $"没有 {poseName} 这个 pose"; return; }
            if (_map[i] < 0) { _status = $"{poseName} 在网格里找不到对应 blendshape"; return; }
            _sweepPose = i;
            _sweepT = 0f;
            _liveDriving = false;
            _status = $"扫掠 {poseName}：0→1→0（盯住脸，看有没有连续变形）";
        }

        public void StopSweep()
        {
            if (_sweepPose >= 0) _w[_sweepPose] = 0f;
            _sweepPose = -1;
            _status = "扫掠停止";
        }

        /// <summary>嘴部一组同时拉满，用来确认下半脸整体是活的。</summary>
        public void MouthGroup()
        {
            string[] group = { "jawOpen", "mouthFunnel", "mouthPucker", "mouthClose", "mouthShrugUpper", "mouthShrugLower" };
            foreach (var g in group) SetPose(g, 1f);
            _status = "嘴部一组拉满";
        }

        public void BlinkGroup()
        {
            SetPose("eyeBlinkLeft", 1f);
            SetPose("eyeBlinkRight", 1f);
            _status = "眨眼拉满";
        }

        public void PlayClipAndDrive()
        {
            if (Clip == null) { _status = "没有拖 AudioClip 进来"; return; }
            if (Driver == null) { _status = "没有 Audio2FaceDiffusionComponent，无法驱动"; return; }
            if (!Driver.IsReady) { _status = "管线还没就绪"; return; }

            _liveDriving = true;
            _sweepPose = -1;
            _driveFrames = 0;
            _driveJawPeak = 0f;

            // 声音统一交给 Driver 播放（PlayAudioWhileProcessing）：ProcessClipRoutine 会用那个
            // AudioSource 的播放时钟同步动画帧，音画才对得上。
            // ⚠ 之前这里自己再播一遍 Clip：如果是同一个 AudioSource，ProcessClipRoutine 预热完
            // 会把 time=0 重播（声音跳回开头、动画时钟被重置）；如果是两个 AudioSource，会「双声 + 嘴对不上音」。
            // 两种都让「播放并驱动」看着像嘴不动。所以只有 Driver 不出声时，本组件才自己出声兜底。
            bool driverWillPlay = Driver.PlayAudioWhileProcessing;
            if (PlayThroughSpeakers && !driverWillPlay)
            {
                _audio.clip = Clip;
                _audio.Play();
            }
            Driver.ProcessAudioClip(Clip);
            _status = "正在播放并驱动…（看嘴）";
            EnsureMap();
            string tname = Target != null
                ? $"{Target.name}(mesh={(Target.sharedMesh != null ? Target.sharedMesh.name : "无")})" : "未解析";
            int jidx = IndexOf("jawOpen");
            string order = _poses == null ? "未建"
                : ReferenceEquals(_poses, ArkItBlendshapeNames.All52) ? "ARKit 52 字母序(兜底⚠ 与 frame 不同序)"
                : "Driver.PoseNames(与 frame 同序)";
            Debug.Log($"[A2F驱动] 启动播放并驱动：Target={tname} " +
                      $"音频={(driverWillPlay ? "Driver统一播放(音画同步)" : "本组件播放(兜底)")}" +
                      $" | pose序={order}×{_poses?.Length ?? 0}" +
                      $" | jawOpen@idx={jidx}→网格#{(jidx >= 0 && _map != null ? _map[jidx] : -1)}" +
                      "。请盯 Console 的 [A2F驱动]/[A2F写入]/[A2F实测] 三条判断嘴到底动没动。");
        }

        public void StopDrive()
        {
            _liveDriving = false;
            if (_audio != null) _audio.Stop();
            ZeroAll();
            _status = "驱动已停";
        }

        private int IndexOf(string poseName)
        {
            if (_poses == null) return -1;
            for (int i = 0; i < _poses.Length; i++)
                if (_poses[i] == poseName) return i;
            return -1;
        }

        // ================= 面板 =================

        void OnGUI()
        {
            if (!ShowPanel || !_ready) return;

            float s = Mathf.Max(0.5f, PanelScale);
            var oldMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(new Vector3(10, 10, 0), Quaternion.identity, new Vector3(s, s, 1));

            const float W = 430f;
            float H = Screen.height / s - 20f;

            GUILayout.BeginVertical(GUI.skin.box, GUILayout.Width(W), GUILayout.Height(H));

            GUILayout.Label($"<b>blendshape 测试台</b>  {Target.sharedMesh.name}");
            GUILayout.Label($"顶点 {Target.sharedMesh.vertexCount} · blendshape {Target.sharedMesh.blendShapeCount} · " +
                            $"匹配 <b>{_matched}/{_poses.Length}</b>");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("全部归零")) ZeroAll();
            if (GUILayout.Button("张嘴")) SetPose("jawOpen", 1f);
            if (GUILayout.Button("眨眼")) BlinkGroup();
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("扫掠 jawOpen")) StartSweep("jawOpen");
            if (GUILayout.Button("停扫掠")) StopSweep();
            if (GUILayout.Button("嘴部一组")) MouthGroup();
            if (GUILayout.Button("打印映射表")) Debug.Log(_report);
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUI.enabled = Clip != null && Driver != null;
            if (GUILayout.Button("播放并驱动")) PlayClipAndDrive();
            GUI.enabled = true;
            if (GUILayout.Button("停驱动")) StopDrive();
            GUILayout.Label(Driver == null ? "Driver 未连接" : (Driver.IsReady ? "Driver 就绪" : "Driver 未就绪"));
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUILayout.Label($"口型增益 ×{DriveGain:F2}", GUILayout.Width(120));
            DriveGain = GUILayout.HorizontalSlider(DriveGain, 0.8f, 1.5f, GUILayout.Width(150));
            GUILayout.Label("(默认1，别再用它放大口型)", GUILayout.Width(170));
            GUILayout.EndHorizontal();

            if (_liveDriving && _live.Count > 0)
            {
                var sb = new StringBuilder("实时权重 top(已放大): ");
                for (int i = 0; i < _live.Count && i < 6; i++)
                    sb.Append($"{_live[i].Key}={_live[i].Value:F2}  ");
                GUILayout.Label(sb.ToString());
                GUILayout.Label($"原始峰值={_rawPeak:F3} · 输出峰值={Mathf.Min(1f, _rawPeak * DriveGain):F2}" +
                                (_rawPeak < 0.03f ? "  ← 原始权重 <0.03，幅度问题应在 stylization 层解决（lower_face_strength/input_strength）" : ""));
            }

            if (!string.IsNullOrEmpty(_status)) GUILayout.Label(_status);

            GUILayout.Space(4);
            Scroll = GUILayout.BeginScrollView(Scroll, GUILayout.Width(W - 20), GUILayout.Height(H - 190));

            for (int i = 0; i < _poses.Length; i++)
            {
                GUILayout.BeginHorizontal();
                if (_map[i] < 0)
                {
                    var old = GUI.color;
                    GUI.color = Color.gray;
                    GUILayout.Label($"✗ {_poses[i]}  (网格无此形状)", GUILayout.Width(W - 40));
                    GUI.color = old;
                }
                else
                {
                    GUILayout.Label($"{_poses[i]}  #{_map[i]}", GUILayout.Width(190));
                    float nv = GUILayout.HorizontalSlider(_w[i], 0f, 1f, GUILayout.Width(150));
                    if (!Mathf.Approximately(nv, _w[i]))
                    {
                        _w[i] = nv;
                        _sweepPose = -1;
                        _liveDriving = false;
                    }
                    GUILayout.Label(_w[i].ToString("F2"), GUILayout.Width(34));
                }
                GUILayout.EndHorizontal();
            }

            GUILayout.EndScrollView();
            GUILayout.EndVertical();

            GUI.matrix = oldMatrix;
        }
    }
}
