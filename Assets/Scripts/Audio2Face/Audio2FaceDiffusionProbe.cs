using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// 一次性自检脚本：加载模型 → 喂 3 个窗口的音频（跨过预热）→ 打印
    /// prediction 统计、反解出的 blendshape 权重和耗时。
    /// 用来确认扩散管线在 Unity 里真的能跑通、以及单次推理耗时是否够实时。
    /// </summary>
    public class Audio2FaceDiffusionProbe : MonoBehaviour
    {
        public Audio2FaceDiffusionConfig Config = new Audio2FaceDiffusionConfig();
        [Tooltip("可选：测试音频，留空则用静音")]
        public AudioClip TestClip;
        public bool RunOnStart = true;

        private Audio2FaceDiffusionPipeline _pipeline;

        IEnumerator Start()
        {
            if (!RunOnStart) yield break;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                _pipeline = new Audio2FaceDiffusionPipeline(Config);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[A2F Probe] 初始化失败: {ex.Message}\n{ex.StackTrace}");
                yield break;
            }
            sw.Stop();
            Debug.Log($"[A2F Probe] 管线初始化耗时 {sw.ElapsedMilliseconds} ms");

            var info = _pipeline.Info;
            Debug.Log($"[A2F Probe] 模型: type={info.ModelType} output={info.OutputType} " +
                      $"skin={info.SkinSize} tongue={info.TongueSize} jaw={info.JawSize} eyes={info.EyesSize} " +
                      $"总计={info.TotalDim}");
            Debug.Log($"[A2F Probe] 帧: 每推理 {info.FramesPerInference} 帧 " +
                      $"(左丢 {info.FramesLeftTruncate} / 取中 {info.FramesCenter} / 右丢 {info.FramesRightTruncate}) " +
                      $"步长 {info.StrideSamples} 样本 帧率 {info.FrameRate:F1} 预热 {info.WarmupInferences} 次");

            // 准备音频
            float[] pcm = null;
            if (TestClip != null)
            {
                var raw = new float[TestClip.samples * TestClip.channels];
                TestClip.GetData(raw, 0);
                var mono = new float[TestClip.samples];
                for (int i = 0; i < TestClip.samples; i++) mono[i] = raw[i * TestClip.channels];
                pcm = mono;
            }

            // 跨过预热，跑到第一次真正出帧
            int needed = info.WarmupInferences + 1;
            int cursor = 0;
            for (int n = 0; n < needed; n++)
            {
                var window = new float[info.BufferLength];
                if (pcm != null)
                {
                    int copy = Mathf.Min(info.BufferLength, pcm.Length - cursor);
                    if (copy > 0) Array.Copy(pcm, cursor, window, 0, copy);
                    cursor += copy;
                }

                var t0 = Time.realtimeSinceStartup;
                yield return RunOnce(window, info);
                Debug.Log($"[A2F Probe] 第 {n + 1} 次推理: {_pipeline.LastInferenceMs:F1} ms " +
                          $"(墙钟 {(Time.realtimeSinceStartup - t0) * 1000f:F1} ms)");
            }

            // 诊断阶段必须先停掉后台线程：求解器复用 _target/_x/_result，
            // 后台同时在解帧会把自检结果冲掉（现象是「权重对、残差乱跳」）。
            _pipeline.SetPaused(true);

            // prediction 统计
            if (_pipeline.Prediction != null)
            {
                var pred = _pipeline.Prediction;
                int totalDim = info.TotalDim;
                ReportSegment("skin", pred, info.SkinOffset, info.SkinSize, totalDim);
                ReportSegment("tongue", pred, info.TongueOffset, info.TongueSize, totalDim);
                ReportSegment("jaw", pred, info.JawOffset, info.JawSize, totalDim);
                ReportSegment("eyes", pred, info.EyesOffset, info.EyesSize, totalDim);
            }

            // prediction 量级诊断：模型输出到底是 delta 还是绝对几何？
            // 判据：neutral 的 mean|v| ≈ 54、max ≈ 170；delta 应该远小于这个量级。
            ReportDeltaSanity();

            // 逐帧 RMS：验证 prediction 的帧布局（左15/中30/右15），顺带看有没有异常帧
            ReportPerFrameRms();

            // 求解器自检：先把「已知 pose」喂进去，确认掩码/AMat/BVLS 这一套本身是对的。
            // 这一步过了，再谈 prediction 语义对不对 —— 否则残差 0.97 根本分不清是谁的锅。
            RunSolverSelfTest();

            // 解出的权重：两种模式各跑一次，直接看出该不该减 neutral
            Debug.Log($"[A2F Probe] 帧队列 {_pipeline.PendingFrames} 帧");
            var frame = new float[_pipeline.FrameStride];
            if (_pipeline.TryDequeueFrame(frame))
            {
                var skin = _pipeline.SkinSolver;

                // 1) 管线实际路径
                ReportWeights(_pipeline.UsingSkinAnimator
                    ? "管线路径 (animator ON：绝对顶点 − neutral)"
                    : "管线路径 (animator OFF：prediction 直接当 delta)");

                // 2) 反例：把 delta 语义弄反一次，权重应该全部顶到上界。
                //    没饱和反而说明当前那条路径的 target 量级本身就不对。
                bool origDelta = skin.TargetIsDelta;
                skin.TargetIsDelta = !origDelta;
                skin.Reset();       // 清掉上一解的时间正则，避免污染对比
                ReportWeights($"反例 (TargetIsDelta={(!origDelta).ToString().ToLower()}) → 权重应饱和");
                skin.TargetIsDelta = origDelta;
                skin.Reset();

                // 3) 去偏置：把中心 30 帧的均值当静态偏置减掉。
                //    扩散模型静音时给的是「合理但未必等于 rig neutral」的脸，这个常量偏置
                //    会让 BVLS 硬凑出非零权重（人没说话嘴也在动）。减掉后静音权重应回到 0。
                var bias = new float[skin.SolvedPositionCount];
                float biasRms = ComputeBias(bias);
                skin.TargetBias = bias;
                skin.Reset();
                ReportWeights($"去偏置模式 (减中心{info.FramesCenter}帧均值, 偏置RMS={biasRms:F4})");
                skin.TargetBias = null;
                skin.Reset();
            }
            else
            {
                Debug.LogWarning("[A2F Probe] 帧队列是空的，预热次数可能大于实际推理次数");
            }

            // animator 到底改变了什么：同一帧开/关对比
            ReportAnimatorEffect();

            // 噪声敏感性：同一个静音窗口只换噪声种子，看输出变多少。
            // 扩散模型是生成式的，换种子会换一张「同样合理的静音脸」，
            // 所以这一步要回答的是：静音下那个常量偏置是噪声采样带来的、还是模型系统性的。
            ReportNoiseSensitivity();

            // 音频响应：静音 → 类语音激励，比较中心 30 帧的 jawOpen 权重动态范围。
            // 用类语音（基频+共振峰+音节包络）而不是 440Hz 纯音：模型是在语音上训练的，对纯音几乎没反应。
            var silence = new float[info.BufferLength];
            var speech = MakeSpeechLike(info.BufferLength, info.SampleRate);

            var jSilence = JawOpenRange(silence, info);
            var jSpeech = JawOpenRange(speech, info);

            Debug.Log($"[A2F Probe] 音频响应(中心30帧 jawOpen): 静音 {jSilence.x:F3}~{jSilence.y:F3} " +
                      $"(动态 {jSilence.y - jSilence.x:F3}) → 类语音 {jSpeech.x:F3}~{jSpeech.y:F3} " +
                      $"(动态 {jSpeech.y - jSpeech.x:F3}) → " +
                      $"{(jSpeech.y - jSpeech.x > 3f * Mathf.Max(1e-4f, jSilence.y - jSilence.x) ? "模型有响应" : "模型没响应，音频没进到网络")}");

            // 静息脸标定（权重空间）：静音时模型给的是「训练集的边际平均脸」，不是 rig 的 neutral，
            // 所以静音也会解出非零权重（嘴微张）。标定把这张脸记成基线，输出变成「相对静息脸的形变」。
            //
            // 注意必须在<b>权重空间</b>做，不能在顶点空间：静音输出有约 25% 能量落在 blendshape
            // 张成的子空间之外，而且这部分还在缓慢漂移。减顶点均值（CalibrateSilenceBaseline）
            // 实测只能把 jawOpen 从 0.080 压到 0.075 —— 减了等于没减。
            //
            // ⚠ 标定前必须先喂「干净的冷静音」：把模型 GRU 状态清空、连跑几个静音窗口让 prediction 收敛。
            // 否则若在「语音之后的静音」上标定，GRU 的时间记忆会把静息 jawOpen 抬到 0.12+，超过语音峰值(0.12)，
            // 一减一 clamp，整个张嘴动态就被吃掉了 —— 之前这步误报「静音已归零，可以直接用」。
            _pipeline.SetNoiseSeed(12345);
            _pipeline.Reset();
            for (int n = 0; n < info.WarmupInferences + 3; n++) _pipeline.RunNow(silence);
            var restPose = _pipeline.CalibrateRestPose();
            var jSilCal = JawOpenRange(silence, info);
            var jSpCal = JawOpenRange(speech, info);

            int jawIdx = PoseIndex("jawOpen");
            float jawBase = jawIdx >= 0 ? restPose[jawIdx] : 0f;
            var sb = new StringBuilder();
            sb.AppendLine($"[A2F Probe] === 静息脸标定（权重空间）jawOpen 基线={jawBase:F3} ===");
            sb.Append("    基线 top8: ");
            foreach (int i in TopWeightIndices(restPose, 8))
                sb.Append($"{_pipeline.SkinSolver.PoseNames[i]}={restPose[i]:F3}  ");
            sb.AppendLine();
            sb.AppendLine($"    标定后: 静音 jawOpen {jSilCal.x:F3}~{jSilCal.y:F3} " +
                          $"→ 类语音 {jSpCal.x:F3}~{jSpCal.y:F3} (动态 {jSpCal.y - jSpCal.x:F3})");
            // 真正判据：静音要归零「且」语音张嘴动态必须保留。只看静音归零会漏掉「动态被吃光」的失败。
            bool silOk = jSilCal.y < 0.02f;
            bool spOk = (jSpCal.y - jSpCal.x) > 0.02f;
            sb.AppendLine($"    → {(silOk ? "静音已归零" : "静音仍有残余")}；" +
                          (spOk
                              ? $"语音张嘴动态保留 (动态 {jSpCal.y - jSpCal.x:F3})，可以直接用"
                              : "⚠ 语音张嘴动态被吃掉了！基线估计偏高（多在「语音后静音」上标定导致），请改用冷静音标定或加标定增益"));
            Debug.Log(sb.ToString());

            // 真实语音对照：合成激励（110Hz 嗡鸣）对在真实语音上训练的模型只是个弱信号，
            // 动态范围天然偏小。放一个 16bit PCM wav 进来才能真正判断口型幅度够不够。
            var real = TryLoadWav16k(Path.Combine(Application.streamingAssetsPath, "a2f_test.wav"), info.SampleRate);
            if (real != null)
            {
                var jRealBefore = JawOpenRange(real, info);
                _pipeline.ClearRestPose();
                Debug.Log($"[A2F Probe] 真实语音 a2f_test.wav ({real.Length * 1000f / info.SampleRate:F0} ms): " +
                          $"未标定 jawOpen {jRealBefore.x:F3}~{jRealBefore.y:F3} (动态 {jRealBefore.y - jRealBefore.x:F3}) " +
                          $"→ {(jRealBefore.y - jRealBefore.x > 0.15f ? "幅度充足" : "幅度偏小，检查音频电平/采样率")}");
            }
            else
            {
                Debug.Log("[A2F Probe] 未找到 StreamingAssets/a2f_test.wav（16bit PCM）→ 跳过真实语音对照。" +
                          "放一个进来可以直接看口型幅度是否够。");
            }

            _pipeline.ClearRestPose();   // 清掉，别影响后续

            float budget = info.StrideSamples * 1000f / info.SampleRate;
            Debug.Log($"[A2F Probe] 实时预算: 每 {budget:F0} ms 必须完成一次推理，当前 {_pipeline.LastInferenceMs:F1} ms → " +
                      $"{(_pipeline.LastInferenceMs < budget ? "可以实时" : "追不上，口型会卡顿")}");

            _pipeline.SetPaused(false);
        }

        /// <summary>类语音激励：110Hz 基频 + 3 个共振峰谐波 + 4Hz 音节包络。比纯音更能触发模型。</summary>
        private static float[] MakeSpeechLike(int length, int sampleRate)
        {
            var buf = new float[length];
            var rnd = new System.Random(7);
            float[] formants = { 500f, 1500f, 2500f };
            for (int i = 0; i < length; i++)
            {
                float t = (float)i / sampleRate;
                float s = 0f;
                for (int h = 1; h <= 5; h++) s += Mathf.Sin(2f * Mathf.PI * 110f * h * t) / h;
                for (int f = 0; f < formants.Length; f++)
                    s += 0.5f * Mathf.Sin(2f * Mathf.PI * formants[f] * t) / (f + 1);
                float syllable = 0.5f + 0.5f * Mathf.Sin(2f * Mathf.PI * 4f * t);
                buf[i] = 0.4f * s * syllable + 0.02f * ((float)rnd.NextDouble() - 0.5f);
            }
            return buf;
        }

        private int PoseIndex(string name)
        {
            var names = _pipeline.SkinSolver.PoseNames;
            for (int i = 0; i < names.Length; i++) if (names[i] == name) return i;
            return -1;
        }

        /// <summary>按权重从大到小取前 n 个 pose 下标。</summary>
        private static List<int> TopWeightIndices(float[] w, int n)
        {
            var list = new List<int>();
            for (int i = 0; i < w.Length; i++) list.Add(i);
            list.Sort((a, b) => w[b].CompareTo(w[a]));
            return list.GetRange(0, Mathf.Min(n, list.Count));
        }

        /// <summary>
        /// 读 16bit PCM wav 并重采样到模型采样率。只支持最朴素的 RIFF/PCM，读不了就返回 null
        /// （这是个可选诊断项，不能因为一个测试文件把自检搞挂）。
        /// </summary>
        private static float[] TryLoadWav16k(string path, int targetRate)
        {
            try
            {
                if (!File.Exists(path)) return null;
                var bytes = File.ReadAllBytes(path);
                if (bytes.Length < 44) return null;
                if (bytes[0] != 'R' || bytes[1] != 'I' || bytes[2] != 'F' || bytes[3] != 'F') return null;

                int fmtOffset = -1, dataOffset = -1, dataSize = 0;
                int pos = 12;
                while (pos + 8 <= bytes.Length)
                {
                    string id = System.Text.Encoding.ASCII.GetString(bytes, pos, 4);
                    int size = BitConverter.ToInt32(bytes, pos + 4);
                    if (id == "fmt ") fmtOffset = pos + 8;
                    else if (id == "data") { dataOffset = pos + 8; dataSize = size; break; }
                    pos += 8 + size + (size % 2);
                }
                if (fmtOffset < 0 || dataOffset < 0) return null;

                int audioFormat = BitConverter.ToUInt16(bytes, fmtOffset);
                int channels = BitConverter.ToUInt16(bytes, fmtOffset + 2);
                int rate = BitConverter.ToInt32(bytes, fmtOffset + 4);
                int bits = BitConverter.ToUInt16(bytes, fmtOffset + 14);
                if (audioFormat != 1 || bits != 16 || channels < 1) return null;

                int samples = Mathf.Max(0, Mathf.Min(dataSize, bytes.Length - dataOffset)) / 2;
                int frames = samples / channels;
                var mono = new float[frames];
                for (int i = 0; i < frames; i++)
                {
                    float acc = 0f;
                    for (int c = 0; c < channels; c++)
                        acc += BitConverter.ToInt16(bytes, dataOffset + (i * channels + c) * 2) / 32768f;
                    mono[i] = acc / channels;
                }

                if (rate == targetRate) return mono;

                // 线性重采样（诊断用，不追求抗混叠）
                int outLen = (int)((long)frames * targetRate / rate);
                var res = new float[outLen];
                for (int i = 0; i < outLen; i++)
                {
                    double t = (double)i * rate / targetRate;
                    int i0 = (int)t;
                    float frac = (float)(t - i0);
                    float a = i0 < frames ? mono[i0] : 0f;
                    float b = i0 + 1 < frames ? mono[i0 + 1] : a;
                    res[i] = a + (b - a) * frac;
                }
                return res;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[A2F Probe] wav 读取失败: {e.Message}");
                return null;
            }
        }

        /// <summary>跑一个窗口，返回中心 30 帧 jawOpen 权重的最小/最大值。</summary>
        private Vector2 JawOpenRange(float[] window, Audio2FaceNetworkInfo info)
        {
            int jawIdx = -1;
            var names = _pipeline.SkinSolver.PoseNames;
            for (int i = 0; i < names.Length; i++)
                if (names[i] == "jawOpen") { jawIdx = i; break; }
            if (jawIdx < 0) return Vector2.zero;

            _pipeline.RunNow(window);
            var pred = _pipeline.Prediction;

            // animator 开着时必须先过 animator：求解器等待的是绝对顶点（TargetIsDelta=false），
            // 直接喂 prediction 会被当成绝对坐标减一次 neutral，权重全部饱和。
            var animator = _pipeline.UsingSkinAnimator ? _pipeline.SkinAnimator : null;
            var verts = _pipeline.SkinVertices;
            float dt = 1f / info.FrameRate;
            if (animator != null) animator.Reset();

            float lo = float.MaxValue, hi = float.MinValue;
            for (int f = info.FramesLeftTruncate; f < info.FramesLeftTruncate + info.FramesCenter; f++)
            {
                float[] w;
                if (animator != null)
                {
                    animator.Animate(pred, f * info.TotalDim + info.SkinOffset, verts, 0, dt);
                    w = _pipeline.SkinSolver.Solve(verts, 0);
                }
                else
                {
                    w = _pipeline.SkinSolver.Solve(pred, f * info.TotalDim + info.SkinOffset);
                }
                if (w == null) continue;
                if (w[jawIdx] < lo) lo = w[jawIdx];
                if (w[jawIdx] > hi) hi = w[jawIdx];
            }
            _pipeline.SkinSolver.Reset();
            return lo > hi ? Vector2.zero : new Vector2(lo, hi);
        }

        /// <summary>
        /// 只换噪声种子、其他输入完全相同，看 prediction 变多少。
        /// 变化小 → 输出由音频决定；变化大 → 输出被噪声采样主导（静音偏置多半来自这里）。
        /// </summary>
        private void ReportNoiseSensitivity()
        {
            var info = _pipeline.Info;
            var silence = new float[info.BufferLength];

            _pipeline.SetNoiseSeed(1001);
            _pipeline.Reset();
            for (int n = 0; n < info.WarmupInferences + 1; n++) _pipeline.RunNow(silence);
            float rmsA = CenterSkinRms();
            var snapA = CenterFrameSnapshot();

            _pipeline.SetNoiseSeed(9999);
            _pipeline.Reset();
            for (int n = 0; n < info.WarmupInferences + 1; n++) _pipeline.RunNow(silence);
            float rmsB = CenterSkinRms();
            var snapB = CenterFrameSnapshot();

            double diff = 0.0, norm = 0.0;
            for (int i = 0; i < snapA.Length; i++)
            {
                double d = snapA[i] - snapB[i];
                diff += d * d;
                norm += (double)snapA[i] * snapA[i];
            }
            float rel = norm > 0 ? (float)Math.Sqrt(diff / norm) : 0f;

            Debug.Log($"[A2F Probe] 噪声敏感性(同静音/换种子): 中心帧 skin RMS {rmsA:F4} vs {rmsB:F4}，差异 {rel * 100f:F1}% → " +
                      $"{(rel > 0.3f ? "输出被噪声主导，静音偏置主要来自噪声采样" : "输出基本由音频/身份决定")}");
        }

        /// <summary>中心 30 帧 skin 段的一份拷贝，避免被下一次推理覆盖。</summary>
        private float[] CenterFrameSnapshot()
        {
            var pred = _pipeline.Prediction;
            var info = _pipeline.Info;
            var dst = new float[info.FramesCenter * info.SkinSize];
            for (int f = 0; f < info.FramesCenter; f++)
                Array.Copy(pred, (info.FramesLeftTruncate + f) * info.TotalDim + info.SkinOffset,
                           dst, f * info.SkinSize, info.SkinSize);
            return dst;
        }

        /// <summary>推一段音频进去并等到这次推理真正跑完（用 InferenceCount 计数，避免 IsBusy 竞态）。</summary>
        private IEnumerator RunOnce(float[] window, Audio2FaceNetworkInfo info)
        {
            int target = _pipeline.InferenceCount + 1;
            _pipeline.PushAudio(window, 0, info.BufferLength);
            _pipeline.Tick();
            yield return new WaitUntil(() => _pipeline.InferenceCount >= target);
        }

        private static float SegmentRms(float[] pred, int offset, int size)
        {
            if (pred == null || size <= 0) return 0f;
            double s = 0.0;
            for (int i = 0; i < size; i++) s += (double)pred[offset + i] * pred[offset + i];
            return (float)Math.Sqrt(s / size);
        }

        /// <summary>
        /// 判定 prediction 是不是 delta：
        /// 把 skin 段按 (顶点, 3) 算位移幅度的分位数，并比较 neutral 的幅度。
        /// 若 prediction 是绝对几何，|v| 的分位数会和 neutral 同量级（几十）；
        /// 若是 delta，应该只有个位数。
        /// </summary>
        private void ReportDeltaSanity()
        {
            var pred = _pipeline.Prediction;
            var info = _pipeline.Info;
            if (pred == null) return;

            int totalDim = info.TotalDim;
            int frame = info.FramesLeftTruncate;              // 取中心窗口的第一帧
            int row = frame * totalDim + info.SkinOffset;
            int n = info.SkinSize / 3;

            // 每顶点位移幅度
            var mags = new float[n];
            for (int v = 0; v < n; v++)
            {
                float x = pred[row + 3 * v], y = pred[row + 3 * v + 1], z = pred[row + 3 * v + 2];
                mags[v] = Mathf.Sqrt(x * x + y * y + z * z);
            }
            Array.Sort(mags);
            float p50 = mags[n / 2], p90 = mags[(int)(n * 0.9f)], p99 = mags[(int)(n * 0.99f)], mx = mags[n - 1];

            // neutral 的幅度（用未掩码的 neutral 不好取，这里用模型段的绝对值粗略对比）
            float neutralMeanAbs = 54.27f;   // 实测 bs_skin_Mark.npz neutral: mean|v|=54.27, rms=88.89

            Debug.Log($"[A2F Probe] skin 每顶点位移幅度 中位={p50:F4} p90={p90:F4} p99={p99:F4} max={mx:F4}");
            Debug.Log($"[A2F Probe] 对比 neutral mean|v|={neutralMeanAbs:F2} → " +
                      (mx < neutralMeanAbs * 0.5f
                          ? "prediction 是 DELTA（不要减 neutral）"
                          : "prediction 像绝对几何（需要减 neutral）"));

            // 帧间变化：静音输入下中心 30 帧应该几乎不变
            float diff = 0f;
            int frames = info.FramesCenter;
            for (int f = 1; f < frames; f++)
            {
                int a = (info.FramesLeftTruncate + f - 1) * totalDim + info.SkinOffset;
                int b = (info.FramesLeftTruncate + f) * totalDim + info.SkinOffset;
                double s = 0.0;
                for (int i = 0; i < info.SkinSize; i += 7) s += Math.Abs(pred[b + i] - pred[a + i]);
                diff += (float)(s / (info.SkinSize / 7));
            }
            Debug.Log($"[A2F Probe] 中心 {frames} 帧平均帧间变化 = {diff / (frames - 1):F5} " +
                      $"(静音输入下应接近 0)");
        }

        /// <summary>
        /// 中心 30 帧的 skin RMS。左右截断区是模型填的固定值，不能用来判断音频响应。
        /// </summary>
        private float CenterSkinRms()
        {
            var pred = _pipeline.Prediction;
            var info = _pipeline.Info;
            if (pred == null) return 0f;

            int td = info.TotalDim;
            double s = 0.0;
            long n = 0;
            for (int f = 0; f < info.FramesCenter; f++)
            {
                int row = (info.FramesLeftTruncate + f) * td + info.SkinOffset;
                for (int i = 0; i < info.SkinSize; i++)
                {
                    s += (double)pred[row + i] * pred[row + i];
                    n++;
                }
            }
            return n > 0 ? (float)Math.Sqrt(s / n) : 0f;
        }

        /// <summary>
        /// 求解器自检。分两层，判据不同，别混在一起看：
        ///   A) 关掉正则，A = D^T·D 纯恒等检验 → 必须解出该 pose = 1.000、残差 ≈ 0。
        ///      这一条只验「掩码 + delta 矩阵 + BVLS」，任何一项错这里就过不去。
        ///   B) 打开正则（SDK 真实路径）→ 单 pose 解的权重理论值 ≈ |d|²/(|d|² + 对角正则)。
        ///      Mark 的配置里对角正则 = L2*10*sf + Temporal*100*sf ≈ 5.94，
        ///      而 52 个 pose 的 |d|² 只有 7.8 ~ 4565（中位 118），
        ///      所以 jawOpen(|d|²=4565) 能解到 0.999，eyeSquint(|d|²=10) 只能解到 0.61 —— 这是设计使然，不是 bug。
        ///      同理 symmetry 项 = 100*10*sf ≈ 990，单独喂一侧 pose 会被强行拉成左右相等，残差天然偏高。
        /// </summary>
        private void RunSolverSelfTest()
        {
            var solver = _pipeline.SkinSolver;
            int p = solver.SolvedPositionCount;
            var buf = new float[p];
            var sb = new StringBuilder();
            sb.AppendLine($"[A2F Probe] === 求解器自检 (分量={p}, symmetryPoses={solver.SymmetryPosesCount}, " +
                          $"对角正则={solver.DiagonalRegularization:F2}) ===");

            var top = TopDeltaPoses(6);

            // A) 无正则恒等检验
            solver.DisableRegularization = true;
            sb.AppendLine("  A) 无正则恒等检验 (A = D^T·D)：应 自身≈1.000 残差≈0");
            foreach (int idx in top)
            {
                solver.Reset();
                if (!solver.TryGetMaskedDelta(idx, buf)) continue;
                var w = solver.SolveMaskedTarget(buf);
                float res = solver.ComputeRelativeResidual();
                bool ok = w[idx] > 0.999f && res < 1e-3f;
                sb.AppendLine($"     {solver.PoseNames[idx],-18} |d|={solver.PoseDeltaNorm(idx),7:F2} → " +
                              $"自身={w[idx]:F4} 残差={res:E2} {(ok ? "OK" : "异常 → 掩码/D/BVLS 有错")}");
            }
            solver.DisableRegularization = false;

            // B) 带正则（真实路径）
            // 只挑无对称伙伴的 pose：symmetry 项 ≈ 990，单独喂一侧 delta 会被强行拉成左右相等，
            // 自身权重必然远低于理论值（实测 mouthSmileLeft 0.22 vs 理论 0.99）—— 那是设计，不是 bug。
            var topAsym = TopDeltaPoses(6, true);

            sb.AppendLine("  B) 带正则（SDK 真实路径）：理论 w ≈ |d|²/(|d|²+对角正则)，小 pose 被压扁属正常");
            float reg = solver.DiagonalRegularization;
            foreach (int idx in topAsym)
            {
                solver.Reset();
                if (!solver.TryGetMaskedDelta(idx, buf)) continue;
                var w = solver.SolveMaskedTarget(buf);
                float res = solver.ComputeRelativeResidual();
                float nd = solver.PoseDeltaNorm(idx);
                float expect = reg > 0f ? (nd * nd) / (nd * nd + reg) : 1f;
                bool ok = res < 0.05f && w[idx] > 0.9f;
                sb.AppendLine($"     {solver.PoseNames[idx],-18} |d|²={nd * nd,8:F1} → 自身={w[idx]:F4} " +
                              $"(理论 {expect:F4}) 残差={res:F4} {(ok ? "OK" : "偏小 — 只在 |d|² 远大于对角正则时才该接近 1")}");
            }

            // 对称对联合：target = dL + dR，两侧都应解出 ≈1
            for (int i = 0; i < _pipeline.SkinPoseCount; i++)
            {
                int partner;
                if (!solver.IsPoseActive(i) || !solver.TryGetSymmetryPartner(i, out partner)) continue;
                if (partner <= i) continue;

                if (!solver.TryGetMaskedDelta(i, buf)) continue;
                var pair = new float[p];
                if (!solver.TryGetMaskedDelta(partner, pair)) continue;
                for (int k = 0; k < p; k++) pair[k] += buf[k];

                solver.Reset();
                var w2 = solver.SolveMaskedTarget(pair);
                float res2 = solver.ComputeRelativeResidual();
                bool ok2 = res2 < 0.25f && w2[i] > 0.3f && w2[partner] > 0.3f;
                sb.AppendLine($"     {solver.PoseNames[i]}+{solver.PoseNames[partner]} (对称对) → " +
                              $"左={w2[i]:F4} 右={w2[partner]:F4} 残差={res2:F4} {(ok2 ? "OK" : "异常")}");
                break;
            }

            // 反例：neutral 不在 delta 张成的空间里，残差应接近 1
            solver.Reset();
            var nw = solver.SolveMaskedTarget(solver.NeutralMasked);
            float nres = solver.ComputeRelativeResidual();
            float nsum = 0f;
            for (int i = 0; i < _pipeline.SkinPoseCount; i++) nsum += nw[i];
            sb.AppendLine($"     neutral(反例)      → 权重合计={nsum:F3} 残差={nres:F4} " +
                          $"{(nres > 0.9f ? "OK（neutral 确实不是 blendshape 能凑出来的）" : "异常 → 求解器退化")}");
            solver.Reset();

            Debug.Log(sb.ToString());
        }

        /// <summary>
        /// 按 |d| 从大到小挑活跃 pose（自检要挑大 pose，小 pose 会被正则压扁）。
        /// </summary>
        /// <param name="skipSymmetric">
        /// true = 排除有左右对称伙伴的 pose。带正则自检时必须开：
        /// symmetry 项 ≈ 990，单独喂一侧 delta 会被强行拉成左右相等，
        /// 自身权重必然远低于理论值（实测 mouthSmileLeft 0.22 vs 理论 0.99），那是设计不是 bug。
        /// </param>
        private List<int> TopDeltaPoses(int count, bool skipSymmetric = false)
        {
            var solver = _pipeline.SkinSolver;
            var list = new List<KeyValuePair<int, float>>();
            for (int i = 0; i < _pipeline.SkinPoseCount; i++)
            {
                if (!solver.IsPoseActive(i)) continue;
                if (skipSymmetric && solver.HasSymmetryPartner(i)) continue;
                list.Add(new KeyValuePair<int, float>(i, solver.PoseDeltaNorm(i)));
            }
            list.Sort((a, b) => b.Value.CompareTo(a.Value));

            var res = new List<int>();
            for (int i = 0; i < list.Count && res.Count < count; i++) res.Add(list[i].Key);
            return res;
        }

        /// <summary>逐帧 skin RMS，验证 prediction 是 [帧][维] 布局，并标出左/中/右截断区。</summary>
        private void ReportPerFrameRms()
        {
            var pred = _pipeline.Prediction;
            var info = _pipeline.Info;
            if (pred == null) return;

            int td = info.TotalDim;
            var sb = new StringBuilder();
            sb.AppendLine($"[A2F Probe] 逐帧 skin RMS (L=左截断{info.FramesLeftTruncate} / C=中心{info.FramesCenter} / R=右截断{info.FramesRightTruncate}):");
            for (int f = 0; f < info.FramesPerInference; f++)
            {
                float rms = SegmentRms(pred, f * td + info.SkinOffset, info.SkinSize);
                string tag = f < info.FramesLeftTruncate ? "L"
                           : (f >= info.FramesLeftTruncate + info.FramesCenter ? "R" : "C");
                sb.Append($"  {tag}{f:D2}={rms:F3}");
                if (f % 6 == 5) sb.AppendLine();
            }
            Debug.Log(sb.ToString());
        }

        /// <summary>解一帧并打印 top-12 权重。animator 开着时走「绝对顶点 − neutral」，否则直接喂 prediction。</summary>
        private void ReportWeights(string modeLabel)
        {
            var pred = _pipeline.Prediction;
            var info = _pipeline.Info;
            int totalDim = info.TotalDim;
            int row = info.FramesLeftTruncate * totalDim;

            float[] skin;
            if (_pipeline.UsingSkinAnimator)
            {
                var verts = _pipeline.SkinVertices;
                _pipeline.SkinAnimator.Reset();   // 每次都从干净状态解单帧，三次对比才公平
                _pipeline.SkinAnimator.Animate(pred, row + info.SkinOffset, verts, 0, 1f / info.FrameRate);
                skin = _pipeline.SkinSolver.Solve(verts, 0);
            }
            else
            {
                skin = _pipeline.SkinSolver.Solve(pred, row + info.SkinOffset);
            }
            if (skin == null) return;

            float targetRms = _pipeline.SkinSolver.LastTargetRms;
            float residual = _pipeline.SkinSolver.ComputeRelativeResidual();

            var sb = new StringBuilder();
            sb.AppendLine($"[A2F Probe] --- {modeLabel} ---");
            sb.AppendLine($"    target RMS={targetRms:F4}  相对残差={residual:F4} " +
                          $"(残差>0.8 说明这帧几何根本不是 blendshape 能表示的脸)");
            sb.AppendLine("    skin 权重 top 12:");

            var items = new List<KeyValuePair<string, float>>();
            for (int i = 0; i < _pipeline.SkinPoseCount; i++)
                items.Add(new KeyValuePair<string, float>(_pipeline.SkinSolver.PoseNames[i], skin[i]));
            items.Sort((a, b) => b.Value.CompareTo(a.Value));
            for (int i = 0; i < Mathf.Min(12, items.Count); i++)
                sb.AppendLine($"      {items[i].Key,-20} {items[i].Value:F4}");
            Debug.Log(sb.ToString());

            float sum = 0f, max = 0f;
            int saturated = 0;
            for (int i = 0; i < _pipeline.SkinPoseCount; i++)
            {
                sum += skin[i];
                if (skin[i] > max) max = skin[i];
                if (skin[i] > 0.999f) saturated++;
            }
            Debug.Log($"[A2F Probe] [{modeLabel}] 合计={sum:F3} 最大={max:F4} 顶到上界的个数={saturated}/52 " +
                      $"→ {(saturated > 6 ? "饱和，说明 target 量级错了" : "正常")}");
        }

        /// <summary>
        /// 中心 30 帧的逐分量均值，作为静态偏置。
        /// animator 开着时必须在 animator 输出上算——那才是求解器看到的量；
        /// 而求解器是「先减 neutral、再减 bias」，所以这里要先把 neutral 扣掉。
        /// </summary>
        private float ComputeBias(float[] bias)
        {
            var skin = _pipeline.SkinSolver;
            var info = _pipeline.Info;
            var pred = _pipeline.Prediction;

            if (_pipeline.UsingSkinAnimator)
            {
                int size = info.SkinSize;
                var all = new float[info.FramesCenter * size];
                float dt = 1f / info.FrameRate;
                _pipeline.SkinAnimator.Reset();
                for (int f = 0; f < info.FramesCenter; f++)
                    _pipeline.SkinAnimator.Animate(
                        pred, (info.FramesLeftTruncate + f) * info.TotalDim + info.SkinOffset,
                        all, f * size, dt);
                skin.ComputeMaskedMean(all, 0, size, info.FramesCenter, bias);

                var n = skin.NeutralMasked;
                for (int i = 0; i < bias.Length && i < n.Length; i++) bias[i] -= n[i];
            }
            else
            {
                skin.ComputeMaskedMean(pred,
                    info.FramesLeftTruncate * info.TotalDim + info.SkinOffset,
                    info.TotalDim, info.FramesCenter, bias);
            }

            double s = 0.0;
            for (int i = 0; i < bias.Length; i++) s += (double)bias[i] * bias[i];
            return (float)Math.Sqrt(s / Mathf.Max(1, bias.Length));
        }

        /// <summary>
        /// 同一帧、同一个求解器，比较「过 animator」和「不过 animator」两条路。
        /// 差别应该集中在嘴部：下半脸 ×lowerFaceStrength(1.3)，外加 eyeClose/lipOpen 两个静态 pose。
        /// </summary>
        private void ReportAnimatorEffect()
        {
            if (!_pipeline.UsingSkinAnimator)
            {
                Debug.Log("[A2F Probe] skin animator 未启用，跳过开/关对比");
                return;
            }

            var solver = _pipeline.SkinSolver;
            var info = _pipeline.Info;
            var pred = _pipeline.Prediction;
            int row = info.FramesLeftTruncate * info.TotalDim + info.SkinOffset;
            float dt = 1f / info.FrameRate;

            var verts = _pipeline.SkinVertices;
            _pipeline.SkinAnimator.Reset();
            _pipeline.SkinAnimator.Animate(pred, row, verts, 0, dt);

            solver.Reset();
            solver.TargetIsDelta = false;
            var on = (float[])solver.Solve(verts, 0).Clone();
            float resOn = solver.ComputeRelativeResidual();
            float rmsOn = solver.LastTargetRms;

            // 不过 animator：prediction 直接当 delta
            var raw = new float[info.SkinSize];
            Array.Copy(pred, row, raw, 0, info.SkinSize);
            solver.Reset();
            solver.TargetIsDelta = true;
            var off = (float[])solver.Solve(raw, 0).Clone();
            float resOff = solver.ComputeRelativeResidual();
            float rmsOff = solver.LastTargetRms;

            // 恢复管线默认设置
            solver.TargetIsDelta = false;
            solver.Reset();

            int jaw = IndexOfPose("jawOpen");
            int close = IndexOfPose("mouthClose");
            var sb = new StringBuilder();
            sb.AppendLine("[A2F Probe] === skin animator 开/关对比（同一帧、同一求解器）===");
            sb.AppendLine($"    OFF: target RMS={rmsOff:F4} 残差={resOff:F4} " +
                          $"jawOpen={off[jaw]:F4} mouthClose={off[close]:F4}");
            sb.AppendLine($"    ON : target RMS={rmsOn:F4} 残差={resOn:F4} " +
                          $"jawOpen={on[jaw]:F4} mouthClose={on[close]:F4}");
            sb.AppendLine($"    → 下半脸 ×{_pipeline.SkinAnimator.CurrentParams.lowerFaceStrength:F2}，" +
                          $"嘴部权重与 target 幅度应明显变大");
            Debug.Log(sb.ToString());
        }

        private int IndexOfPose(string name)
        {
            var names = _pipeline.SkinSolver.PoseNames;
            if (names == null) return 0;
            for (int i = 0; i < _pipeline.SkinPoseCount && i < names.Length; i++)
                if (names[i] == name) return i;
            return 0;
        }

        private static void ReportSegment(string name, float[] pred, int offset, int size, int totalDim)
        {
            int frames = pred.Length / totalDim;
            float min = float.MaxValue, max = float.MinValue;
            double sumAbs = 0.0;
            long count = 0;
            for (int f = 0; f < frames; f++)
            {
                int row = f * totalDim + offset;
                for (int i = 0; i < size; i++)
                {
                    float v = pred[row + i];
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sumAbs += Math.Abs(v);
                    count++;
                }
            }
            float meanAbs = count > 0 ? (float)(sumAbs / count) : 0f;
            Debug.Log($"[A2F Probe] {name,-7} min={min:F3} max={max:F3} mean|v|={meanAbs:F4}");
        }

        void OnDestroy()
        {
            _pipeline?.Dispose();
            _pipeline = null;
        }
    }
}
