# audio2face-unity 项目长期笔记

## A2F 帧/姿序不变量（改代码前必读）
- frame[] 布局：skin(52)@0 + tongue(16)@52 + jaw(7)@68 + eyes(4)@75。skin 下标语义 = `SkinSolver.PoseNames`（NVIDIA 自定义序，**≠** `ArkItBlendshapeNames.All52` 字母序）。任何按名字取 frame 系数的代码都必须用 `SkinSolver.PoseNames` 建索引。
- `Audio2FaceBlendshapeTester` 订阅 OnFrameApplied，是 Target 网格的**最终写入者**（晚于 Component.ApplyFrame 写 _bindings）。它经 `EnsureMap()` 保证 _poses 与 frame 同序。bug 指纹：「手动扫掠正常、播放并驱动乱」= 姿序错位。
- pipeline 在 Component.Start() 才 new → Awake 里读 `Driver.PoseNames` 恒为 null（Unity 所有 Awake 先于所有 Start）。跨组件拿 pose 表必须延后到 Start 之后或惰性重建（EnsureMap 模式）。

## ONNX 推理后端规则
- `onnxruntime-cpu` 与 `onnxruntime-cuda` **只能装一个**：两者都带 `Plugins/Win/onnxruntime.dll`，Windows 按基名先加载者胜（CPU 构建先加载则 `MakeSessionOptionWithCudaProvider` 抛异常 → 静默回退 CPU）。现在只留 managed `onnxruntime` + `onnxruntime-cuda`。判据：启动日志 `★ 推理后端=` + `[A2F推理]` 单次耗时。
- **耗时构成（GTX 1070，实测，第三轮优化后）**：每次推理 = 0.5s 音频 / 30 动画帧。`ONNX(CUDA)≈56ms` + 后处理 ≈130ms。后处理内部（优化后 ms/帧）：SkinAnimator 0.46（掩码 24002→4262 顶点）、掩码取样 0.11、**D^T·target 1.52（已升为最大单项）**、BVLS 1.03；tongue 另 1.18。合计 4.3ms/帧 → RTF 0.74 → **0.372**。
- BVLS 收敛判据是 `max|g|`（g=Aᵀ(Ax−b)），**必须用相对容差**：SDK 的绝对 1e-10 对 1e7 量级的梯度永远达不到。现阈值 = `max(绝对, RelativeTolerance×|A·b|)`，`BlendshapeSolver.BvlsRelativeTolerance`（默认 1e-6，设 0 = 旧行为）。⚠ 但**实测放宽容差几乎不省时间**（18.4→17.4 次迭代）：主循环出口是 `VarToDetach==-1`（KKT 真正满足）而非阈值，active-set 算法一次迭代只放开一个边界变量，43 维要 ~17 次是固有成本。别再指望调容差提速。
- 想砍 BVLS 应走另一条路：**`cancelPairs` 导致的每帧第二次完整 Solve**（占 BVLS 约一半）。若所有 cancel 对的 loser 当前值已 ≈0，第二次解与第一次结果相同 → 可跳过。
- `SolveVerified`（一次性）跑三遍：生产路径 / 强制 QR / 阈值压 0，分别证明「Cholesky≡QR」和「放宽容差没掉精度」，第二行日志 >1e-3 打 ⚠。
- 两个已修的坑：① `[解算耗时]` 早期版本只在第 300 帧测一次却除以 300 → 打印值要 ×300 才是真的；② `Model.Run()` 每 10 次会全量扫 533 万 float 做诊断统计 → ONNX 段周期性从 58ms 飙到 305ms，现已改成 `DiagSampleStep=16` 采样。
- 剖析日志有三条：`[A2F推理] ONNX/后处理/合计`、`[Audio2Face] 第N次推理完成 ONNX/解算30帧`、`[解算耗时] 掩码取样+D^T+BVLS` + `[解算剖析] animator/solver`。
- 读计时日志的坑：`WorkerLoop` 的 `_lastInferenceMs` 覆盖 `Run+SolveCenterFrames`，而 warmup 帧（`_windowEnd < BufferLength`）会整段跳过 30 帧解算 → **那几条 62ms 就是纯 ONNX 时间**，可用来推断拆分。
- **剖析必须每帧都 Restart+累加、只在周期末尾打印并清零**；只在第 N 帧测一次却除以 N 会打出小 N 倍的假数字（旧日志要 ×300）。

## CPU 解算优化（第十六轮已落地，改前先看）
- **animator 掩码视图是最大收益项**：solver 只按 frontalMask 读 4262/24002 顶点，而插值状态**逐分量独立**（l1[i] 只依赖 l1[i]/l2[i]）→ `AnimateMasked` 只重建掩码顶点，输出紧凑数组喂 `SolveMasked`。faceMask 的 Y 归一化必须沿用**全量** neutral 的 minY/maxY，否则 mask 值整体偏移。
- Interpolator degree=2 的第 0 层（raw）只有寄存器价值，`_loL1/_loL2/_upL1/_upL2` 两层即可。
- **BVLS 用正规方程 + Cholesky**：A 对称 → AᵀA = A·A = G 可缓存，A·b 每 Solve 算一次；每次 ConstrainedMin 只剩 rhs_F=(Ab)_F − G_{F,C}·x_C + Cholesky(G_FF)，比 Householder QR（O(2np²)+拷贝 A(:,F)）快约 10×。代价 cond(G)=cond(A)² → 全程 double，不收敛自动回退 QR。开关 `BlendshapeSolver.UseBvlsNormalEquations`。
- BVLS 热路径**禁止 new**：原实现每次 ConstrainedMin new 五六个数组（一帧几十次 × 每帧两遍 Solve），全部走 `BvlsWorkspace` 预分配。
- 语义红线：BVLS 解的是 `min|AMat·x − b|²`（A 被"平方"过），**不是**标准 QP `min 0.5xᵀAx − bᵀx`。两者 KKT 符号条件不同，改成 QP 会改变结果，不要"顺手修正"。

## Config 迁移规则
- 改 `Audio2FaceDiffusionConfig` 任何默认值：`LatestConfigVersion` +1 并在 Migrate() 里**无条件覆盖**（不看旧值合法性）。场景序列化会锁死旧默认值，改代码默认值不会生效。（**新增**字段不受此限，Unity 会用字段初始值。）

## 推流节奏（正确性相关，不只是性能）
- `ProcessClipRoutine` 阶段 3 每帧 push 4096 样本 = **15 倍实时**。只靠 ring 容量背压的话，推理线程永远有活干 → 整段音频被提前算完，帧队列单调涨（实测 30→385，22s 音频最终堆 1300+ 帧）；离线播 clip 只是"提前算完"，**实时输入会持续累积延迟**。
- 现已按播放时钟节流（config `maxPushAheadSec`），节流后生产速率（30 帧 / 0.5s 音频）== 消费速率（60 帧/s），队列收敛。
- **致命坑**：lead 必须取 `max(config, 开播那一刻的实际领先量)`。预缓冲为了攒 prebufferFrames 帧必然已多推 ~2s 音频，直接用配置值会在开播瞬间干等、队列被抽干、脸僵住。只保证不增长，不回压。

## 探针 / 日志约定
- `[A2F实测]`（BakeMesh 量真实唇缝）是「嘴张没张」的终局证据；静息基准 ≈1.19% 网格高，说话上限应到 2~6%。`mesh.vertices` 在无 Read/Write 时返回空数组不抛异常——量顶点一律用 BakeMesh / GetBlendShapeFrameVertices。
- 唇缝探针用 v4（逐 pose 打满烘 BakeMesh 量质心距离系数），别退回投影/符号法（v1~v3 全错）。
- `[解算耗时] n=.. k=..: 掩码取样 + D^T·target + BVLS`（每 300 帧一条）是解算侧的分段计时；末尾会标明点积走的是 `SIMD Vector<float>×N` 还是 `4 路展开 float`（后者 = Unity Mono 没加速 Vector<T>）。`BlendshapeSolver.UseSimdDot=false` 可强制走展开版对拍。
- `[A2F写入]`（Component）与 `[A2F驱动]`（Tester）两条心跳的 jawOpen 峰值必须对账一致；不一致 = 姿序又不同步。
