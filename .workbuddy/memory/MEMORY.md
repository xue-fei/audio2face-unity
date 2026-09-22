# audio2face-unity 项目长期笔记

## A2F 帧/姿序不变量（改代码前必读）
- frame[] 布局：skin(52)@0 + tongue(16)@52 + jaw(7)@68 + eyes(4)@75。skin 下标语义 = `SkinSolver.PoseNames`（NVIDIA 自定义序，**≠** `ArkItBlendshapeNames.All52` 字母序）。任何按名字取 frame 系数的代码都必须用 `SkinSolver.PoseNames` 建索引。
- `Audio2FaceBlendshapeTester` 订阅 OnFrameApplied，是 Target 网格的**最终写入者**（晚于 Component.ApplyFrame 写 _bindings）。它经 `EnsureMap()` 保证 _poses 与 frame 同序。bug 指纹：「手动扫掠正常、播放并驱动乱」= 姿序错位。
- pipeline 在 Component.Start() 才 new → Awake 里读 `Driver.PoseNames` 恒为 null（Unity 所有 Awake 先于所有 Start）。跨组件拿 pose 表必须延后到 Start 之后或惰性重建（EnsureMap 模式）。

## Config 迁移规则
- 改 `Audio2FaceDiffusionConfig` 任何默认值：`LatestConfigVersion` +1 并在 Migrate() 里**无条件覆盖**（不看旧值合法性）。场景序列化会锁死旧默认值，改代码默认值不会生效。

## 探针 / 日志约定
- `[A2F实测]`（BakeMesh 量真实唇缝）是「嘴张没张」的终局证据；静息基准 ≈1.19% 网格高，说话上限应到 2~6%。`mesh.vertices` 在无 Read/Write 时返回空数组不抛异常——量顶点一律用 BakeMesh / GetBlendShapeFrameVertices。
- 唇缝探针用 v4（逐 pose 打满烘 BakeMesh 量质心距离系数），别退回投影/符号法（v1~v3 全错）。
- `[A2F写入]`（Component）与 `[A2F驱动]`（Tester）两条心跳的 jawOpen 峰值必须对账一致；不一致 = 姿序又不同步。
