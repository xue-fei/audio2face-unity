using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace Audio2Face
{
    /// <summary>
    /// ARKit blendshape 名称归一化 + 网格名 → 求解器 pose 名 的映射。
    ///
    /// 为什么需要它：求解器输出的 pose 名是标准 ARKit 名（jawOpen / eyeBlinkLeft / mouthSmileRight ...），
    /// 而美术资产几乎从不这么命名。本项目用的 sloth 头就是典型案例，50 个 blendshape 全部长这样：
    ///
    ///     blendShape2.jawOpen
    ///     blendShape2.eyeBlink_L          ← 有 "blendShape2." 前缀
    ///     blendShape2.mouthSmile_R        ← 用 _L/_R 而不是 Left/Right
    ///     blendShape2.noseSneer_L
    ///
    /// 老的映射只做「全名相等 / 忽略大小写 / 取最后一个 '.' 之后」三步，
    /// 取到 eyeBlink_L 之后仍然对不上 eyeBlinkLeft → <b>0/52 全错过，网格一动不动</b>。
    ///
    /// 这里改成「双向归一化」：把网格名剥前缀、把 _L/_R 展开成 Left/Right，
    /// 再拿结果去查 ARKit 52 白名单。<b>只有落入白名单才认</b>，
    /// 所以 mouthFunnel 这种「末尾也带 l」的名字不会被误判成 mouthFunnelLeft。
    /// </summary>
    public static class ArkItBlendshapeNames
    {
        /// <summary>标准 ARKit 52 名称（Audio2Face 求解器使用的规范名）。</summary>
        public static readonly string[] All52 =
        {
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
            "eyeBlinkLeft", "eyeBlinkRight",
            "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
            "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
            "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
            "jawForward", "jawLeft", "jawOpen", "jawRight",
            "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
            "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
            "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
            "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
            "mouthUpperUpLeft", "mouthUpperUpRight",
            "noseSneerLeft", "noseSneerRight", "tongueOut",
        };

        private static readonly Dictionary<string, string> Lookup = BuildLookup();

        private static Dictionary<string, string> BuildLookup()
        {
            var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var n in All52) d[n] = n;
            return d;
        }

        /// <summary>规范名 → 是否属于 ARKit 52。</summary>
        public static bool IsArkitName(string name) => !string.IsNullOrEmpty(name) && Lookup.ContainsKey(name);

        /// <summary>
        /// 把网格上的任意命名归一到规范 ARKit 名。识别不出返回 null。
        /// 例：<c>blendShape2.mouthSmile_R</c> → <c>mouthSmileRight</c>。
        /// </summary>
        public static string Canonicalize(string rawName)
        {
            if (string.IsNullOrEmpty(rawName)) return null;

            string s = rawName.Trim();

            // 1) 直接命中（顺带覆盖大小写差异，如 arkit 的 mouthSmileLeft）
            if (Lookup.TryGetValue(s, out string direct)) return direct;

            // 2) 剥掉前缀，如 "blendShape2." / "blendShape1." / "mesh."
            int dot = s.LastIndexOf('.');
            if (dot >= 0 && dot + 1 < s.Length) s = s.Substring(dot + 1);
            if (Lookup.TryGetValue(s, out string stripped)) return stripped;

            // 3) 侧向标签展开：_L/-L/" L" → Left；_R/-R/" R" → Right
            foreach (string candidate in SideVariants(s, true))
                if (Lookup.TryGetValue(candidate, out string hit)) return hit;

            // 4) 兜底：末尾就是一个裸字母 l/r（无分隔符），同样要求展开后能命中白名单。
            //    这一步是安全的：mouthFunnel 展开成 mouthFunnellLeft 不在白名单里，会被拒绝。
            foreach (string candidate in SideVariants(s, false))
                if (Lookup.TryGetValue(candidate, out string hit)) return hit;

            return null;
        }

        private static IEnumerable<string> SideVariants(string s, bool requireSeparator)
        {
            if (s.Length < 2) yield break;

            char last = s[s.Length - 1];
            bool isL = last == 'L' || last == 'l';
            bool isR = last == 'R' || last == 'r';
            if (!isL && !isR) yield break;

            char sep = s[s.Length - 2];
            bool hasSep = sep == '_' || sep == '-' || sep == ' ' || sep == '.';
            if (requireSeparator && !hasSep) yield break;

            string body = s.Substring(0, s.Length - 2);
            if (requireSeparator) yield return body + (isL ? "Left" : "Right");

            if (!requireSeparator)
            {
                string body1 = s.Substring(0, s.Length - 1);
                yield return body1 + (isL ? "Left" : "Right");
            }
        }

        /// <summary>网格实际 blendshape 名 → 规范名（识别不出的原样返回并标注）。</summary>
        public static string[] DescribeMesh(Mesh mesh, out string[] canonicals)
        {
            int n = mesh != null ? mesh.blendShapeCount : 0;
            var raw = new string[n];
            canonicals = new string[n];
            for (int i = 0; i < n; i++)
            {
                raw[i] = mesh.GetBlendShapeName(i);
                canonicals[i] = Canonicalize(raw[i]);
            }
            return raw;
        }

        /// <summary>
        /// 把求解器的 pose 名映射到网格 blendshape 下标（找不到 = -1）。
        /// </summary>
        /// <param name="mesh">目标网格。</param>
        /// <param name="poseNames">求解器 pose 名（ARKit 52 或舌头 pose）。</param>
        /// <param name="fallbackToIndex">匹配失败时是否退化为「同名下标」。</param>
        /// <param name="matched">匹配上的数量。</param>
        /// <param name="report">人类可读的映射报告（逐条 + 未用到的形状 + 未认出的命名）。</param>
        public static int[] MapPoses(Mesh mesh, string[] poseNames, bool fallbackToIndex,
                                     out int matched, out string report)
        {
            matched = 0;
            if (mesh == null || poseNames == null)
            {
                report = "[A2F Map] 网格或 pose 名为空";
                return poseNames != null ? NewFilled(poseNames.Length, -1) : new int[0];
            }

            int count = mesh.blendShapeCount;
            var meshNames = new string[count];
            var meshCanon = new string[count];
            for (int j = 0; j < count; j++)
            {
                meshNames[j] = mesh.GetBlendShapeName(j);
                meshCanon[j] = Canonicalize(meshNames[j]);
            }
            // 规范名 → 网格下标（后出现的同名不覆盖先出现的）
            var byCanon = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int j = 0; j < count; j++)
                if (meshCanon[j] != null && !byCanon.ContainsKey(meshCanon[j]))
                    byCanon[meshCanon[j]] = j;

            var map = NewFilled(poseNames.Length, -1);
            var sb = new StringBuilder();
            var used = new HashSet<int>();

            for (int i = 0; i < poseNames.Length; i++)
            {
                string pose = poseNames[i];
                string canon = Canonicalize(pose) ?? pose;

                int idx = -1;
                if (byCanon.TryGetValue(canon, out int exact)) idx = exact;

                if (idx < 0 && fallbackToIndex && i < count) idx = i;

                map[i] = idx;
                if (idx >= 0)
                {
                    matched++;
                    used.Add(idx);
                }
            }

            sb.AppendLine($"[A2F Map] 匹配 {matched}/{poseNames.Length}（网格共 {count} 个 blendshape）");
            sb.AppendLine("  pose → 网格");
            for (int i = 0; i < poseNames.Length; i++)
            {
                if (map[i] >= 0)
                    sb.AppendLine($"    {poseNames[i],-22} → #{map[i]:D2}  {meshNames[map[i]]}");
                else
                    sb.AppendLine($"    {poseNames[i],-22} → ✗ 网格无此形状（该通道不会动，属正常）");
            }

            // 未被使用的网格形状（可能是舌头/自定义，或被漏掉的命名）
            var unused = new List<string>();
            for (int j = 0; j < count; j++)
                if (!used.Contains(j)) unused.Add($"#{j}:{meshNames[j]}");
            if (unused.Count > 0)
                sb.AppendLine($"  网格里未被用到的形状 {unused.Count} 个: {string.Join(", ", unused)}");

            // 认不出命名的形状（归一化失败），提示怎么改
            var unknown = new List<string>();
            for (int j = 0; j < count; j++)
                if (meshCanon[j] == null) unknown.Add(meshNames[j]);
            if (unknown.Count > 0)
                sb.AppendLine($"  ⚠ 命名无法识别 {unknown.Count} 个（既不是 ARKit 名，也剥不出 _L/_R）: " +
                              string.Join(", ", unknown));

            report = sb.ToString();
            return map;
        }

        private static int[] NewFilled(int n, int v)
        {
            var a = new int[n];
            for (int i = 0; i < n; i++) a[i] = v;
            return a;
        }
    }
}
