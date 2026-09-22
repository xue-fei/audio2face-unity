using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace Audio2Face
{
    /// <summary>
    /// .npz (zip of .npy) 读取器。
    /// 不依赖 numpy，纯 C# 解析 .npy 头 + 小端原始数据。
    /// NVIDIA Audio2Face 的 bs_skin_*.npz / bs_tongue_*.npz / model_data_*.npz 都是这种格式。
    /// </summary>
    public sealed class NpzFile : IDisposable
    {
        private readonly Dictionary<string, byte[]> _entries = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        private readonly Dictionary<string, int[]> _shapes = new Dictionary<string, int[]>(StringComparer.Ordinal);
        private readonly Dictionary<string, string> _dtypes = new Dictionary<string, string>(StringComparer.Ordinal);

        public static NpzFile Load(string path)
        {
            var file = new NpzFile();
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var zip = new ZipArchive(fs, ZipArchiveMode.Read))
            {
                foreach (var entry in zip.Entries)
                {
                    if (!entry.FullName.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                        continue;

                    var name = Path.GetFileNameWithoutExtension(entry.FullName);
                    byte[] raw;
                    using (var src = entry.Open())
                    using (var ms = new MemoryStream())
                    {
                        src.CopyTo(ms);
                        raw = ms.ToArray();
                    }

                    int dataOffset;
                    var dtype = ParseNpyHeader(raw, out int[] shape, out dataOffset);
                    if (dtype == null)
                        continue;

                    var payload = new byte[raw.Length - dataOffset];
                    Buffer.BlockCopy(raw, dataOffset, payload, 0, payload.Length);

                    file._entries[name] = payload;
                    file._shapes[name] = shape;
                    file._dtypes[name] = dtype;
                }
            }
            return file;
        }

        private static string ParseNpyHeader(byte[] raw, out int[] shape, out int dataOffset)
        {
            shape = new int[0];
            dataOffset = 0;
            if (raw.Length < 10) return null;
            if (raw[0] != 0x93 || raw[1] != (byte)'N' || raw[2] != (byte)'U' || raw[3] != (byte)'M' ||
                raw[4] != (byte)'P' || raw[5] != (byte)'Y')
            {
                return null;
            }

            int headerLen;
            int headerStart;
            if (raw[6] == 1)
            {
                headerLen = raw[8] | (raw[9] << 8);
                headerStart = 10;
            }
            else
            {
                headerLen = raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24);
                headerStart = 12;
            }

            var header = Encoding.UTF8.GetString(raw, headerStart, headerLen);

            var m = Regex.Match(header, "'descr'\\s*:\\s*'([^']+)'");
            var dtype = m.Success ? m.Groups[1].Value : null;

            m = Regex.Match(header, "'shape'\\s*:\\s*\\(([^)]*)\\)");
            if (m.Success)
            {
                var parts = m.Groups[1].Value.Split(',');
                var list = new List<int>();
                foreach (var p in parts)
                {
                    var t = p.Trim();
                    if (t.Length == 0) continue;
                    int v;
                    if (int.TryParse(t, out v)) list.Add(v);
                }
                shape = list.ToArray();
            }

            dataOffset = headerStart + headerLen;
            return dtype;
        }

        public bool Has(string name) => _entries.ContainsKey(name);

        public int[] GetShape(string name)
        {
            int[] s;
            return _shapes.TryGetValue(name, out s) ? s : new int[0];
        }

        /// <summary>读取 float32 数组（&lt;f4）。</summary>
        public float[] GetFloats(string name)
        {
            byte[] payload;
            if (!_entries.TryGetValue(name, out payload)) return null;

            int count = payload.Length / 4;
            var result = new float[count];
            Buffer.BlockCopy(payload, 0, result, 0, count * 4);
            if (!BitConverter.IsLittleEndian)
            {
                for (int i = 0; i < count; i++)
                {
                    var bytes = BitConverter.GetBytes(result[i]);
                    Array.Reverse(bytes);
                    result[i] = BitConverter.ToSingle(bytes, 0);
                }
            }
            return result;
        }

        /// <summary>读取 int32 数组（&lt;i4）。</summary>
        public int[] GetInts(string name)
        {
            byte[] payload;
            if (!_entries.TryGetValue(name, out payload)) return null;

            int count = payload.Length / 4;
            var result = new int[count];
            Buffer.BlockCopy(payload, 0, result, 0, count * 4);
            if (!BitConverter.IsLittleEndian)
            {
                for (int i = 0; i < count; i++)
                {
                    var bytes = BitConverter.GetBytes(result[i]);
                    Array.Reverse(bytes);
                    result[i] = BitConverter.ToInt32(bytes, 0);
                }
            }
            return result;
        }

        /// <summary>读取定长 ASCII 字符串数组（|S19 之类）。</summary>
        public string[] GetStrings(string name)
        {
            byte[] payload;
            if (!_entries.TryGetValue(name, out payload)) return null;

            var shape = GetShape(name);
            int count = shape.Length > 0 ? shape[0] : 0;
            if (count <= 0) return new string[0];
            int width = payload.Length / count;

            var result = new string[count];
            for (int i = 0; i < count; i++)
            {
                int end = i * width;
                for (int j = width - 1; j >= 0; j--)
                {
                    if (payload[i * width + j] != 0)
                    {
                        end = i * width + j + 1;
                        break;
                    }
                }
                result[i] = Encoding.ASCII.GetString(payload, i * width, end - i * width).Trim();
            }
            return result;
        }

        public void Dispose()
        {
            _entries.Clear();
            _shapes.Clear();
            _dtypes.Clear();
        }
    }
}
