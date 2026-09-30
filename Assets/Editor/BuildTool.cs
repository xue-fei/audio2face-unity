using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public class BuildTool : Editor
{
    static string buildPath;

    [MenuItem("Tools/打包Win")]
    public static void BuildWin()
    {
        buildPath = Application.dataPath + "/../Build/";
        if (!Directory.Exists(buildPath))
        {
            Directory.CreateDirectory(buildPath);
        }

        BuildReport br = BuildPipeline.BuildPlayer(new[]
            {
                "Assets/Scenes/audio2face.unity",
            },
            buildPath + "audio2face.exe", BuildTarget.StandaloneWindows64, BuildOptions.None);
        if (br.GetFiles().Length < 0)
        {
            throw new Exception("BuildPlayer failure: " + br.strippingInfo);
        }
        else
        {
            System.Diagnostics.Process.Start(buildPath);
        }
    }

    [MenuItem("Tools/打包Linux")]
    public static void BuildLinux()
    {
        buildPath = Application.dataPath + "/../Build/";
        if (!Directory.Exists(buildPath))
        {
            Directory.CreateDirectory(buildPath);
        }
        BuildReport br = BuildPipeline.BuildPlayer(new[]
            {
                "Assets/Scenes/audio2face.unity",
            },
            buildPath + "audio2face", BuildTarget.StandaloneLinux64, BuildOptions.None);
        if (br.GetFiles().Length < 0)
        {
            throw new Exception("BuildPlayer failure: " + br.strippingInfo);
        }
        else
        {
            System.Diagnostics.Process.Start(buildPath);
        }
    }
}