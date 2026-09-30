using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading;
/// <summary>
/// 多线程工具：后台线程干重活，结果回主线程落地（UnityEngine 的 API 只能在主线程调）。
/// </summary>
public class Loom : MonoBehaviour
{
    public static int maxThreads = 8;
    static int numThreads;

    private static Loom _current;
    public static Loom Current
    {
        get
        {
            Initialize();
            return _current;
        }
    }

    static bool initialized;

    /// <summary>
    /// 初始化：建一个常驻 GameObject 用 Update 泵送主线程队列。
    /// 必须在主线程调用过一次，否则 QueueOnMainThread 排的回调没人执行。
    /// </summary>
    public static void Initialize()
    {
        if (!initialized)
        {
            initialized = true;
            GameObject g = new GameObject("Loom");
            DontDestroyOnLoad(g);
            _current = g.AddComponent<Loom>();
        }
    }

    private List<Action> _actions = new List<Action>();
    public struct DelayedQueueItem
    {
        public float time;
        public Action action;
    }
    private List<DelayedQueueItem> _delayed = new List<DelayedQueueItem>();

    List<DelayedQueueItem> _currentDelayed = new List<DelayedQueueItem>();

    /// <summary>在主线程中运行。</summary>
    public static void QueueOnMainThread(Action action)
    {
        QueueOnMainThread(action, 0f);
    }
    public static void QueueOnMainThread(Action action, float time)
    {
        if (time != 0)
        {
            if (Current != null)
            {
                lock (Current._delayed)
                {
                    Current._delayed.Add(new DelayedQueueItem { time = Time.time + time, action = action });
                }
            }
        }
        else
        {
            if (Current != null)
            {
                lock (Current._actions)
                {
                    Current._actions.Add(action);
                }
            }
        }
    }

    /// <summary>
    /// 在后台线程执行。
    /// </summary>
    /// <param name="a">重活，里面别碰 UnityEngine API（Debug.Log 可以，它是线程安全的）。</param>
    /// <param name="onError">出错时在主线程回调；不传就只打一条 LogError。</param>
    public static void RunAsync(Action a, Action<Exception> onError = null)
    {
        Initialize();
        Interlocked.Increment(ref numThreads);
        ThreadPool.QueueUserWorkItem(_ => RunAction(a, onError));
    }

    private static void RunAction(Action a, Action<Exception> onError)
    {
        try
        {
            a();
        }
        catch (Exception ex)
        {
            // ⚠ 原来是 catch {} 直接吞掉：后台线程一抛错，完成回调就永不执行，
            // 表现为「一直在加载」且没有任何报错，排查极费劲。这里必须把异常送回主线程。
            if (onError != null)
                QueueOnMainThread(() => onError(ex));
            else
                QueueOnMainThread(() => Debug.LogError($"[Loom] 后台线程异常：\n{ex}"));
        }
        finally
        {
            Interlocked.Decrement(ref numThreads);
        }
    }

    void OnDisable()
    {
        if (_current == this)
        {
            _current = null;
        }
    }

    List<Action> _currentActions = new List<Action>();

    void Update()
    {
        lock (_actions)
        {
            _currentActions.Clear();
            _currentActions.AddRange(_actions);
            _actions.Clear();
        }
        foreach (var a in _currentActions)
        {
            a();
        }
        lock (_delayed)
        {
            _currentDelayed.Clear();
            _currentDelayed.AddRange(_delayed.FindAll(d => d.time <= Time.time));
            foreach (var item in _currentDelayed)
                _delayed.Remove(item);
        }
        foreach (var delayed in _currentDelayed)
        {
            delayed.action();
        }
    }
}
