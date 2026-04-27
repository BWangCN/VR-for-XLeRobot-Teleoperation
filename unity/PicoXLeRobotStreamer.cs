using System;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using LitJson;
using Unity.XR.PXR;
using UnityEngine;
using UnityEngine.XR;
using CommonUsages = UnityEngine.XR.CommonUsages;
using InputDevice = UnityEngine.XR.InputDevice;

/// <summary>
/// PicoXLeRobotStreamer - Streams PICO 4 Ultra controller data to XLeRobot simulation.
///
/// Sends simple JSON via TCP to the PC running the Python bridge server.
/// Protocol: newline-delimited JSON on port 9876.
///
/// Setup:
///   1. Copy this script to your Unity PICO project Assets/Scripts/
///   2. Attach to a GameObject in your scene
///   3. Set pcIpAddress to your PC's IP
///   4. Run the Python bridge: python demo_pico_vr_sim.py
///   5. Run the Unity app on PICO
/// </summary>
public class PicoXLeRobotStreamer : MonoBehaviour
{
    [Header("PC Connection")]
    [Tooltip("IP address of the PC running the XLeRobot simulation")]
    public string pcIpAddress = "192.168.1.161";

    [Tooltip("TCP port (must match Python bridge)")]
    public int port = 9876;

    [Header("Streaming")]
    [Tooltip("Send rate in Hz")]
    public float sendRate = 60f;

    [Tooltip("Auto-reconnect on disconnect")]
    public bool autoReconnect = true;

    // Internal state
    private Socket _socket;
    private Thread _sendThread;
    private ConcurrentQueue<string> _sendQueue = new ConcurrentQueue<string>();
    private volatile bool _isConnected = false;
    private volatile bool _isRunning = true;
    private float _lastSendTime = 0f;
    private float _lastReconnectTime = 0f;

    // Reusable JSON objects to reduce GC
    private JsonData _msgJson = new JsonData();
    private JsonData _leftJson = new JsonData();
    private JsonData _rightJson = new JsonData();
    private JsonData _headJson = new JsonData();

    private void Start()
    {
        // Enable video passthrough (see real world through headset)
        EnablePassthrough();

        _sendThread = new Thread(SendThreadLoop) { IsBackground = true };
        _sendThread.Start();
        Connect();
    }

    private void EnablePassthrough()
    {
        // Make camera background transparent so passthrough is visible
        Camera cam = Camera.main;
        if (cam != null)
        {
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.clear;
        }

        // PICO SDK 3.3.1: PXR_Manager static property
        try
        {
            PXR_Manager.EnableVideoSeeThrough = true;
            Debug.Log("[XLeRobot] Passthrough enabled");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[XLeRobot] Passthrough failed: {e.Message}");
        }
    }

    private void Connect()
    {
        try
        {
            Debug.Log($"[XLeRobot] Connecting to {pcIpAddress}:{port}...");
            _socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            _socket.NoDelay = true;
            _socket.SendTimeout = 5000;
            _socket.BeginConnect(pcIpAddress, port, OnConnected, _socket);
        }
        catch (Exception e)
        {
            Debug.LogError($"[XLeRobot] Connect failed: {e.Message}");
        }
    }

    private void OnConnected(IAsyncResult ar)
    {
        try
        {
            Socket sock = (Socket)ar.AsyncState;
            sock.EndConnect(ar);          // must call BEFORE checking Connected
            if (sock.Connected)
            {
                _isConnected = true;
                Debug.Log($"[XLeRobot] Connected to PC at {pcIpAddress}:{port}");
            }
            else
            {
                Debug.LogError("[XLeRobot] Connection failed");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[XLeRobot] Connection error: {e.Message}");
        }
    }

    private void Update()
    {
        // Auto-reconnect
        if (!_isConnected && autoReconnect)
        {
            if (Time.time - _lastReconnectTime > 3f)
            {
                _lastReconnectTime = Time.time;
                Connect();
            }
            return;
        }

        // Rate limiting
        float interval = 1f / sendRate;
        if (Time.time - _lastSendTime < interval)
            return;
        _lastSendTime = Time.time;

        // Only queue if previous message was consumed
        if (_sendQueue.Count > 1)
            return;

        string json = BuildControllerJson();
        if (json != null)
        {
            _sendQueue.Enqueue(json + "\n");
        }
    }

    private string BuildControllerJson()
    {
        // Predict time for lower-latency controller pose.
        // PXR_System is in the base SDK; no Enterprise module needed.
        double predictTime = 0;
        try
        {
            predictTime = PXR_System.GetPredictedDisplayTime();
        }
        catch { }

        // ---- Left Controller ----
        Vector3 leftPos = PXR_Input.GetControllerPredictPosition(
            PXR_Input.Controller.LeftController, predictTime);
        Quaternion leftRot = PXR_Input.GetControllerPredictRotation(
            PXR_Input.Controller.LeftController, predictTime);

        InputDevice leftDevice = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        leftDevice.TryGetFeatureValue(CommonUsages.trigger, out float leftTrigger);
        leftDevice.TryGetFeatureValue(CommonUsages.grip, out float leftGrip);
        leftDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 leftThumb);
        leftDevice.TryGetFeatureValue(CommonUsages.primaryButton, out bool leftPrimary);
        leftDevice.TryGetFeatureValue(CommonUsages.secondaryButton, out bool leftSecondary);

        // ---- Right Controller ----
        Vector3 rightPos = PXR_Input.GetControllerPredictPosition(
            PXR_Input.Controller.RightController, predictTime);
        Quaternion rightRot = PXR_Input.GetControllerPredictRotation(
            PXR_Input.Controller.RightController, predictTime);

        InputDevice rightDevice = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        rightDevice.TryGetFeatureValue(CommonUsages.trigger, out float rightTrigger);
        rightDevice.TryGetFeatureValue(CommonUsages.grip, out float rightGrip);
        rightDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 rightThumb);
        rightDevice.TryGetFeatureValue(CommonUsages.primaryButton, out bool rightPrimary);
        rightDevice.TryGetFeatureValue(CommonUsages.secondaryButton, out bool rightSecondary);

        // ---- Head (HMD) ----
        PxrSensorState2 sensor = new PxrSensorState2();
        int frameIdx = 0;
        PXR_System.GetPredictedMainSensorStateNew(ref sensor, ref frameIdx);
        Vector3 headPos = new Vector3(
            sensor.pose.position.x,
            sensor.pose.position.y,
            sensor.pose.position.z);
        Quaternion headRot = new Quaternion(
            sensor.pose.orientation.x,
            sensor.pose.orientation.y,
            sensor.pose.orientation.z,
            sensor.pose.orientation.w);

        // ---- Build JSON ----
        // Using string builder for performance (avoid LitJson GC overhead at 60Hz)
        var sb = new StringBuilder(512);
        sb.Append("{\"left\":{");
        AppendPosRot(sb, leftPos, leftRot);
        sb.Append($",\"trigger\":{leftTrigger:F3}");
        sb.Append($",\"grip\":{leftGrip:F3}");
        sb.Append($",\"thumbstick\":[{leftThumb.x:F3},{leftThumb.y:F3}]");
        sb.Append($",\"primaryButton\":{BoolStr(leftPrimary)}");
        sb.Append($",\"secondaryButton\":{BoolStr(leftSecondary)}");
        sb.Append("},\"right\":{");
        AppendPosRot(sb, rightPos, rightRot);
        sb.Append($",\"trigger\":{rightTrigger:F3}");
        sb.Append($",\"grip\":{rightGrip:F3}");
        sb.Append($",\"thumbstick\":[{rightThumb.x:F3},{rightThumb.y:F3}]");
        sb.Append($",\"primaryButton\":{BoolStr(rightPrimary)}");
        sb.Append($",\"secondaryButton\":{BoolStr(rightSecondary)}");
        sb.Append("},\"head\":{");
        AppendPosRot(sb, headPos, headRot);
        sb.Append("},\"timestamp\":");
        sb.Append(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
        sb.Append("}");

        return sb.ToString();
    }

    private static void AppendPosRot(StringBuilder sb, Vector3 pos, Quaternion rot)
    {
        sb.Append($"\"pos\":[{pos.x:F4},{pos.y:F4},{pos.z:F4}]");
        sb.Append($",\"rot\":[{rot.x:F4},{rot.y:F4},{rot.z:F4},{rot.w:F4}]");
    }

    private static string BoolStr(bool v) => v ? "true" : "false";

    private void SendThreadLoop()
    {
        while (_isRunning)
        {
            if (!_isConnected || _socket == null || !_socket.Connected)
            {
                Thread.Sleep(100);
                continue;
            }

            try
            {
                if (_sendQueue.TryDequeue(out string msg))
                {
                    byte[] data = Encoding.UTF8.GetBytes(msg);
                    _socket.Send(data, 0, data.Length, SocketFlags.None);
                }
                else
                {
                    Thread.Sleep(1);
                }
            }
            catch (SocketException e)
            {
                Debug.LogError($"[XLeRobot] Send error: {e.Message}");
                _isConnected = false;
                CloseSocket();
            }
            catch (Exception e)
            {
                Debug.LogError($"[XLeRobot] Send thread error: {e.Message}");
                Thread.Sleep(100);
            }
        }
    }

    private void CloseSocket()
    {
        try
        {
            if (_socket != null)
            {
                if (_socket.Connected)
                    _socket.Shutdown(SocketShutdown.Both);
                _socket.Close();
            }
        }
        catch { }
        _socket = null;
    }

    private void OnDestroy()
    {
        _isRunning = false;
        _isConnected = false;
        CloseSocket();
    }

    private void OnApplicationQuit()
    {
        _isRunning = false;
        _isConnected = false;
        CloseSocket();
    }

    // ---- Optional: On-screen debug display ----
    private void OnGUI()
    {
        string status = _isConnected ? $"Connected to {pcIpAddress}" : "Disconnected";
        GUI.Label(new Rect(10, 10, 400, 30), $"[XLeRobot] {status}");
    }
}
