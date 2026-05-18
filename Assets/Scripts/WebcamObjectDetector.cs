using System.Collections.Generic;
using TMPro;

using UnityEngine;
using UnityEngine.UI;

/// Same idea as SentisObjectDetector but driven by a regular WebCamTexture
/// instead of AR Foundation. Works in the editor on any laptop with a webcam.
public class WebcamObjectDetector : MonoBehaviour
{
    [Header("Model")]
    [SerializeField] Unity.InferenceEngine.ModelAsset modelAsset;
    [SerializeField] TextAsset classNamesFile;
    [SerializeField] int inputSize = 320;
    [SerializeField] float confidenceThreshold = 0.20f;
    [SerializeField] float iouThreshold = 0.45f;
    [SerializeField] string[] targetClasses = { "book" }; // empty array = all classes
    [SerializeField] ClassMessage[] classMessages = {
        new() { className = "book", message = "Found a book!" },
    };

    [Header("Camera")]
    [SerializeField] int requestedWidth = 1280;
    [SerializeField] int requestedHeight = 720;
    [SerializeField] int requestedFps = 30;
    [SerializeField] RawImage previewImage;       // optional: a UI RawImage to show the webcam feed behind the boxes

    [Header("UI")]
    [SerializeField] RectTransform labelParent;
    [SerializeField] GameObject labelPrefab;

    [Header("Performance")]
    [SerializeField] int runEveryNFrames = 5;

    Unity.InferenceEngine.Worker _worker;
    Unity.InferenceEngine.Model _model;
    string[] _classNames;
    WebCamTexture _webcam;
    RenderTexture _resized;
    int _frameCounter;
    readonly List<GameObject> _activeLabels = new();

    void Start()
    {
        Debug.Log("[Webcam] Awake — initialising");

        _model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
        _worker = new Unity.InferenceEngine.Worker(_model, Unity.InferenceEngine.BackendType.GPUCompute);

        _classNames = classNamesFile != null
            ? classNamesFile.text.Split('\n')
            : new[] { "person" };

        // Pick the first available webcam
        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("[Webcam] No webcam detected.");
            return;
        }

        string deviceName = WebCamTexture.devices[0].name;
        _webcam = new WebCamTexture(deviceName, requestedWidth, requestedHeight, requestedFps);
        _webcam.Play();

        if (previewImage != null)
            previewImage.texture = _webcam;

        _resized = new RenderTexture(inputSize, inputSize, 0, RenderTextureFormat.ARGB32);
    }

    void Update()
    {
        if (_webcam == null || !_webcam.didUpdateThisFrame) return;
        _frameCounter++;
        if (_frameCounter % runEveryNFrames != 0) return;

        Detect();
    }

    void OnDestroy()
    {
        if (_webcam != null) _webcam.Stop();
        _worker?.Dispose();
        if (_resized != null) _resized.Release();
    }

    void Detect()
    {
        // 1. Resize webcam frame -> RenderTexture at model input size
        Graphics.Blit(_webcam, _resized);

        // 2. RenderTexture -> tensor (1, 3, H, W), 0..1 range
        using var input = Unity.InferenceEngine.TextureConverter.ToTensor(_resized, inputSize, inputSize, 3);

        // 3. Inference
        _worker.Schedule(input);
        using var output = (_worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>).ReadbackAndClone();

        // 4. Parse YOLOv8 output [1, 84, 8400]
        ClearLabels();
        var data = output.AsReadOnlySpan();
        int numAnchors = output.shape[2];
        int numClasses = _classNames.Length;

        var candidates = new List<Detection>();
        float topScore = 0f; string topName = "";

        for (int i = 0; i < numAnchors; i++)
        {
            int bestClass = 0; float bestScore = 0f;
            for (int c = 0; c < numClasses; c++)
            {
                float s = data[(4 + c) * numAnchors + i];
                if (s > bestScore) { bestScore = s; bestClass = c; }
            }

            if (bestScore > topScore)
            {
                topScore = bestScore;
                topName = _classNames[bestClass].Trim();
            }
            if (bestScore < confidenceThreshold) continue;

            string name = _classNames[bestClass].Trim();
            if (!ClassIsAllowed(name)) continue;

            candidates.Add(new Detection
            {
                Name = name,
                Score = bestScore,
                Cx = data[0 * numAnchors + i],
                Cy = data[1 * numAnchors + i],
                W  = data[2 * numAnchors + i],
                H  = data[3 * numAnchors + i],
            });
        }

        Debug.Log($"[Webcam] Top: {topName} @ {topScore:0.000} | passed threshold: {candidates.Count}");

        // 5. Non-Max Suppression
        candidates.Sort((a, b) => b.Score.CompareTo(a.Score));
        var kept = new List<Detection>();
        foreach (var cand in candidates)
        {
            bool overlaps = false;
            foreach (var k in kept)
                if (IoU(cand, k) > iouThreshold) { overlaps = true; break; }
            if (!overlaps) kept.Add(cand);
        }

        foreach (var k in kept)
            SpawnLabel(k.Name, k.Score, k.Cx, k.Cy, k.W, k.H);
    }

    void SpawnLabel(string name, float score, float cx, float cy, float w, float h)
    {
        var go = Instantiate(labelPrefab, labelParent);
        var rt = (RectTransform)go.transform;
        string display = $"{LookupMessage(name)} ({score:0.00})";

        // Support either TMP_Text or legacy uGUI Text in the prefab
        var tmp = go.GetComponentInChildren<TMP_Text>();
        if (tmp != null) tmp.text = display;
        else
        {
            var text = go.GetComponentInChildren<Text>();
            if (text != null) text.text = display;
            else Debug.LogWarning("[Webcam] LabelPrefab has no TMP_Text or Text child — label can't update");
        }

        float sx = cx / inputSize * Screen.width;
        float sy = (1f - cy / inputSize) * Screen.height;
        rt.position = new Vector3(sx, sy, 0);
        rt.sizeDelta = new Vector2(w / inputSize * Screen.width, h / inputSize * Screen.height);

        _activeLabels.Add(go);
    }

    void ClearLabels()
    {
        foreach (var l in _activeLabels) Destroy(l);
        _activeLabels.Clear();
    }

    string LookupMessage(string name)
    {
        if (classMessages == null) return name;
        for (int i = 0; i < classMessages.Length; i++)
        {
            if (string.Equals(classMessages[i].className?.Trim(), name,
                              System.StringComparison.OrdinalIgnoreCase))
                return classMessages[i].message;
        }
        return name; // fallback to raw class name
    }

    [System.Serializable]
    public struct ClassMessage
    {
        public string className;
        public string message;
    }

    bool ClassIsAllowed(string name)
    {
        if (targetClasses == null || targetClasses.Length == 0) return true; // empty = allow all
        for (int i = 0; i < targetClasses.Length; i++)
        {
            if (!string.IsNullOrWhiteSpace(targetClasses[i]) &&
                string.Equals(targetClasses[i].Trim(), name, System.StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    struct Detection
    {
        public string Name;
        public float Score;
        public float Cx, Cy, W, H;
    }

    static float IoU(Detection a, Detection b)
    {
        float ax1 = a.Cx - a.W / 2, ay1 = a.Cy - a.H / 2;
        float ax2 = a.Cx + a.W / 2, ay2 = a.Cy + a.H / 2;
        float bx1 = b.Cx - b.W / 2, by1 = b.Cy - b.H / 2;
        float bx2 = b.Cx + b.W / 2, by2 = b.Cy + b.H / 2;

        float ix1 = Mathf.Max(ax1, bx1), iy1 = Mathf.Max(ay1, by1);
        float ix2 = Mathf.Min(ax2, bx2), iy2 = Mathf.Min(ay2, by2);
        float iw = Mathf.Max(0, ix2 - ix1), ih = Mathf.Max(0, iy2 - iy1);
        float inter = iw * ih;

        float ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
        return ua > 0 ? inter / ua : 0f;
    }
}