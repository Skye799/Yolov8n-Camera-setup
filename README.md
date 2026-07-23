# Unity Real-Time Object Detection (YOLOv8n / YOLO12s)

Real-time object detection running fully on-device in Unity 6 via Unity Inference Engine. Point a webcam at a room and get live bounding boxes and labels over the feed.

Two detectors are included:

- **`WebcamObjectDetector.cs`** — the main script. Drives detection from a live `WebCamTexture`, so it runs in the editor on any laptop with a webcam. Written to be tuned from the Inspector without code changes:
  - confidence and IoU thresholds exposed as fields
  - runtime class filtering (`targetClasses`) — restrict detection to chosen classes, or leave empty for all 80 COCO classes
  - per-class trigger messages (e.g. fire an event when a "book" is seen)
  - inference throttling (`runEveryNFrames`) to trade latency against frame rate
- **`RunYOLO.cs`** — Unity's reference sample that runs the model over a video file. Kept for comparison; the webcam detector extends the same approach to a live feed.

## Models

Ships with two ONNX models in `Assets/Models`, swappable by dragging a different asset onto the detector:

| Model | Notes |
|-------|-------|
| `yolov8n.onnx` | Nano model, fastest |
| `yolo12s.onnx` | YOLO12 small, better accuracy at higher cost |

Both were benchmarked against each other on latency vs accuracy for live household object recognition.

## Running it

1. Open in Unity 6 (6000.3+). The project uses URP and the new Input System.
2. Open `Assets/Scenes/SampleScene`.
3. On the detector, assign a model from `Assets/Models` and `Assets/Resources/classes.txt`.
4. Press Play. Boxes and labels render over the webcam preview.
