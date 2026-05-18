# YOLO Webcam Object Detection

A Unity app that runs real-time object detection on a live webcam feed using YOLOv8n and YOLO12s models via Unity Sentis.

## Tech
- Unity 6 + Unity Sentis (on-device ML inference)
- YOLOv8n + YOLO12s (ONNX)
- URP (Universal Render Pipeline)

## What it does
Captures webcam input and runs YOLO inference on each frame, drawing bounding box labels over detected objects in real time — no server, runs fully on-device.
