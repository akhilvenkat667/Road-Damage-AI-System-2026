# Models Folder

Place your model weight files here:

- `best.pt`  — your custom-trained YOLOv8 road damage detection model (preferred, used automatically if present).
- `yolov8n.pt` — the stock pretrained YOLOv8 nano model, used automatically as a fallback if `best.pt` is not present.

If neither file exists, the Ultralytics library will attempt to auto-download `yolov8n.pt` on first run
(requires an internet connection). For real road-damage detection accuracy, train or download a custom
`best.pt` model on a pothole/crack dataset (e.g. RDD2022) and place it here.
