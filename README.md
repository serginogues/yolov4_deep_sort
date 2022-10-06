# YOLO + Deep SORT
YOLO + Deep SORT (detection + tracking) pipeline.

### `run_video.py`
Track people from your video with:
```
python run_video.py --input path/to/video.mp4 --save True
```

### `generate_clips.py`
Generate `.npy` files from tracking clips with:
```
python generate_clips.py --input path/to/repo/with/videos
```

## New pipeline
### YOLOv5 (Ultralytics)

```python
from yolo.yolov5utils import Detector

yolo_model = Detector()  # init
bboxes = yolo_model.forward(frame)  # inference
```

### DeepSORT

```python
from deep_sort.tracker import Tracker

tracker = Tracker()  # init
for frame in video:
    bboxes = yolo_model.forward(frame)  # inference yolo
    tracker.predict(frame, boxes)
    frame = tracker.display(frame)  # draw bboxes
```