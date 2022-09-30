import torch
import numpy as np


class Detector:
    def __init__(self):
        self.model = self._load()
        self.results = None

    def forward(self, frame):
        """
        - self.results.pandas().xyxy[0]  # to obtain pd dict from single predicted image
        - self.results.xyxy[0] = results.pred[0]  # bboxes
        - self.results.pandas().xywh[0]
        - self.results.xywh[0]
        - crops = self.results.crop(save=False)
                - crops[i]['im']  # cropped image
                - crops[i]['box']  # box coordinates
                - crops[i]['conf'].item()  # prediction confidence   where @i is [0, n crops)
        """
        self.results = self.model(frame)
        bboxes = self._xyxy_to_xywh(self.results.xyxy[0].cpu())
        return bboxes

    @staticmethod
    def _load():
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        yolo_model.to(device)
        print("YOLO device: ", device)
        yolo_model.classes = 0
        yolo_model.multi_label = False
        yolo_model.conf = 0.5
        return yolo_model

    @staticmethod
    def _xyxy_to_xywh(bboxes):
        """
        Convert from
        :param bboxes: YOLO output. 2D array of size [# detections, 6],
        where the second dim is [xmin, ymin, xmax, ymax, confidence, class]
        :return: boxes = [# detections, 6] array where second dim is [xmin, ymin, width, height, confidence, class]
        """
        boxes = []
        for bbox in bboxes:
            if bbox[5] == 0:
                boxes.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[4], int(bbox[5])])

        return np.array(boxes)
