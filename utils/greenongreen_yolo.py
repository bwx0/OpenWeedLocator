#!/usr/bin/env python
import cv2
from roiyolowd.weed_detector import YOLOv8WithROIDetector
from ultralytics.models.yolo.model import YOLO


class GreenOnGreenYOLO:
    def __init__(self, model_path='models', confidence=0.3, crop_labels="", imgsz=640, show_ROIs=False):
        if model_path is None:
            raise Exception("No model path provided")
        self.yolo_model = YOLO(model_path)
        self.detector = YOLOv8WithROIDetector(self.yolo_model,
                                              confidence_threshold=confidence,
                                              use_native_reassembler=True,
                                              imgsz=imgsz)
        self.crop_labels = [int(x) for x in crop_labels.split(",")]
        self.show_ROIs = show_ROIs

    def inference(self, image):
        import time
        st = time.time()
        result, labelled_image = self.detector.detect_and_draw(image)
        t = time.time() - st
        print(f"inference time={t}s   fps={1/t}")
        print("\n" + " " * 60)

        if self.show_ROIs:
            cv2.imshow("Reassembled Image", self.detector.prev_reassembled_image)

        self.weed_centers = []
        self.boxes = []

        for weed_label in result:
            r = weed_label.rect
            if weed_label.cls in self.crop_labels:  # Filter out crop detections
                continue
            self.boxes.append([r.x, r.y, r.w, r.h])
            cx, cy = r.x + r.w // 2, r.y + r.h // 2
            self.weed_centers.append([cx, cy])
            cv2.circle(labelled_image, (cx, cy), radius=3, color=(0, 0, 255), thickness=3)

        return None, self.boxes, self.weed_centers, labelled_image
