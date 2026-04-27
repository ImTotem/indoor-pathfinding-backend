import threading
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PERSON_CLASS = 0  # COCO class index for "person"


class PersonMasker:
    """Singleton YOLO-based person masker.

    Detects person bounding boxes in query images and zeros out those regions
    before feature extraction, reducing false matches caused by moving people.
    Falls back silently if ultralytics or weights are unavailable.
    """

    _instance = None
    _lock = threading.Lock()
    _model = None
    _available: bool | None = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self) -> bool:
        if self._available is not None:
            return self._available
        with self._lock:
            if self._available is not None:
                return self._available
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8n.pt")
                self._available = True
                logger.info("[PersonMasker] YOLOv8n loaded")
            except Exception as e:
                self._available = False
                logger.warning(f"[PersonMasker] YOLO unavailable, masking disabled: {e}")
        return self._available

    def mask(self, img_bytes: bytes, grayscale: np.ndarray) -> np.ndarray:
        """Zero out person regions in grayscale image.

        Args:
            img_bytes: Raw image bytes (for YOLO detection in color).
            grayscale: Already-decoded grayscale image (feature extraction target).

        Returns:
            Grayscale image with person bounding boxes filled with 0.
            Returns original grayscale unchanged on any failure.
        """
        if not self._load_model():
            return grayscale

        try:
            bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return grayscale

            results = self._model(bgr, classes=[_PERSON_CLASS], verbose=False)

            masked = grayscale.copy()
            h_ratio = grayscale.shape[0] / bgr.shape[0]
            w_ratio = grayscale.shape[1] / bgr.shape[1]

            num_masked = 0
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    gx1 = int(x1 * w_ratio)
                    gy1 = int(y1 * h_ratio)
                    gx2 = int(x2 * w_ratio)
                    gy2 = int(y2 * h_ratio)
                    masked[gy1:gy2, gx1:gx2] = 0
                    num_masked += 1

            if num_masked:
                logger.debug(f"[PersonMasker] Masked {num_masked} person(s)")
            return masked

        except Exception as e:
            logger.warning(f"[PersonMasker] Masking failed: {e}")
            return grayscale
