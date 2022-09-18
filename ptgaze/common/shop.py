import sys
from typing import Optional, List

import cv2
import numpy as np

from .camera import Camera
from .face import Face


class Shop:
    def __init__(self, camera: Camera, shop_width: float, shop_ids: List[int]):
        self._camera = camera
        self.shop_width = shop_width
        self.shop_ids = shop_ids    # list of ids of objects inside the shop window


    def estimate_ooi(self, face: Face) -> int:
        point_on_screen = face.center - face.gaze_vector * face.center[2] / face.gaze_vector[2]
        point_on_screen[-1] = 0.0   # the camera plane has z=0.0
        # point_on_screen2d = self._camera.project_points(point_on_screen.reshape(1, -1))[0]
        # point_on_screen2d = np.round(point_on_screen2d).astype(np.int).tolist()
        # point_on_screen2d = (point_on_screen2d[0], self.image.shape[0]//2)    # for visualization i don't need the y
        # poi2d_x = point_on_screen2d[0]

        start = -self.shop_width/2
        for obj_id in self.shop_ids:
            end = min(start + round(self.shop_width/len(self.shop_ids), 2), self.shop_width/2)
            if start < point_on_screen[0] < end:
                return obj_id
            start = end
        return -1