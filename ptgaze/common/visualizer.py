import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .face import Face

AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # B, G, R


class Visualizer:
    def __init__(self, camera: Camera, center_point_index: int):
        self._camera = camera
        self._center_point_index = center_point_index
        self.image: Optional[np.ndarray] = None
        self.max_width_m = 0

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def draw_bbox(self,
                  bbox: np.ndarray,
                  name: int,
                  distance: float,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  lw: int = 1) -> None:
        assert self.image is not None
        assert bbox.shape == (2, 2)
        bbox = np.round(bbox).astype(np.int).tolist()
        cv2.rectangle(self.image, tuple(bbox[0]), tuple(bbox[1]), color, lw)
        cv2.putText(self.image, str(name), (bbox[0][0], bbox[0][1] + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(self.image, str(round(distance, 2)), (bbox[0][0], bbox[1][1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
        center = [bbox[0][0] + (bbox[1][0] - bbox[0][0])//2, bbox[0][1] + (bbox[1][1] - bbox[0][1])//2]

        if center[0] <= self.image.shape[1]//3:
            cv2.putText(self.image, "sx", (bbox[0][0], bbox[0][1] - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
        elif center[0] <= self.image.shape[1]//3*2:
            cv2.putText(self.image, "center", (bbox[0][0], bbox[0][1] - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(self.image, "dx", (bbox[0][0], bbox[0][1] - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _convert_pt(point: np.ndarray) -> Tuple[int, int]:
        return tuple(np.round(point).astype(np.int).tolist())

    def draw_points(self,
                    points: np.ndarray,
                    color: Tuple[int, int, int] = (0, 0, 255),
                    size: int = 3) -> None:
        assert self.image is not None
        assert points.shape[1] == 2
        for pt in points:
            pt = self._convert_pt(pt)
            cv2.circle(self.image, pt, size, color, cv2.FILLED)

    def draw_3d_points(self,
                       points3d: np.ndarray,
                       color: Tuple[int, int, int] = (255, 0, 255),
                       size=3) -> None:
        assert self.image is not None
        assert points3d.shape[1] == 3
        points2d = self._camera.project_points(points3d)
        self.draw_points(points2d, color=color, size=size)

    def draw_3d_line(self,
                     point0: np.ndarray,
                     point1: np.ndarray,
                     color: Tuple[int, int, int] = (255, 255, 0),
                     lw=1) -> None:
        assert self.image is not None
        assert point0.shape == point1.shape == (3, )
        points3d = np.vstack([point0, point1])
        points2d = self._camera.project_points(points3d)
        pt0 = self._convert_pt(points2d[0])
        pt1 = self._convert_pt(points2d[1])
        cv2.line(self.image, pt0, pt1, color, lw, cv2.LINE_AA)
    
    def custom_draw_3d_line(self,
                     face: Face,
                     lenght: float,
                     color: Tuple[int, int, int] = (255, 255, 0),
                     lw=2) -> None:
        point0 = face.center
        gaze_vector = face.gaze_vector
        # point1 = point0 + lenght * face.gaze_vector
        point1 = point0 + lenght * gaze_vector
        assert self.image is not None
        assert point0.shape == point1.shape == (3, )
        points3d = np.vstack([point0, point1])
        points2d = self._camera.project_points(points3d)
        pt0 = self._convert_pt(points2d[0])
        pt1 = self._convert_pt(points2d[1])
        cv2.line(self.image, pt0, pt1, color, lw, cv2.LINE_AA)
        pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
        bbox = np.round(face.bbox).astype(np.int).tolist()
        cv2.putText(self.image, f"yaw: {np.round(yaw, 2)}", (bbox[0][0], bbox[1][1] + 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.90, color, 1, cv2.LINE_AA)
        point_on_screen = face.center - face.gaze_vector * face.center[2] / face.gaze_vector[2]
        point_on_screen[-1] = 0.0
        point_on_screen2d = self._convert_pt(self._camera.project_points(point_on_screen.reshape(1, -1))[0])
        point_on_screen2d = (point_on_screen2d[0], self.image.shape[0]//2)
        if point_on_screen2d[0] < 0 or point_on_screen2d[0] > self.image.shape[1]:
            print("width exceeded")
        self.max_width_m = max(self.max_width_m, np.abs(point_on_screen[0]))
        cv2.circle(self.image, point_on_screen2d, 5, (0,0,255), cv2.FILLED)

    def draw_model_axes(self, face: Face, length: float, lw: int = 2) -> None:
        assert self.image is not None
        assert face is not None
        assert face.head_pose_rot is not None
        assert face.head_position is not None
        assert face.landmarks is not None
        # Get the axes of the model coordinate system
        axes3d = np.eye(3, dtype=np.float) @ Rotation.from_euler(
            'XYZ', [0, np.pi, 0]).as_matrix()
        axes3d = axes3d * length
        axes2d = self._camera.project_points(axes3d,
                                             face.head_pose_rot.as_rotvec(),
                                             face.head_position)

        # system_axes2d = self._camera.project_points(axes3d,
        #                                      np.zeros(3),
        #                                      face.head_position)

        center = face.landmarks[self._center_point_index]
        center = self._convert_pt(center)
        for pt, color in zip(axes2d, AXIS_COLORS):
            pt = self._convert_pt(pt)
            cv2.line(self.image, center, pt, color, lw, cv2.LINE_AA)

        # for pt, color in zip(system_axes2d, AXIS_COLORS):
        #     pt = self._convert_pt(pt)
        #     cv2.line(self.image, center, pt, color, lw, cv2.LINE_AA)


        bbox = np.round(face.bbox).astype(np.int).tolist()
        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        cv2.putText(self.image, f"yaw: {np.round(yaw, 2)}", (bbox[0][0], bbox[1][1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.90, color, 1, cv2.LINE_AA)
