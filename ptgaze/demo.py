import datetime
import logging
import pathlib
from typing import Optional, Dict

import cv2
import numpy as np
from omegaconf import DictConfig

from .common import Face, FacePartsName, Visualizer, Shop
from .gaze_estimator import GazeEstimator
from .utils import get_3d_face_model

import time
import itertools
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

        self.shop = Shop(self.gaze_estimator.camera, self.config.shop.width, self.config.shop.ids)


    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError


    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)


    def _iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """

        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union


    def _send_results(self, track: Dict) -> Dict:
        occurencies = Counter(track['objects_of_interest'])
        del occurencies[-1]
        person_id = track["id"]     # TODO: replace with re-id module
        return {'id': person_id, 'result': occurencies}     # TODO: send API message
            

    def _run_on_video(self) -> None:
        tracks_active = []
        tracks_finished = []
        sigma_iou = 0.5
        t_min = 60   # minimum number of frames to consider the track valid
        tracks_progressive_id = itertools.count()

        frame_num = 0
        while True:
            start_fps = time.perf_counter()
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.resize(frame,\
                (int(frame.shape[1] * self.config.demo.frame_scale),\
                int(frame.shape[0] * self.config.demo.frame_scale)))
            undistorted = cv2.undistort(frame, self.gaze_estimator.camera.camera_matrix,\
                self.gaze_estimator.camera.dist_coefficients)
            self.visualizer.set_image(frame.copy())

            faces = self.gaze_estimator.detect_faces(undistorted)
            for face in faces:
                self.gaze_estimator.estimate_gaze(undistorted, face)

            updated_tracks = []
            for track in tracks_active:
                if len(faces) > 0:
                    # get det with highest iou
                    best_match = max(faces, key=lambda x: self._iou(track['bboxes'][-1], x.bbox.flatten()))
                    if self._iou(track['bboxes'][-1], best_match.bbox.flatten()) >= sigma_iou:
                        track['bboxes'].append(best_match.bbox.flatten())
                        track['faces'].append(best_match)
                        track['objects_of_interest'].append(self.shop.estimate_ooi(face))
                        updated_tracks.append(track)

                        # remove the best matching detection from detections
                        del faces[faces.index(best_match)]

                # if track was not updated, finish it
                if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                    if len(track['bboxes']) >= t_min:
                        tracks_finished.append(track)
                        self._send_results(track)
            
            # create new tracks if there are detections left
            new_tracks = []
            for face in faces:
                new_tracks.append({'id': next(tracks_progressive_id),\
                    'bboxes': [face.bbox.flatten()],\
                    'faces': [face],\
                    'start_frame': frame_num,\
                    'objects_of_interest': [self.shop.estimate_ooi(face)]})
            tracks_active = updated_tracks + new_tracks
            
            # visualization
            for idx, track in enumerate(tracks_active):
                if len(track['bboxes']) >= t_min:
                    face = track['faces'][-1]
                    self._draw_face_bbox(face, track['id'])
                    self._draw_head_pose(face)
                    self._draw_landmarks(face)
                    self._draw_face_template_model(face)
                    self._draw_gaze_vector(face)
                    self._display_normalized_image(face)
    
            if self.config.demo.use_camera:
                self.visualizer.image = self.visualizer.image[:, ::-1]
            if self.writer:
                self.writer.write(self.visualizer.image)
            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
            
            logging.info(f"fps: {1.0 / (time.perf_counter() - start_fps)}")
            frame_num += 1

        self.cap.release()
        if self.writer:
            self.writer.release()


    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for idx, face in enumerate(faces):
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face, idx)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)


    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.gaze_estimator.camera.width*self.config.demo.frame_scale) )
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.gaze_estimator.camera.height*self.config.demo.frame_scale))
        return cap


    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir


    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')


    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (int(self.gaze_estimator.camera.width),
                                  int(self.gaze_estimator.camera.height)))
        if writer is None:
            raise RuntimeError
        return writer


    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True


    def _draw_face_bbox(self, face: Face, track_id: int) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox, track_id, face.distance * self.config.gaze_estimator.normalized_camera_distance)
        bbox = np.round(face.bbox).astype(np.int).tolist()
        logger.info(f'{track_id}: [bbox] top-left [{bbox[0][0]}, {bbox[0][1]}] - bottom-right [{bbox[1][0]}, {bbox[1][1]}]')


    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance * self.config.gaze_estimator.normalized_camera_distance:.2f}')


    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)


    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)


    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)


    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            # self.visualizer.draw_3d_line(
            #     face.center, face.center + length * face.gaze_vector)
            self.visualizer.custom_draw_3d_line(face, length)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError
