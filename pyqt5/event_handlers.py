# event_handlers.py

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QMessageBox
import numpy as np
from sam2_wrapper import SAM2Interface

class MouseEventHandler:
    """
    Handles mouse interactions on the left frame:
    - Left click: add positive prompt
    - Right click: add negative prompt
    """
    def __init__(self, main_window, predictor):
        self.main_window = main_window
        self.predictor = predictor

    def handle_mouse_press(self, event, coords):
        if event.button() == Qt.LeftButton:
            self._handle_left_click(coords)
        elif event.button() == Qt.RightButton:
            self._handle_right_click(coords)

    def _handle_left_click(self, coords):
        obj_id = self.main_window.prompt_for_object_id()
        if obj_id is None:
            return

        frame_idx = self.main_window.current_frame_idx
        merged = self.predictor.add_new_points_or_box(
            obj_id=obj_id,
            frame_idx=frame_idx,
            points=[coords],
            labels=[1],  # positive
        )
        self.main_window.update_left_frame(self.predictor.get_cached_frame(frame_idx))
        self.main_window.update_object_id_counter(obj_id)

    def _handle_right_click(self, coords):
        frame_idx = self.main_window.current_frame_idx
        obj_id = self.predictor.get_object_id_at_point(coords, frame_idx)
        if obj_id is None:
            return

        # Add negative prompt and update overlay
        merged = self.predictor.add_new_points_or_box(
            obj_id=obj_id,
            frame_idx=frame_idx,
            points=[coords],
            labels=[0],  # negative
        )
        self.main_window.update_left_frame(self.predictor.get_cached_frame(frame_idx))