# ui_main_window.py

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QSlider, QMessageBox, QSpinBox, QLabel
)
from PyQt5.QtCore import Qt

from frame_display_widget import FrameDisplayWidget
from sam2_wrapper import SAM2Interface
from event_handlers import MouseEventHandler
from file_utils import DirectoryCache


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Interactive Video Annotator")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Image display widgets
        self.left_display = FrameDisplayWidget()
        self.right_display = FrameDisplayWidget()
        self.left_display.mousePressEvent = self._mouse_press_event

        # Layout for images
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.left_display)
        self.image_layout.addWidget(self.right_display)

        # Frame slider and label
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_slider_change)

        self.slider_label = QLabel("0 / 0")
        slider_bar_layout = QHBoxLayout()
        slider_bar_layout.addWidget(self.slider)
        slider_bar_layout.addWidget(self.slider_label)

        # Action buttons
        self.btn_propagate_curr = QPushButton("Propagate Current")
        self.btn_propagate_next = QPushButton("Propagate Next")
        self.btn_save = QPushButton("Save Mask")
        self.btn_reset = QPushButton("Reset")
        self.btn_dummy1 = QPushButton("<Noname 1>")
        self.btn_dummy2 = QPushButton("<Noname 2>")

        # Connect signals
        self.btn_propagate_curr.clicked.connect(self.on_propagate_curr_clicked)
        self.btn_propagate_next.clicked.connect(self.on_propagate_next_clicked)
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)

        # Button layout
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.btn_propagate_curr)
        self.button_layout.addWidget(self.btn_propagate_next)
        self.button_layout.addWidget(self.btn_save)
        self.button_layout.addWidget(self.btn_reset)
        self.button_layout.addWidget(self.btn_dummy1)
        self.button_layout.addWidget(self.btn_dummy2)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(slider_bar_layout)
        self.main_layout.addLayout(self.button_layout)
        self.central_widget.setLayout(self.main_layout)

        # Predictor and utilities
        self.predictor = SAM2Interface()
        self.event_handler = MouseEventHandler(self, self.predictor)
        self.dir_cache = DirectoryCache()

        # Load video or folder
        path = self.dir_cache.ask_directory(
            self, key="load_video", title="Select Video or Image Folder"
        )
        if path:
            self.predictor.init(path)
            self.slider.setMaximum(len(self.predictor) - 1)

        # State
        self.current_frame_idx = 0
        self.object_id_counter = 0

        self.update_slider_range()
        self.update_display_frames()

    def update_slider_range(self):
        total = len(self.predictor)
        self.slider.setMaximum(max(0, total - 1))
        self.slider_label.setText(f"{self.current_frame_idx} / {total - 1 if total > 0 else 0}")

    def on_slider_change(self, value):
        self.current_frame_idx = value
        total = len(self.predictor)
        self.slider_label.setText(f"{value} / {total - 1 if total > 0 else 0}")
        self.update_display_frames()

    def update_display_frames(self):
        raw = self.predictor.get_raw_frame(self.current_frame_idx)
        left_overlay = self.predictor.get_cached_frame(self.current_frame_idx)
        right_overlay = self.predictor.get_masked_frame(self.current_frame_idx)

        self.left_display.update_image(left_overlay if left_overlay is not None else raw)
        self.right_display.update_image(right_overlay if right_overlay is not None else raw)

    def on_propagate_curr_clicked(self):
        self.predictor.propagate_until(self.current_frame_idx)
        self.update_display_frames()

    def on_propagate_next_clicked(self):
        next_idx = self.predictor.propagate_next(self.current_frame_idx)
        if next_idx is not None:
            self.current_frame_idx = next_idx
            self.slider.setValue(next_idx)
        else:
            QMessageBox.information(self, "Info", "Already at last frame.")
        self.update_display_frames()

    def on_save_clicked(self):
        path = self.dir_cache.ask_directory(
            self, key="save_mask", title="Select Save Directory"
        )
        if path:
            self.predictor.save_cached_frames(path)

    def on_reset_clicked(self):
        self.predictor.reset_state()
        self.update_display_frames()

    def _mouse_press_event(self, event):
        coords = self._map_mouse_to_image(event.pos(), self.left_display)
        if coords:
            self.event_handler.handle_mouse_press(event, coords)

    def _map_mouse_to_image(self, pos, label_widget):
        return (int(pos.x()), int(pos.y()))

    def prompt_for_object_id(self):
        spinbox = QSpinBox()
        spinbox.setMinimum(1)
        spinbox.setMaximum(20)
        spinbox.setValue(self.object_id_counter + 1)

        msg = QMessageBox(self)
        msg.setWindowTitle("Select Object ID")
        msg.setText("Enter Object ID for the prompt:")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.layout().addWidget(spinbox)

        if msg.exec_() == QMessageBox.Ok:
            return spinbox.value()
        return None

    def update_object_id_counter(self, obj_id):
        if 1 <= obj_id <= 20 and obj_id > self.object_id_counter:
            self.object_id_counter = obj_id

    def update_left_frame(self, overlay_img):
        self.left_display.update_image(overlay_img)
