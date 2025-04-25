# frame_display_widget.py

from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np


def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    """
    Convert a NumPy image (HWC, RGB or grayscale) to QPixmap.
    """
    if image.ndim == 2:  # Grayscale
        h, w = image.shape
        bytes_per_line = w
        qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    else:  # RGB
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    return QPixmap.fromImage(qimage.copy())


class FrameDisplayWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)

    def update_image(self, image_np: np.ndarray):
        pixmap = numpy_to_qpixmap(image_np)
        self.setPixmap(pixmap)

    def clear_image(self):
        self.clear()
