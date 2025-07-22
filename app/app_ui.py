import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox, QTextEdit, QDateEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QDate, Qt, QTimer, QRect


class UploadAnalyzeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Upload Image")
        self.btn_analyze = QPushButton("Analyse")
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_analyze)

        self.result_text = QLabel("Material: -\nComposition: -\nSustainability Index: -")
        self.result_text.setAlignment(Qt.AlignLeft)
        self.result_text.setStyleSheet("background-color: #f0f0f0; padding: 5px;")

        layout.addWidget(self.image_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.result_text)

        self.btn_select.clicked.connect(self.load_image)
        self.btn_analyze.clicked.connect(self.dummy_analysis)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fname:
            self.current_image_path = fname
            pix = QPixmap(fname).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pix)
            self.result_text.setText("Material: -\nComposition: -\nSustainability Index: -")

    def dummy_analysis(self):
        if hasattr(self, 'current_image_path'):
            self.result_text.setText("Material: Steel\nComposition: Fe, C\nSustainability Index: 0.041")


class CaptureAnalyzeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer(self)
        self.frame = None
        self.state = "stopped"
        self.roi = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.image_label = QLabel("No image captured")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        btn_layout = QHBoxLayout()
        self.btn_toggle_cam = QPushButton("Start Camera")
        self.btn_crop = QPushButton("Crop Snapshot")
        self.btn_analyze = QPushButton("Analyse")
        btn_layout.addWidget(self.btn_toggle_cam)
        btn_layout.addWidget(self.btn_crop)
        btn_layout.addWidget(self.btn_analyze)

        self.result_text = QLabel("Material: -\nComposition: -\nSustainability Index: -")
        self.result_text.setAlignment(Qt.AlignLeft)
        self.result_text.setStyleSheet("background-color: #f0f0f0; padding: 5px;")

        layout.addWidget(self.image_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.result_text)

        self.btn_toggle_cam.clicked.connect(self.toggle_camera_or_snapshot)
        self.btn_crop.clicked.connect(self.crop_roi)
        self.timer.timeout.connect(self.update_frame)
        self.btn_analyze.clicked.connect(self.dummy_analysis)

    def toggle_camera_or_snapshot(self):
        if self.state == "stopped" or self.state == "captured":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.result_text.setText("Error: Cannot access webcam")
                self.cap = None
                return
            self.timer.start(30)
            self.btn_toggle_cam.setText("Take Snapshot")
            self.state = "previewing"

        elif self.state == "previewing":
            self.timer.stop()
            if self.frame is not None:
                cv2.imwrite("captured_snapshot.jpg", self.frame)
                self.display_image(self.frame)
                self.result_text.setText("Snapshot saved as captured_snapshot.jpg")
                self.btn_toggle_cam.setText("Retake Snapshot")
                self.state = "captured"

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.display_image(frame)

    def display_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def crop_roi(self):
        if self.frame is not None:
            roi = cv2.selectROI("Select ROI", self.frame, False, False)
            cv2.destroyWindow("Select ROI")
            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                cropped = self.frame[y:y+h, x:x+w]
                self.display_image(cropped)
                self.frame = cropped

    def dummy_analysis(self):
        self.result_text.setText("Material: Stainless Steel\nComposition: Fe, Cr, Ni\nSustainability Index: 0.065")


class TrainPredictTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        train_box = QGroupBox("Training")
        train_layout = QVBoxLayout(train_box)
        self.btn_upload_train = QPushButton("Upload Training Images and Labels")
        self.btn_train_model = QPushButton("Train Model")
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        train_layout.addWidget(self.btn_upload_train)
        train_layout.addWidget(self.btn_train_model)
        train_layout.addWidget(self.train_log)

        pred_box = QGroupBox("Prediction")
        pred_layout = QVBoxLayout(pred_box)
        self.btn_select_image = QPushButton("Select Image to Predict")
        self.pred_image_label = QLabel("No image selected")
        self.pred_image_label.setAlignment(Qt.AlignCenter)
        self.pred_image_label.setFixedSize(400, 300)
        self.pred_image_label.setStyleSheet("border: 1px solid gray;")
        self.pred_result = QLabel("Prediction: -\nSustainability Index: -")
        self.pred_result.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        pred_layout.addWidget(self.btn_select_image)
        pred_layout.addWidget(self.pred_image_label)
        pred_layout.addWidget(self.pred_result)

        layout.addWidget(train_box)
        layout.addWidget(pred_box)

        self.btn_select_image.clicked.connect(self.load_image_pred)
        self.btn_train_model.clicked.connect(self.dummy_train_model)

    def load_image_pred(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fname:
            pix = QPixmap(fname).scaled(self.pred_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pred_image_label.setPixmap(pix)
            self.pred_result.setText("Prediction: Steel\nSustainability Index: 0.041")

    def dummy_train_model(self):
        self.train_log.append("Training started...")
        self.train_log.append("Training complete. Model saved to model.pth")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sustainability Assessment")
        self.resize(900, 700)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        header_layout = QHBoxLayout()
        title_label = QLabel("Sustainability Assessment")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        date_edit = QDateEdit(QDate.currentDate())
        date_edit.setDisplayFormat("dd/MM/yyyy")
        date_edit.setCalendarPopup(True)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Date:"))
        header_layout.addWidget(date_edit)

        tabs = QTabWidget()
        tabs.addTab(UploadAnalyzeTab(), "Upload Assess")
        tabs.addTab(CaptureAnalyzeTab(), "Capture Assess")
        tabs.addTab(TrainPredictTab(), "Train Predict")

        main_layout.addLayout(header_layout)
        main_layout.addWidget(tabs)

        self.setCentralWidget(central)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
