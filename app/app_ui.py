import sys
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox, QTextEdit, QDateEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QDate, Qt, QTimer


def resize_with_aspect_ratio(image, max_width=800, max_height=600):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > max_width or h > max_height:
        if aspect_ratio > (max_width / max_height):
            new_w = max_width
            new_h = int(max_width / aspect_ratio)
        else:
            new_h = max_height
            new_w = int(max_height * aspect_ratio)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def draw_annotations(image, lines):
    image = resize_with_aspect_ratio(image)
    h, w, _ = image.shape

    scale = max(0.5, min(1.5, h / 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(scale * 2))
    line_gap = int(30 * scale)

    sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text in lines]
    text_width = max(size[0] for size in sizes)
    text_height = sum(size[1] for size in sizes) + line_gap // 2 * (len(lines) - 1)

    margin = 10
    x0 = w - text_width - margin * 2
    y0 = h - text_height - margin * 2

    x0 = max(margin, x0)
    y0 = max(margin, y0)

    cv2.rectangle(
        image,
        (x0 - margin, y0 - margin),
        (x0 + text_width + margin, y0 + text_height + margin),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )

    y_cursor = y0 + sizes[0][1]
    for idx, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (x0, y_cursor),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        if idx < len(lines) - 1:
            y_cursor += sizes[idx + 1][1] + line_gap // 2

    return image


class UploadAnalyzeTab(QWidget):
    def __init__(self, date_selector: QDateEdit, parent=None):
        super().__init__(parent)
        self.date_selector = date_selector
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
        self.btn_analyze.clicked.connect(self.run_analysis)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fname:
            self.current_image_path = fname
            pix = QPixmap(fname).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pix)
            self.result_text.setText("Material: -\nComposition: -\nSustainability Index: -")

    def run_analysis(self):
        if not hasattr(self, 'current_image_path'):
            QMessageBox.information(self, "No image", "Please upload an image first.")
            return

        material = "Steel"
        composition = "Fe, C"
        index = "0.041"
        self.result_text.setText(f"Material: {material}\nComposition: {composition}\nSustainability Index: {index}")

        image = cv2.imread(self.current_image_path)
        if image is None:
            QMessageBox.warning(self, "Load Error", "Failed to read the image file.")
            return

        lines = [
            f"Material: {material}",
            f"Composition: {composition}",
            f"Sustainability Index: {index}",
            f"Date: {self.date_selector.date().toString('dd/MM/yyyy')}"
        ]

        annotated = draw_annotations(image, lines)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"analyzed_snapshots/annotated_upload_{timestamp}.jpg"
        cv2.imwrite(out_path, annotated)

        image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        qt_image = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        QMessageBox.information(self, "Saved", f"Annotated image saved as {out_path}")


class CaptureAnalyzeTab(QWidget):
    def __init__(self, date_selector: QDateEdit, parent=None):
        super().__init__(parent)
        self.date_selector = date_selector
        self.cap = None
        self.timer = QTimer(self)
        self.frame = None
        self.state = "stopped"
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
        self.btn_analyze.clicked.connect(self.run_analysis)

    def toggle_camera_or_snapshot(self):
        if self.state in {"stopped", "captured"}:
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
        qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def crop_roi(self):
        if self.frame is not None:
            roi = cv2.selectROI("Select ROI", self.frame, False, False)
            cv2.destroyWindow("Select ROI")
            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                cropped = self.frame[y:y + h, x:x + w]
                if cropped.size == 0:
                    QMessageBox.warning(self, "Invalid ROI", "Selected area is empty.")
                    return

                min_w, min_h = 400, 300
                ch, cw = cropped.shape[:2]
                if cw < min_w or ch < min_h:
                    scale_w = min_w / cw
                    scale_h = min_h / ch
                    scale = max(scale_w, scale_h)
                    new_w, new_h = int(cw * scale), int(ch * scale)
                    cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                self.frame = cropped
                self.display_image(cropped)


    def run_analysis(self):
        if self.frame is None:
            QMessageBox.information(self, "No snapshot", "Please take a snapshot first.")
            return

        material = "Stainless Steel"
        composition = "Fe, Cr, Ni"
        index = "0.065"
        self.result_text.setText(f"Material: {material}\nComposition: {composition}\nSustainability Index: {index}")

        lines = [
            f"Material: {material}",
            f"Composition: {composition}",
            f"Sustainability Index: {index}",
            f"Date: {self.date_selector.date().toString('dd/MM/yyyy')}"
        ]

        annotated = draw_annotations(self.frame, lines)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"analyzed_snapshots/annotated_upload_{timestamp}.jpg"
        cv2.imwrite(out_path, annotated)

        self.display_image(annotated)
        QMessageBox.information(self, "Saved", f"Annotated snapshot saved as {out_path}")

    def dummy_analysis(self):
        self.run_analysis()


class TrainPredictTab(QWidget):
    def __init__(self, date_selector: QDateEdit, parent=None):
        super().__init__(parent)
        self.date_selector = date_selector
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
            material = "Steel"
            index = "0.041"
            composition = "Fe, C"
            self.pred_result.setText(f"Prediction: {material}\nSustainability Index: {index}")

            image = cv2.imread(fname)
            if image is None:
                QMessageBox.warning(self, "Load Error", "Failed to read the image file.")
                return

            lines = [
                f"Material: {material}",
                f"Composition: {composition}",
                f"Sustainability Index: {index}",
                f"Date: {self.date_selector.date().toString('dd/MM/yyyy')}"
            ]

            annotated = draw_annotations(image, lines)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"analyzed_snapshots/annotated_upload_{timestamp}.jpg"
            cv2.imwrite(out_path, annotated)

            image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            qt_image = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(self.pred_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pred_image_label.setPixmap(pixmap)
            QMessageBox.information(self, "Saved", f"Annotated prediction saved as {out_path}")

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
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setDisplayFormat("dd/MM/yyyy")
        self.date_edit.setCalendarPopup(True)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Date:"))
        header_layout.addWidget(self.date_edit)

        tabs = QTabWidget()
        tabs.addTab(UploadAnalyzeTab(self.date_edit), "Upload Assess")
        tabs.addTab(CaptureAnalyzeTab(self.date_edit), "Capture Assess")
        tabs.addTab(TrainPredictTab(self.date_edit), "Train Predict")

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
