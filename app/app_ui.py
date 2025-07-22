import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTabWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox, QTextEdit, QDateEdit)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDate, Qt


class UploadAnalyzeTab(QWidget):
    # Tab 1: Upload an image and assess sustainability

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Image preview area
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Upload Image")
        self.btn_analyze = QPushButton("Analyse")
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_analyze)

        # Results display
        self.result_text = QLabel(
            "Material: -\nComposition: -\nSustainability Index: -"
        )
        self.result_text.setAlignment(Qt.AlignLeft)
        self.result_text.setStyleSheet("background-color: #f0f0f0; padding: 5px;")

        # Assemble layout
        layout.addWidget(self.image_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.result_text)

        # Connections
        self.btn_select.clicked.connect(self.load_image)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if fname:
            pix = QPixmap(fname).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pix)
            self.result_text.setText(
                "Material: -\nComposition: -\nSustainability Index: -"
            )


class CaptureAnalyzeTab(QWidget):
    # Tab 2: Capture image via webcam, select region, and assess

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.image_label = QLabel("No image captured")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        btn_layout = QHBoxLayout()
        self.btn_capture = QPushButton("Take Picture")
        self.btn_roi = QPushButton("Select Region of Interest")
        self.btn_analyze = QPushButton("Analyse")
        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_roi)
        btn_layout.addWidget(self.btn_analyze)

        self.result_text = QLabel(
            "Material: -\nComposition: -\nSustainability Index: -"
        )
        self.result_text.setAlignment(Qt.AlignLeft)
        self.result_text.setStyleSheet("background-color: #f0f0f0; padding: 5px;")

        layout.addWidget(self.image_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.result_text)


class TrainPredictTab(QWidget):
    # Tab 3: Train model with images & predict new images

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Training section
        train_box = QGroupBox("Training")
        train_layout = QVBoxLayout(train_box)
        self.btn_upload_train = QPushButton("Upload Training Images && Labels")
        self.btn_train_model = QPushButton("Train Model")
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        train_layout.addWidget(self.btn_upload_train)
        train_layout.addWidget(self.btn_train_model)
        train_layout.addWidget(self.train_log)

        # Prediction section
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

        # Connections
        self.btn_select_image.clicked.connect(self.load_image_pred)

    def load_image_pred(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if fname:
            pix = QPixmap(fname).scaled(
                self.pred_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.pred_image_label.setPixmap(pix)
            self.pred_result.setText("Prediction: -\nSustainability Index: -")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sustainability Assessment")
        self.resize(900, 700)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Header with title & date
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

        # Tab widget containing all functionalities
        tabs = QTabWidget()
        tabs.addTab(UploadAnalyzeTab(), "Upload && Assess")
        tabs.addTab(CaptureAnalyzeTab(), "Capture && Assess")
        tabs.addTab(TrainPredictTab(), "Train && Predict")

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
