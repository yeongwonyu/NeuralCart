import sys
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QListWidget, QFileDialog, QTextEdit, QGroupBox, QFormLayout
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF
from PySide6.QtCore import Qt, QPointF


class ArchitectureCanvas(QWidget):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.setMinimumWidth(500)
        self.setMinimumHeight(700)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        x_center = self.width() // 2
        y = 30
        block_h = 70
        gap = 35
        current_dim = 4

        for i, layer in enumerate(self.layers):
            layer_type = layer["type"]

            if layer_type == "Linear":
                in_dim = layer["in_features"]
                out_dim = layer["out_features"]

                top_w = max(60, min(220, in_dim * 22))
                bottom_w = max(60, min(220, out_dim * 22))

                points = QPolygonF([
                    QPointF(x_center - top_w / 2, y),
                    QPointF(x_center + top_w / 2, y),
                    QPointF(x_center + bottom_w / 2, y + block_h),
                    QPointF(x_center - bottom_w / 2, y + block_h),
                ])

                painter.setBrush(QBrush(QColor("#7DB7FF")))
                painter.setPen(QPen(QColor("#2B6CB0"), 2))
                painter.drawPolygon(points)

                painter.setPen(QColor("#1A365D"))
                painter.drawText(
                    int(x_center - 80), y + 25, 160, 20,
                    Qt.AlignCenter, "Linear"
                )
                painter.drawText(
                    int(x_center - 80), y + 48, 160, 20,
                    Qt.AlignCenter, f"{in_dim} → {out_dim}"
                )

                current_dim = out_dim

            else:
                width = max(90, min(240, current_dim * 22))
                x = x_center - width / 2

                painter.setBrush(QBrush(QColor("#FFD28A")))
                painter.setPen(QPen(QColor("#C77C00"), 2))
                painter.drawRoundedRect(int(x), y, int(width), block_h, 12, 12)

                painter.setPen(QColor("#3B2F1E"))
                painter.drawText(
                    int(x), y + 25, int(width), 25,
                    Qt.AlignCenter, layer_type
                )

            if i < len(self.layers) - 1:
                painter.setPen(QPen(QColor("#555555"), 2))
                x = x_center
                y1 = y + block_h
                y2 = y + block_h + gap - 8
                painter.drawLine(x, y1, x, y2)

                # arrow head
                painter.drawLine(x, y2, x - 6, y2 - 8)
                painter.drawLine(x, y2, x + 6, y2 - 8)

            y += block_h + gap

        self.setMinimumHeight(max(700, y + 50))


class NeuralCartBuilder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralCart Builder")
        self.resize(1200, 800)

        self.layers = []

        main = QWidget()
        self.setCentralWidget(main)

        root = QHBoxLayout(main)

        # Left panel
        left_panel = QVBoxLayout()
        root.addLayout(left_panel, 1)

        # Center canvas
        self.canvas = ArchitectureCanvas(self.layers)
        root.addWidget(self.canvas, 2)

        # Right panel
        right_panel = QVBoxLayout()
        root.addLayout(right_panel, 1)

        # Layer settings
        layer_box = QGroupBox("Add Block")
        layer_form = QFormLayout(layer_box)

        self.layer_type = QComboBox()
        self.layer_type.addItems(["Linear", "ReLU", "Sigmoid", "Tanh"])
        self.layer_type.currentTextChanged.connect(self.update_layer_inputs)

        self.in_features = QSpinBox()
        self.in_features.setRange(1, 10000)
        self.in_features.setValue(2)

        self.out_features = QSpinBox()
        self.out_features.setRange(1, 10000)
        self.out_features.setValue(8)

        self.activation_hint = QComboBox()
        self.activation_hint.addItems(["None", "relu", "leaky_relu", "sigmoid", "tanh", "linear"])

        self.init_method = QComboBox()
        self.init_method.addItems(["None", "xavier", "he"])

        self.distribution = QComboBox()
        self.distribution.addItems(["None", "normal", "uniform"])

        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.0, 100.0)
        self.gain.setSingleStep(0.1)
        self.gain.setValue(1.0)

        layer_form.addRow("Block Type", self.layer_type)
        layer_form.addRow("in_features", self.in_features)
        layer_form.addRow("out_features", self.out_features)
        layer_form.addRow("activation hint", self.activation_hint)
        layer_form.addRow("init", self.init_method)
        layer_form.addRow("distribution", self.distribution)
        layer_form.addRow("gain", self.gain)

        add_btn = QPushButton("Add to Cart")
        add_btn.clicked.connect(self.add_layer)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_layer)

        clear_btn = QPushButton("Clear Cart")
        clear_btn.clicked.connect(self.clear_layers)

        left_panel.addWidget(layer_box)
        left_panel.addWidget(add_btn)
        left_panel.addWidget(delete_btn)
        left_panel.addWidget(clear_btn)

        # Training components
        train_box = QGroupBox("Training Components")
        train_form = QFormLayout(train_box)

        self.loss_type = QComboBox()
        self.loss_type.addItems(["MSELoss"])

        self.optimizer_type = QComboBox()
        self.optimizer_type.addItems(["GD", "SGD", "Adam"])

        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.000001, 10.0)
        self.lr.setDecimals(6)
        self.lr.setSingleStep(0.001)
        self.lr.setValue(0.01)

        train_form.addRow("Loss", self.loss_type)
        train_form.addRow("Optimizer", self.optimizer_type)
        train_form.addRow("Learning Rate", self.lr)

        left_panel.addWidget(train_box)

        save_btn = QPushButton("Save model_config.json")
        save_btn.clicked.connect(self.save_config)
        left_panel.addWidget(save_btn)

        left_panel.addStretch()

        # Cart list
        right_panel.addWidget(QLabel("Layer Cart"))

        self.layer_list = QListWidget()
        right_panel.addWidget(self.layer_list)

        right_panel.addWidget(QLabel("Generated Config"))

        self.config_view = QTextEdit()
        self.config_view.setReadOnly(True)
        right_panel.addWidget(self.config_view)

        self.update_layer_inputs()
        self.refresh_ui()

    def update_layer_inputs(self):
        is_linear = self.layer_type.currentText() == "Linear"

        self.in_features.setEnabled(is_linear)
        self.out_features.setEnabled(is_linear)
        self.activation_hint.setEnabled(is_linear)
        self.init_method.setEnabled(is_linear)
        self.distribution.setEnabled(is_linear)
        self.gain.setEnabled(is_linear)

    def add_layer(self):
        layer_type = self.layer_type.currentText()

        layer = {"type": layer_type}

        if layer_type == "Linear":
            layer["in_features"] = self.in_features.value()
            layer["out_features"] = self.out_features.value()

            if self.activation_hint.currentText() != "None":
                layer["activation"] = self.activation_hint.currentText()

            if self.init_method.currentText() != "None":
                layer["init"] = self.init_method.currentText()

            if self.distribution.currentText() != "None":
                layer["distribution"] = self.distribution.currentText()

            layer["gain"] = self.gain.value()

        self.layers.append(layer)
        self.refresh_ui()

    def delete_layer(self):
        row = self.layer_list.currentRow()
        if row >= 0:
            self.layers.pop(row)
            self.refresh_ui()

    def clear_layers(self):
        self.layers.clear()
        self.refresh_ui()

    def build_config(self):
        return {
            "model": {
                "type": "Sequential",
                "layers": self.layers
            },
            "loss": {
                "type": self.loss_type.currentText()
            },
            "optimizer": {
                "type": self.optimizer_type.currentText(),
                "lr": self.lr.value()
            }
        }

    def refresh_ui(self):
        self.layer_list.clear()

        for i, layer in enumerate(self.layers):
            self.layer_list.addItem(f"{i}. {layer}")

        self.config_view.setPlainText(
            json.dumps(self.build_config(), indent=2, ensure_ascii=False)
        )

        self.canvas.update()

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config",
            "model_config.json",
            "JSON Files (*.json)"
        )

        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.build_config(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralCartBuilder()
    window.show()
    sys.exit(app.exec())