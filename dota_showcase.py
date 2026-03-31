import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QSplitter, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont, QColor, QPainter, QPen

from ai_engine import FusionWatchAI
from asset_manager import assets

class DotaShowcaseWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUSIONWATCH // ALGORITHM DIAGNOSTICS SUITE")
        self.resize(1200, 700)
        self.setStyleSheet("background-color: #0b0f1c; color: #c8d0e0; font-family: 'Segoe UI';")
        
        self.ai = FusionWatchAI(assets.get_model_path("best.pt"))
        self.raw_detections = []
        self.original_pixmap = None
        self.class_filters = {}
        
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        header = QHBoxLayout()
        title = QLabel("YOLOv8 ALGORITHM VALIDATION (DYNAMIC FILTERING)")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #38bdf8;")
        
        btn_load = QPushButton("LOAD DOTA IMAGE (.PNG/.JPG)")
        btn_load.setStyleSheet("background-color: #0284c7; color: white; padding: 8px; font-weight: bold;")
        btn_load.clicked.connect(self._run_diagnostics)
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(btn_load)
        layout.addLayout(header)

        # Filters Bar
        self.filter_layout = QHBoxLayout()
        self.filter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(self.filter_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.img_label = QLabel("AWAITING IMAGE...")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background-color: #060a14; border: 1px solid #1e2a40;")
        splitter.addWidget(self.img_label)
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["CLASS", "CONF", "CENTER X", "CENTER Y"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("QTableWidget { background-color: #060a14; color: #c8d0e0; border: 1px solid #1e2a40; } QHeaderView::section { background-color: #0d1220; color: #5a6a80; font-weight: bold; }")
        splitter.addWidget(self.table)
        
        splitter.setSizes([800, 400])
        layout.addWidget(splitter)

    def _run_diagnostics(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DOTA Image", str(Path.cwd()), "Images (*.png *.jpg *.jpeg)")
        if not path: return

        self.img_label.setText("RUNNING YOLOv8 INFERENCE...")
        QApplication.processEvents()

        # Run inference
        self.raw_detections, speed = self.ai.scan_raw(path)
        self.original_pixmap = QPixmap(path)

        # Build dynamic checkboxes
        unique_classes = set(tgt["class"] for tgt in self.raw_detections)
        
        # Clear old checkboxes
        for i in reversed(range(self.filter_layout.count())): 
            widget = self.filter_layout.itemAt(i).widget()
            if widget: widget.deleteLater()

        self.class_filters = {}
        for cls in sorted(unique_classes):
            cb = QCheckBox(cls)
            cb.setChecked(True)
            cb.setStyleSheet("QCheckBox { color: #c8d0e0; font-weight: bold; margin-right: 15px; }")
            cb.stateChanged.connect(self._apply_filters)
            self.filter_layout.addWidget(cb)
            self.class_filters[cls] = cb

        self._apply_filters()

    def _apply_filters(self):
        if not self.original_pixmap: return
        
        active_classes = [cls for cls, cb in self.class_filters.items() if cb.isChecked()]

        # 1. Re-draw Table
        self.table.setRowCount(0)
        for tgt in self.raw_detections:
            if tgt["class"] in active_classes:
                row = self.table.rowCount()
                self.table.insertRow(row)
                c_item = QTableWidgetItem(tgt["class"])
                conf_item = QTableWidgetItem(f"{tgt['confidence']*100:.1f}%")
                x_item = QTableWidgetItem(str(tgt["pixel_coords"]["cx"]))
                y_item = QTableWidgetItem(str(tgt["pixel_coords"]["cy"]))
                
                for item in [c_item, conf_item, x_item, y_item]:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, [c_item, conf_item, x_item, y_item].index(item), item)

        # 2. Re-draw Image Boxes
        canvas = self.original_pixmap.copy()
        painter = QPainter(canvas)
        
        for tgt in self.raw_detections:
            if tgt["class"] in active_classes:
                # Color code boxes
                color = QColor("#38bdf8") if tgt["class"] in ["SMALL-VEHICLE", "LARGE-VEHICLE"] else QColor("#e74c3c")
                if tgt["class"] == "SHIP": color = QColor("#f39c12")
                
                painter.setPen(QPen(color, max(2, canvas.width()//500)))
                
                x1, y1 = tgt["pixel_coords"]["x1"], tgt["pixel_coords"]["y1"]
                w = tgt["pixel_coords"]["x2"] - x1
                h = tgt["pixel_coords"]["y2"] - y1
                painter.drawRect(int(x1), int(y1), int(w), int(h))

        painter.end()
        self.img_label.setPixmap(canvas.scaled(self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))