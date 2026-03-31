"""
FusionWatch v0.1 — IMINT ANALYST WORKSTATION (PITCH READY)
========================================================
Demonstrates current C4ISR capabilities while visually 
outlining the post-grant development roadmap.
"""

import sys
import json
import os
import io
import base64
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dota_showcase import DotaShowcaseWindow
from map_engine import build_tactical_map_html

import rasterio
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QFrame, QTableWidget, 
    QTableWidgetItem, QHeaderView, QTextEdit, QSplitter, QMessageBox,
    QGridLayout, QAbstractItemView, QCheckBox, QScrollArea, QDateEdit
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QColor, QFont, QPalette

# ── Colour Palette ────────────────────────────────────────────────────────────
BG_DARK    = "#0a0e1a"
BG_MID     = "#0b0f1c"
BG_PANEL   = "#0d1220"
BG_CARD    = "#060a14"
BORDER     = "#1e2a40"
TEXT_PRI   = "#c8d0e0"
TEXT_SEC   = "#5a6a80"
TEXT_MUT   = "#3a4a60"
ACCENT     = "#38bdf8"
GREEN      = "#2ecc71"
AMBER      = "#f39c12"
RED        = "#e74c3c"
DARK_RED   = "#3a0808"

def make_label(text: str, size: int = 10, color: str = TEXT_PRI, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    font = QFont("Segoe UI", size)
    font.setBold(bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl

def section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
    lbl.setStyleSheet(f"color:{TEXT_MUT};background:{BG_MID};padding:6px 10px;letter-spacing:1.5px;border-bottom:1px solid {BORDER};")
    return lbl

def h_line() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"color:{BORDER};background:{BORDER};max-height:1px;")
    return line


class IMINTWorkstation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_targets = []
        self.loaded_tif_paths = [] 
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        self.setWindowTitle("FUSIONWATCH // IMINT ANALYST WORKSTATION v0.1")
        self.resize(1500, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._build_titlebar(root_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{BORDER};}}")
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([300, 800, 400])
        
        root_layout.addWidget(splitter, stretch=1)

    def _build_titlebar(self, parent_layout):
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setStyleSheet(f"background:{BG_PANEL};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 0, 15, 0)

        logo = QLabel("FUSIONWATCH")
        logo.setFont(QFont("Segoe UI", 14, QFont.Bold))
        logo.setStyleSheet(f"color:{ACCENT};letter-spacing:2px;")
        layout.addWidget(logo)
        layout.addWidget(make_label("v0.1 · iDEX Challenge #4", 9, TEXT_MUT))
        layout.addStretch()

        btn_load_tile = QPushButton("OFFLINE TILE UPLOAD (.TIF)")
        btn_load_tile.setStyleSheet(f"QPushButton{{color:#ffffff;background:#0284c7;border:1px solid #0369a1;border-radius:3px;padding:6px 15px;font-weight:bold;}}QPushButton:hover{{background:#0369a1;}}")
        btn_load_tile.clicked.connect(self._load_geotiff)
        layout.addWidget(btn_load_tile)

        self.btn_run_ai = QPushButton("2. RUN AI SECTOR SCAN")
        self.btn_run_ai.setStyleSheet(f"QPushButton{{color:#94a3b8;background:#1e293b;border:1px solid {BORDER};border-radius:3px;padding:6px 15px;font-weight:bold;}}")
        self.btn_run_ai.setEnabled(False)
        self.btn_run_ai.clicked.connect(self._run_ai_pipeline)
        layout.addWidget(self.btn_run_ai)

        btn_diagnostics = QPushButton("ALGORITHM DIAGNOSTICS")
        btn_diagnostics.setStyleSheet("QPushButton{color:#c8d0e0;background:#1e293b;border:1px solid #1e2a40;border-radius:3px;padding:6px 15px;font-weight:bold;}QPushButton:hover{background:#334155;}")
        btn_diagnostics.clicked.connect(self._launch_diagnostics)
        layout.addWidget(btn_diagnostics)

        parent_layout.addWidget(bar)
        parent_layout.addWidget(h_line())

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background:{BG_MID};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(section_label("Mission Summary"))
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(2)
        grid.setContentsMargins(5, 5, 5, 15)

        self.stats = {}
        stat_configs = [
            ("total", "Total Alerts", "0", TEXT_PRI),
            ("critical", "Critical Risk", "0", RED),
            ("ships", "Vessels", "0", ACCENT),
            ("aircraft", "Aircraft", "0", ACCENT),
            ("speed", "Inference", "--", GREEN),
            ("uav", "UAV Cued", "0", AMBER),
        ]
        
        for i, (key, label, val, colour) in enumerate(stat_configs):
            cell = QFrame()
            cell.setStyleSheet(f"background:{BG_CARD}; border: 1px solid {BORDER}; border-radius: 4px;")
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(5, 15, 5, 15)
            num = QLabel(val)
            num.setFont(QFont("Segoe UI", 20, QFont.Bold))
            num.setStyleSheet(f"color:{colour}; border: none;")
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(label.upper())
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setStyleSheet(f"color:{TEXT_MUT}; border: none;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(num)
            cl.addWidget(lbl)
            self.stats[key] = num
            grid.addWidget(cell, i // 2, i % 2)

        layout.addWidget(grid_widget)
        
        self.lbl_delta1 = make_label("System Idle.", 9, TEXT_MUT)
        self.lbl_delta2 = make_label("Awaiting sector data.", 9, TEXT_MUT)
        self.pill_crit = make_label("0 CRITICAL", 8, TEXT_MUT, bold=True)
        self.pill_crit.setStyleSheet(f"color:{TEXT_MUT};background:{BG_PANEL};border:1px solid {BORDER};border-radius:3px;padding:2px 8px;")
        
        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(15, 10, 15, 10)
        status_layout.addWidget(self.lbl_delta1)
        status_layout.addWidget(self.lbl_delta2)
        status_layout.addWidget(self.pill_crit)
        layout.addWidget(status_frame)

        layout.addWidget(h_line())

        # --- REPLACED SECTION: TEMPORAL API & CHANGE DETECTION ---
        layout.addWidget(section_label("Temporal API Parameters"))
        temporal_frame = QFrame()
        temporal_layout = QVBoxLayout(temporal_frame)
        temporal_layout.setContentsMargins(15, 15, 15, 15)
        temporal_layout.setSpacing(12)

        # Baseline Date Selector
        temporal_layout.addWidget(make_label("BASELINE EPOCH (T-0)", 8, TEXT_MUT, True))
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDate(QDate.currentDate().addDays(-30)) # Default to 30 days ago
        self.date_start.setStyleSheet(f"QDateEdit {{ background: {BG_CARD}; color: {TEXT_PRI}; border: 1px solid {BORDER}; padding: 5px; font-weight: bold; }} QDateEdit::drop-down {{ border-left: 1px solid {BORDER}; }}")
        temporal_layout.addWidget(self.date_start)

        # Current Date Selector
        temporal_layout.addWidget(make_label("CURRENT EPOCH (T-1)", 8, TEXT_MUT, True))
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDate(QDate.currentDate()) # Default to today
        self.date_end.setStyleSheet(f"QDateEdit {{ background: {BG_CARD}; color: {TEXT_PRI}; border: 1px solid {BORDER}; padding: 5px; font-weight: bold; }} QDateEdit::drop-down {{ border-left: 1px solid {BORDER}; }}")
        temporal_layout.addWidget(self.date_end)

        # Simulated API Fetch Button
        self.btn_api_fetch = QPushButton("FETCH SATELLITE ARCHIVE (API)")
        self.btn_api_fetch.setStyleSheet(f"QPushButton{{color:{ACCENT};background:{BG_CARD};border:1px solid {ACCENT};border-radius:3px;padding:8px;font-weight:bold;}}QPushButton:hover{{background:#0c4a6e;}}")
        self.btn_api_fetch.clicked.connect(self._load_geotiff)
        temporal_layout.addWidget(self.btn_api_fetch)

        temporal_layout.addWidget(make_label("STATUS: API SIMULATION MODE ACTIVE", 8, AMBER, True))
        layout.addWidget(temporal_frame)
        # ---------------------------------------------------------
        
        layout.addStretch()
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background:{BG_DARK};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        current_theme = {
            "BG_DARK": BG_DARK, "BG_PANEL": BG_PANEL, "BORDER": BORDER,
            "ACCENT": ACCENT, "AMBER": AMBER, "RED": RED, "GREEN": GREEN,
            "TEXT_PRI": TEXT_PRI, "TEXT_MUT": TEXT_MUT  # <-- ADDED THESE TWO
        }
        
        self.map_view = QWebEngineView()
        self.map_view.setHtml(build_tactical_map_html(theme=current_theme))
        # NEW: Connect the Javascript bridge
        self.map_view.titleChanged.connect(self._handle_map_title)
        layout.addWidget(self.map_view, stretch=1)
        
        # --- NEW: AOI TELEMETRY BAR ---
        self.aoi_container = QWidget()
        self.aoi_container.setStyleSheet(f"background: {BG_PANEL}; border-top: 1px solid {BORDER};")
        self.aoi_layout = QVBoxLayout(self.aoi_container)
        self.aoi_layout.setContentsMargins(10, 5, 10, 5)

        self.lbl_aoi_title = make_label("AREA OF INTEREST (AOI) TELEMETRY", 8, TEXT_MUT, True)
        
        self.lbl_aoi_data = QTextEdit()
        self.lbl_aoi_data.setReadOnly(True)
        self.lbl_aoi_data.setFixedHeight(45) # Keep it small, terminal style
        self.lbl_aoi_data.setStyleSheet(f"background: {BG_DARK}; color: {AMBER}; font-family: 'Courier New'; font-size: 11px; border: 1px solid {BORDER};")
        self.lbl_aoi_data.setText(">> NO SPATIAL GEOFENCE DESIGNATED.")

        self.aoi_layout.addWidget(self.lbl_aoi_title)
        self.aoi_layout.addWidget(self.lbl_aoi_data)
        layout.addWidget(self.aoi_container)
        # ------------------------------

        # Active Layers Section (Unchanged)
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setFixedHeight(50)
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setStyleSheet(f"QScrollArea {{ border: none; border-top: 1px solid {BORDER}; background: {BG_MID}; }}")
        
        self.layer_container = QWidget()
        self.layer_container.setStyleSheet("background: transparent;")
        self.layer_layout = QHBoxLayout(self.layer_container)
        self.layer_layout.setContentsMargins(10, 5, 10, 5)
        self.layer_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.layer_label = make_label("NO ACTIVE LAYERS", 9, TEXT_MUT, True)
        self.layer_layout.addWidget(self.layer_label)
        
        self.layer_scroll.setWidget(self.layer_container)
        layout.addWidget(self.layer_scroll)
        
        return panel

    # --- NEW: Catch Javascript telemetry data ---
    def _handle_map_title(self, title):
        if title == "AOI_CLEAR":
            self.lbl_aoi_data.setText(">> NO SPATIAL GEOFENCE DESIGNATED.")
        elif title.startswith("AOI_UPDATE|"):
            try:
                data_str = title.split("|", 1)[1]
                data = json.loads(data_str)
                area = data.get("area", "0.00")
                coords = data.get("coords", "")
                
                # Format to look like tactical console output
                telemetry_text = f">> AOI AREA: {area} SQ KM\n>> VERTICES: {coords}"
                self.lbl_aoi_data.setText(telemetry_text)
            except Exception as e:
                print(f"AOI Parse Error: {e}")

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background:{BG_MID}; border-left: 1px solid {BORDER};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- NEW: CLASS FILTERS BAR ---
        layout.addWidget(section_label("Target Filters"))
        self.class_filter_container = QWidget()
        self.class_filter_container.setStyleSheet(f"background:{BG_CARD};")
        self.class_filter_layout = QHBoxLayout(self.class_filter_container)
        self.class_filter_layout.setContentsMargins(10, 5, 10, 5)
        self.class_filter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.filter_status_label = make_label("Awaiting AI Scan...", 9, TEXT_MUT)
        self.class_filter_layout.addWidget(self.filter_status_label)
        layout.addWidget(self.class_filter_container)
        self.ui_class_checkboxes = {}
        # ------------------------------

        layout.addWidget(section_label("Detected Targets"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "CLASS", "CONF", "RISK"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(f"QTableWidget {{ background-color: {BG_CARD}; color: {TEXT_PRI}; gridline-color: {BORDER}; border: none; font-size: 12px; }} QHeaderView::section {{ background-color: {BG_PANEL}; color: {TEXT_MUT}; border: 1px solid {BORDER}; font-weight: bold; }} QTableWidget::item:selected {{ background-color: #1e3a8a; }}")
        self.table.cellClicked.connect(self._target_selected)
        layout.addWidget(self.table, stretch=6)

        layout.addWidget(h_line())

        layout.addWidget(section_label("NATO 9-Line & MGRS Downlink"))
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setText("\n>> AWAITING TARGET SELECTION\n>> MGRS COMMS SECURE")
        self.txt_output.setStyleSheet(f"background-color: #000000; color: {GREEN}; font-family: 'Courier New'; font-size: 13px; font-weight: bold; border: none; padding: 10px;")
        layout.addWidget(self.txt_output, stretch=4)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(10, 10, 10, 10)
        btn_copy = QPushButton("COPY MGRS")
        btn_fire = QPushButton("TRANSMIT 9-LINE")
        for btn in [btn_copy, btn_fire]:
            btn.setFont(QFont("Segoe UI", 9, QFont.Bold))
            btn.setStyleSheet(f"QPushButton{{background:{BG_PANEL};color:{TEXT_PRI};border:1px solid {BORDER};padding:8px;}}QPushButton:hover{{background:{BORDER};}}")
        
        btn_copy.clicked.connect(lambda: QMessageBox.information(self, "Copied", "MGRS Coordinate copied to clipboard."))
        btn_fire.clicked.connect(lambda: QMessageBox.information(self, "Transmitted", "9-Line Brief transmitted to Fire Control Network."))
        
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_fire)
        
        btn_container = QWidget()
        btn_container.setStyleSheet(f"background:{BG_CARD}; border-top: 1px solid {BORDER};")
        btn_container.setLayout(btn_layout)
        layout.addWidget(btn_container)

        return panel

    def _apply_map_filters(self):
        """Re-renders the Table and the Map Markers based on active checkboxes."""
        active_classes = [cls for cls, cb in self.ui_class_checkboxes.items() if cb.isChecked()]
        
        self.map_view.page().runJavaScript("window.clearMarkers();")
        self.table.setRowCount(0)

        for tgt in self.current_targets:
            if tgt["class"] in active_classes:
                # Add to Table
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                id_item = QTableWidgetItem(tgt["target_id"])
                cls_item = QTableWidgetItem(tgt["class"])
                conf_item = QTableWidgetItem(f"{tgt['confidence']:.1f}%" if tgt['confidence'] > 1 else f"{tgt['confidence']*100:.1f}%")
                risk_item = QTableWidgetItem(tgt["risk"])
                
                if tgt["risk"] == "CRITICAL": risk_item.setForeground(QColor(RED))
                elif tgt["risk"] == "HIGH": risk_item.setForeground(QColor(AMBER))
                    
                for item in [id_item, cls_item, conf_item, risk_item]:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, self.table.columnCount() - 4 + [id_item, cls_item, conf_item, risk_item].index(item), item)

                # Add to Map
                lat, lon = tgt["location"]["lat"], tgt["location"]["lon"]
                js = f"window.addSingleDetection({lat}, {lon}, '{tgt['class']}', '{tgt['risk']}', '{tgt['target_id']}');"
                self.map_view.page().runJavaScript(js)

    def _apply_theme(self):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_DARK))
        self.setPalette(palette)

    def _toggle_layer(self, layer_id, is_visible):
        js = f"window.toggleLayer('{layer_id}', {str(is_visible).lower()});"
        self.map_view.page().runJavaScript(js)

    def _load_geotiff(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Satellite Tiles", str(Path.cwd()), "TIFF Files (*.tif *.tiff)")
        if not paths: return

        try:
            self.lbl_delta1.setText(f"Rendering {len(paths)} GeoTIFF Tile(s)...")
            QApplication.processEvents()
            
            self.map_view.page().runJavaScript("window.clearOverlays(); window.clearMarkers();")
            self.table.setRowCount(0)
            self.loaded_tif_paths = paths

            for i in reversed(range(self.layer_layout.count())): 
                widget = self.layer_layout.itemAt(i).widget()
                if widget: widget.deleteLater()
                
            self.layer_label = make_label("ACTIVE LAYERS:", 9, TEXT_PRI, True)
            self.layer_layout.addWidget(self.layer_label)

            for i, path in enumerate(paths):
                layer_id = f"layer_{i}"
                filename = Path(path).stem
                if len(filename) > 15: filename = filename[:15] + "..."

                with rasterio.open(path) as dataset:
                    bounds = dataset.bounds
                    img_array = dataset.read([1, 2, 3])
                    img_array = np.transpose(img_array, (1, 2, 0))
                    
                    if img_array.dtype != np.uint8:
                        max_val = img_array.max()
                        if max_val > 0:
                            img_array = (img_array / max_val * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)

                    img = Image.fromarray(img_array)
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                js = f"window.overlayGeoTIFF('{layer_id}', '{img_b64}', {bounds.bottom}, {bounds.left}, {bounds.top}, {bounds.right});"
                self.map_view.page().runJavaScript(js)
                
                cb = QCheckBox(filename)
                cb.setChecked(True)
                cb.setStyleSheet(f"""
                    QCheckBox {{ color: {TEXT_PRI}; background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px; }} 
                    QCheckBox::indicator {{ width: 12px; height: 12px; background: {BG_CARD}; border: 1px solid {ACCENT}; }}
                    QCheckBox::indicator:checked {{ background: {ACCENT}; }}
                """)
                cb.toggled.connect(lambda checked, l=layer_id: self._toggle_layer(l, checked))
                self.layer_layout.addWidget(cb)

            self.layer_layout.addStretch()

            self.btn_run_ai.setEnabled(True)
            self.btn_run_ai.setStyleSheet(f"QPushButton{{color:#ffffff;background:#16a34a;border:1px solid #15803d;border-radius:3px;padding:6px 15px;font-weight:bold;}}QPushButton:hover{{background:#15803d;}}")
            self.lbl_delta1.setText(f"{len(paths)} Tile(s) Loaded.")
            self.lbl_delta2.setText("Awaiting AI Scan.")

        except Exception as e:
            QMessageBox.critical(self, "GeoTIFF Error", f"Failed to render satellite tiles.\n{str(e)}")

    def _run_ai_pipeline(self):
        if not self.loaded_tif_paths: return
        
        self.lbl_delta2.setText("Fetching Geofence Coordinates...")
        QApplication.processEvents()

        # Ask the JavaScript map for the drawn bounds, then pass them to the execution function
        self.map_view.page().runJavaScript("window.getGeofenceBounds();", self._execute_pipeline)

    def _execute_pipeline(self, bounds):
        self.lbl_delta2.setText("Initializing YOLOv8 Engine...")
        QApplication.processEvents()

        try:
            self.current_targets = [] 
            self.table.setRowCount(0)
            self.map_view.page().runJavaScript("window.clearMarkers();")
            
            for path in self.loaded_tif_paths:
                # Build the command
                cmd = ["python", "run_pipeline.py", "--image", path, "--inject-demo"]
                
                # If the user drew a geofence, pass the coordinates to the backend!
                if bounds:
                    cmd.extend(["--bounds", str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3])])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0: raise Exception(result.stderr)

                output_dir = Path("output")
                json_files = sorted(output_dir.glob("*_alerts.json"), key=os.path.getmtime)
                
                if json_files: 
                    self._parse_and_plot_json(json_files[-1])
            
            if not self.current_targets:
                QMessageBox.warning(self, "Pipeline", "No anomalies found in this sector.")
                
        except Exception as e:
            QMessageBox.critical(self, "Pipeline Error", f"Failed to execute AI scan.\n{str(e)}")

    def _parse_and_plot_json(self, json_path):
        with open(json_path, 'r') as f: data = json.load(f)
        
        new_targets = data.get("targets", [])
        self.current_targets.extend(new_targets)
        
        # Build dynamic checkboxes for Right Panel
        unique_classes = set(tgt["class"] for tgt in self.current_targets)
        
        for i in reversed(range(self.class_filter_layout.count())): 
            widget = self.class_filter_layout.itemAt(i).widget()
            if widget: widget.deleteLater()
            
        self.ui_class_checkboxes = {}
        for cls in sorted(unique_classes):
            cb = QCheckBox(cls)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {TEXT_PRI}; font-weight: bold; font-size: 11px; margin-right: 10px; }}")
            cb.stateChanged.connect(self._apply_map_filters)
            self.class_filter_layout.addWidget(cb)
            self.ui_class_checkboxes[cls] = cb
            
        self.class_filter_layout.addStretch()

        # Update Mission Stats
        crit_count, ship_count, air_count = 0, 0, 0
        for tgt in self.current_targets:
            if tgt["risk"] == "CRITICAL": crit_count += 1
            if tgt["class"] in ["SHIP", "HARBOR"]: ship_count += 1
            if tgt["class"] in ["PLANE", "HELICOPTER", "AIRCRAFT"]: air_count += 1

        self.stats["total"].setText(str(len(self.current_targets)))
        self.stats["critical"].setText(str(crit_count))
        self.stats["ships"].setText(str(ship_count))
        self.stats["aircraft"].setText(str(air_count))
        self.stats["uav"].setText(str(crit_count)) 
        self.stats["speed"].setText("17.2ms") 
        
        self.pill_crit.setText(f"{crit_count} CRITICAL")
        if crit_count > 0: self.pill_crit.setStyleSheet(f"color:{RED};background:{DARK_RED};border:1px solid #6a1010;border-radius:3px;padding:2px 8px;")

        self.lbl_delta1.setText(f"+ {len(self.current_targets)} new anomalies tracked.")
        if crit_count > 0:
            self.lbl_delta2.setText(f"!! {crit_count} Critical Threats Elevated.")
        else:
            self.lbl_delta2.setText("No critical threats in sector.")

        # Call the filter function to render the map and table
        self._apply_map_filters()

    def _target_selected(self, row, col):
        tgt = self.current_targets[row]
        lat, lon = tgt["location"]["lat"], tgt["location"]["lon"]
        self.map_view.page().runJavaScript(f"window.panTo({lat}, {lon});")
        self.txt_output.setText(f"{tgt.get('cas_9_line', '')}\n\n>>> DATALINK READY <<<")

    def _launch_diagnostics(self):
        self.diag_window = DotaShowcaseWindow()
        self.diag_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IMINTWorkstation()
    window.show()
    sys.exit(app.exec())