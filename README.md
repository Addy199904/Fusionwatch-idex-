T# FusionWatch v0.1
### AI-Native Satellite Intelligence System — iDEX Challenge #4

![FusionWatch Detection](demo/P2239_fw100ep.jpg)

Best pt-- https://drive.google.com/file/d/1r3139Dvb3GLn988VkOwMxTzsDk6a-1HK/view?usp=drive_link
---

# FusionWatch — AI-Native Satellite Intelligence Infrastructure

![TRL Status](https://img.shields.io/badge/Status-TRL--4%20Prototype-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

**iDEX Challenge #4 SPARK Grant Submission (Individual Innovator: Adarsh T Swathiraj)**

FusionWatch is an edge-deployable, AI-native C4ISR architecture designed to reduce satellite intelligence analysis time from hours to under 20 seconds. It features a Dual-Model (EDSR + YOLOv8) pipeline to overcome the 10m GSD Domain Resolution Gap, precise MGRS coordinate extraction, and automated NATO 9-Line Close Air Support generation.

> **⚠️ EVALUATOR NOTICE:** This repository contains the structural C4ISR architecture, UI framework, and pipeline logic. Proprietary trained neural weights (`best.pt`) and classified API endpoints are omitted for operational security. The pipeline can be validated using the included public Sentinel-2 sample and the built-in Tactical Target Simulator.

## 🏗️ System Architecture (6-Module Pipeline)
*(Insert Architecture Diagram Here: `![Architecture](assets/architecture_diagram.png)`)*

1. **`data_ingest.py`**: GeoTIFF ingestion, Affine transform extraction, and EDSR super-resolution preprocessing.
2. **`ai_engine.py`**: YOLOv8m inference with SAHI (Sliced Aided Hyper Inference) logic.
3. **`context_analyst.py`**: Pixel-to-GPS/MGRS translation and AOI geofence culling.
4. **`intel_export.py`**: NATO 9-Line auto-generation and JSON intelligence packaging.
5. **`map_engine.py`**: Leaflet.js tactical map and Python-to-JavaScript bridge.
6. **`ui_terminal.py`**: Offline-capable PySide6 IMINT workstation desktop application.
7. **`behaviour_engine.py`**: *(Stub)* Contextual threat scoring gated behind MoD MoU (Phase 4).

## 🚀 Quick Start Guide
See `INSTALL.md` for detailed environment setup instructions.
```bash
# 1. Clone the repository
git clone [https://github.com/Addy199904/Fusionwatch-idex-.git](https://github.com/Addy199904/Fusionwatch-idex-.git)
cd Fusionwatch-idex-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Tactical IMINT Terminal
python ui_terminal.py
