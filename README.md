# Fusionwatch-idex-
AI-Native Satellite Intelligence System — iDEX Challenge #4
## ⚠️ Publication Notice

T# FusionWatch v0.1
### AI-Native Satellite Intelligence System — iDEX Challenge #4

![FusionWatch Detection](demo/P2239_fw100ep.jpg)

Best pt-- https://drive.google.com/file/d/1r3139Dvb3GLn988VkOwMxTzsDk6a-1HK/view?usp=drive_link
---

## Live Detection Results

**Military airfield — 34 aircraft detected (0.82–0.90 confidence)**
![Airfield](demo/P2239_fw100ep.jpg)

**Aircraft parking apron — 59 detections**
![Apron](demo/P0170_fw100ep.jpg)

**Open ocean — 72 vessels detected**
![Ships](demo/P2330_fw100ep.jpg)

**Naval harbor — ship + harbor infrastructure**
![Harbor](demo/P0300_fw100ep.jpg)

---

## Model Performance (v0.1 — 100 epochs)

| Metric | Score |
|--------|-------|
| mAP50 (all classes) | 0.546 |
| mAP50-95 | 0.309 |
| Precision | 0.781 |
| Recall | 0.491 |
| Inference speed | 17.2ms / image |
| Training time | 1.85 hours |
| Training data | 926 images, DOTA v1.5 |

### Defence-relevant classes

| Class | mAP50 |
|-------|-------|
| Aircraft | 0.783 |
| Ship | 0.644 |
| Large vehicle | 0.757 |
| Harbor | 0.600 |
| Helicopter | 0.619 |

---

## System Architecture

Five-layer intelligence pipeline:
```
Satellite Data (Sentinel-1 SAR + Cartosat-3 Optical)
        ↓
SAR–Optical Fusion (SegFormer dual-encoder)
        ↓
Object Detection (YOLOv8m fine-tuned on DOTA v1.5)
        ↓
Geospatial Conversion (pixel → GPS via GDAL affine transform)
        ↓
Change Detection + Risk Scoring + UAV Cueing + QR Alert
        ↓
Intelligence Dashboard (React + Mapbox GL JS)
```

---

## Repository Structure
```
fusionwatch-idex/
├── day1_setup.py              # DOTA dataset preprocessing
├── day2_train.py              # YOLOv8 training pipeline  
├── day3_change_detection.py   # Temporal change detection
├── demo/                      # Detection output images
│   ├── P2239_fw100ep.jpg      # Military airfield
│   ├── P0170_fw100ep.jpg      # Aircraft apron
│   ├── P2330_fw100ep.jpg      # 72 ships detected
│   └── P0300_fw100ep.jpg      # Naval harbor
└── README.md
```

---

## ⚠️ Publication Notice

This repository contains FusionWatch v0.1 — a proof-of-concept 
trained on publicly available DOTA v1.5 dataset (926 images, 
civilian aerial imagery).

**Why a limited release:** The full FusionWatch system integrates 
Indian satellite data (Cartosat-3, RISAT-2B), SAR-optical fusion, 
and military-specific training data. For operational security 
reasons, only the baseline detection prototype is published here.

The complete system requires:
- ISRO Cartosat-3 / RISAT-2B data access
- Dedicated GPU inference infrastructure
- Military-labelled training dataset
- UAV ground control system integration

All of which are contingent on iDEX SPARK Grant approval.

---

## Key Differentiators

- **All-weather capability** — SAR fusion works through cloud, 
  rain, and night (optical-only systems are blind)
- **Indian satellite first** — primary integration with 
  Cartosat-3 and RISAT-2B, not foreign commercial providers
- **Closed-loop confirmation** — autonomous UAV cueing on 
  high-confidence detections
- **QR-coded alerts** — GPS coordinates delivered as 
  scannable QR codes for field use
- **Edge deployable** — INT8 quantised models run on 
  NVIDIA Jetson without cloud connectivity

---

## Submitted To

**Innovations for Defence Excellence (iDEX)**  
Challenge #4 — AI Based Satellite Image Analysis  
Ministry of Defence, Government of India

**Applicant:** Adarsh T Swathiraj  
**Contact:** adarshswathiraj@gmail.com
