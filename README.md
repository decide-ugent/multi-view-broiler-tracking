# Multi-View Broiler Tracking

![Multi-View Tracking](visualizations/camera_setup_animation.gif)

Welcome to the repository for **Multi-camera Detection and Tracking for Individual Broiler Monitoring**. This project is designed to advance poultry research by offering robust multi-view tracking systems that improve upon traditional single-camera methods. The proposed pipeline enables comprehensive localization and tracking of broilers over their entire lifespan, offering a significant step forward in animal welfare monitoring and research.

## 🖼️ Visualization & Dataset Access
Visualization scripts are available to explore different subsets of the MVBroTrack dataset:
- `visualize_single_view_dataset.py` for single-view detections
- `visualize_ground_plane_detection_dataset.py` for ground-plane detections
- `visualize_ground_plane_tracking_dataset.py` for ground-plane tracks

These scripts illustrate how camera calibrations are loaded and document the formats of bounding boxes and tracking annotations used in the dataset.

The following packages are required for the visualization: opencv-python, tqdm, colorhash, pandas

**Dataset download**: wget -O MVBroTrack_v2.zip https://cloud.ilabt.imec.be/index.php/s/kECieoDK79Ms7ck/download

## Reference 
If you found this dataset or the accompanying code useful, please cite the following paper:

```bibtex
@article{CARDOEN2025110435,
  title   = {Multi-camera detection and tracking for individual broiler monitoring},
  journal = {Computers and Electronics in Agriculture},
  volume  = {237},
  pages   = {110435},
  year    = {2025},
  issn    = {0168-1699},
  doi     = {https://doi.org/10.1016/j.compag.2025.110435},
  url     = {https://www.sciencedirect.com/science/article/pii/S0168169925005411},
  author  = {Thorsten Cardoen and Patricia Soster {de Carvalho} and Gunther Antonissen and Frank A.M. Tuyttens and Sam Leroux and Pieter Simoens},
  keywords = {Multi-camera detection, Object tracking, Broiler welfare monitoring, Sensor fusion}
}
```

## 🔍 Overview
This repository will host:
- **Code**:
  - Object detection training
  - Multi-view detection
  - Multi-view tracking

- **MVBroTrack dataset**:
  - Multi-view dataset of broilers annotated for various subtasks
  - Calibration parameters for the multi-view setup

Both the code and dataset are currently being prepared and will be released soon.

---

## Installation

```bash
conda create -n mvbroilertracking python=3.10
./install.sh
```

## 📚 MVBroTrack Dataset
The MVBroTrack dataset includes annotations for single-view detection, multi-view detection, and multi-view tracking tasks, covering different stages of broiler growth. Below is a summary of the dataset:

### Labels per Subtask
| Task                   | Stage    | Frames       | Broilers | Image Plane (BBoxes) | Ground Plane (Points) | Ground Plane (Tracks) |
|-------------------------|----------|--------------|----------|-----------------------|------------------------|------------------------|
| Single-view detection   | Starter  | 56           | 4106     | 4106                 | -                      | -                      |
|                         | Grower   | 65           | 5714     | 5714                 | -                      | -                      |
|                         | Finisher | 114          | 11414    | 11414                | -                      | -                      |
|                         | **Total**| **235**      | **21234**| **21234**            | -                      | -                      |
| Multi-view detection    | Starter  | 7x4          | 893      | -                    | 893                    | -                      |
|                         | Grower   | 21x4         | 2791     | -                    | 2791                  | -                      |
|                         | Finisher | 23x4         | 3022     | -                    | 3022                  | -                      |
|                         | **Total**| **51x4**     | **6706** | -                    | **6706**              | -                      |
| Multi-view tracking     | Starter  | 1423x4       | 275      | -                    | -                      | 275                    |
|                         | Grower   | 1494x4       | 276      | -                    | -                      | 276                    |
|                         | Finisher | 1439x4       | 265      | -                    | -                      | 265                    |
|                         | **Total**| **4356x4**   | **816**  | -                    | -                      | **816**                |

### 📸 Single-view Ground Truth Detections
![Single-view GT detections](visualizations/bounding_box_example.png)

### 🎥 Multi-View Ground Truth Tracks Example
Below we show one camera view out of the four synchronized views, with ground plane tracks visualized using up to 12 seconds of track history.

![Multi-View GT tracks](visualizations/dataset_tracking_example.gif)

---

## Tracking Performance
Our proposed **tracking-by-curve-matching (TBCM)** method improves upon traditional tracking-by-detection (TBD) techniques. Below are the results:

| Method          | IDF1 ↑  | Recall ↑  | Precision ↑  | Mostly Tracked (MT) ↑ | Mostly Lost (ML) ↓ | ID Switches (IDs) ↓ | Fragmentations (FM) ↓ | MOTA ↑  | MOTP ↓  |
|------------------|---------|-----------|--------------|------------------------|--------------------|---------------------|-----------------------|---------|---------|
| TBD              | 74.7    | 81.6      | **92.8**     | 71.6                  | 7.3                | 912                 | 2852                  | **74.9**| 3.809   |
| TBCM (SORT)      | 80.3    | 83.3      | 90.6         | 76.0                  | 7.3                | 530                 | 1955                  | 74.5    | **3.585**|
| TBCM (muSSP)     | **80.8**| **85.1**  | 89.1         | **78.5**              | **7.1**            | **404**             | **1917**              | 74.5    | 3.597   |
