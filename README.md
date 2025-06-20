# ICH Preprocessing Viewer

This is a graphical application built with PyQt5 and VTK for preprocessing and visualizing brain CT scans stored in NIfTI format. The tool allows users to load NIfTI files, apply windowing, segment brain tissues (including hemorrhage, skull, brain tissue, and cerebrospinal fluid), and visualize 2D slices and 3D renderings.
(Basic)
![basic](https://github.com/user-attachments/assets/bab56332-c92b-4706-9c58-b5661c7ba04d)
(Standard)
![standard](https://github.com/user-attachments/assets/c716a52e-fc69-46ef-961a-c0fdfd2a64e5)
<img width="705" alt="standard2" src="https://github.com/user-attachments/assets/9f87863e-01a7-4eeb-99b1-93e528f7178b" />

## Overview

The `ICH Preprocessing Viewer` is designed for medical imaging analysis, specifically for intracranial hemorrhage (ICH) detection and brain tissue segmentation. It provides an interactive interface to:
- Load and browse NIfTI files from a folder.
- Adjust window level and width for CT visualization.
- Segment and display original, skull-removed, and entropy-thresholded images.
- Generate 3D visualizations of segmented regions (hemorrhage, skull, brain tissue, and CSF).
- Export processed images.

## Features

- **2D Visualization**: Display axial, coronal, and sagittal views with adjustable windowing (level and width).
- **3D Visualization**: Render 3D models of hemorrhage, skull, brain tissue, and CSF using VTK.
- **Preprocessing**: Apply entropy-based thresholding, skull removal, and tissue segmentation.
- **Thumbnail Navigation**: Browse slices via thumbnail previews with clickable navigation.
- **Export Functionality**: Save processed 2D images (original, axial, coronal, sagittal) to a selected directory.
- **Progress Tracking**: Visual progress bar during file loading and processing.

## Installation

### Prerequisites
- Python 3.7 or higher
- Ensure the following Python packages are installed:
  - `numpy`
  - `opencv-python` (cv2)
  - `PyQt5`
  - `vtk`
  - `nibabel`

### Installation Steps
1. Clone or download the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
