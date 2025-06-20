import sys
import cv2
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QTabWidget, QDial, QGroupBox,
                             QFormLayout, QProgressBar, QListWidget, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsOpacityEffect
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import nibabel as nib
import SimpleITK as sitk

def window_ct(ct_scan, w_level=40, w_width=120):
    """Apply windowing to CT scan to enhance visualization."""
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    if len(ct_scan.shape) == 3:
        num_slices = ct_scan.shape[2]
        for s in range(num_slices):
            slice_s = ct_scan[:, :, s].copy()
            slice_s = (slice_s - w_min) * (255 / (w_max - w_min))
            slice_s[slice_s < 0] = 0
            slice_s[slice_s > 255] = 255
            ct_scan[:, :, s] = slice_s
    else:
        windowed = (ct_scan - w_min) * (255 / (w_max - w_min))
        windowed[windowed < 0] = 0
        windowed[windowed > 255] = 255
        ct_scan = windowed
    return ct_scan

def calculate_entropy(hist, total_pixels):
    """Calculate entropy from histogram."""
    prob = hist / total_pixels
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def threshold_entropy(img):
    """Entropy-based thresholding."""
    hist, bins = np.histogram(img[img > 0], bins=256, range=[0, 256])
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return np.zeros_like(img)
    max_entropy = 0
    optimal_threshold = 0
    for t in range(1, 255):
        background = hist[:t]
        background_sum = np.sum(background)
        foreground = hist[t:]
        foreground_sum = np.sum(foreground)
        if background_sum == 0 or foreground_sum == 0:
            continue
        background_entropy = calculate_entropy(background, background_sum)
        foreground_entropy = calculate_entropy(foreground, foreground_sum)
        combined_entropy = background_entropy + foreground_entropy
        if combined_entropy > max_entropy:
            max_entropy = combined_entropy
            optimal_threshold = t
    _, binary = cv2.threshold(img, optimal_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def apply_entropy_and_morphology(img):
    """Apply entropy-based thresholding and morphological operations."""
    img = img.astype(np.uint8)
    try:
        thresh_value = threshold_entropy(img)
        _, binary = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    except Exception as e:
        print(f"Error in entropy thresholding: {e}. Using Otsu instead.")
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def remove_skull(img):
    """Remove skull regions from brain image and return skull mask."""
    img = img.astype(np.uint8)
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    _, skull_thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    brain_mask = cv2.bitwise_and(thresh, cv2.bitwise_not(skull_thresh))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(brain_mask, connectivity=8)
    if num_labels > 1:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        brain_mask = np.zeros_like(brain_mask)
        brain_mask[labels == largest_label] = 255
    skull_mask = skull_thresh
    skull_mask = cv2.morphologyEx(skull_mask, cv2.MORPH_CLOSE, kernel)
    skull_mask = cv2.morphologyEx(skull_mask, cv2.MORPH_OPEN, kernel)
    skull_removed = cv2.bitwise_and(img, img, mask=brain_mask)
    return skull_removed, brain_mask, skull_mask

def segment_brain_tissue(img):
    """Segment brain tissue (gray and white matter) based on intensity."""
    brain_tissue = np.zeros_like(img, dtype=np.uint8)
    brain_tissue[(img >= 20) & (img <= 50)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brain_tissue = cv2.morphologyEx(brain_tissue, cv2.MORPH_CLOSE, kernel)
    brain_tissue = cv2.morphologyEx(brain_tissue, cv2.MORPH_OPEN, kernel)
    return brain_tissue

def segment_csf(img):
    """Segment cerebrospinal fluid (CSF) based on intensity."""
    csf = np.zeros_like(img, dtype=np.uint8)
    csf[(img >= 0) & (img <= 15)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    csf = cv2.morphologyEx(csf, cv2.MORPH_CLOSE, kernel)
    csf = cv2.morphologyEx(csf, cv2.MORPH_OPEN, kernel)
    return csf

def read_nifti_file(file_path):
    """Read a NIfTI file and return the 3D volume."""
    try:
        nifti_img = nib.load(file_path)
        volume_data = nifti_img.get_fdata()
        if len(volume_data.shape) > 3:
            volume_data = volume_data[:, :, :, 0]
        volume_data = volume_data.astype(np.int16)
        if nib.aff2axcodes(nifti_img.affine) != ('R', 'A', 'S'):
            print("Note: Converting NIfTI to standard orientation")
            volume_data = nib.as_closest_canonical(nifti_img).get_fdata().astype(np.int16)
        return volume_data
    except Exception as e:
        print(f"Error reading NIfTI file: {e}")
        return None

class ClickableLabel(QLabel):
    """Custom QLabel that emits a signal when clicked."""
    def __init__(self, slice_index, parent=None):
        super().__init__(parent)
        self.slice_index = slice_index

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent().parent().parent().on_thumbnail_clicked(self.slice_index)

class VTKWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.renderer.SetBackground(0.8, 0.8, 0.8)
        self.interactor.Initialize()
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("No data loaded")
        self.text_actor.GetTextProperty().SetColor(0.3, 0.3, 0.3)
        self.text_actor.GetTextProperty().SetFontSize(24)
        self.text_actor.SetPosition(200, 200)
        self.renderer.AddActor(self.text_actor)
        self.volume_text_actor = vtk.vtkTextActor()
        self.vtkWidget.GetRenderWindow().Render()

    def create_3d_from_binary(self, hemorrhage_volume_data, skull_data, brain_tissue_data, csf_data):
        """Create 3D representation with hemorrhage volume, skull, brain tissue, and CSF."""
        self.renderer.RemoveAllViewProps()
        dims = hemorrhage_volume_data.shape
        width, height, depth = dims

        hemorrhage_volume = self.create_volume_data(hemorrhage_volume_data, dims, vtk.VTK_UNSIGNED_CHAR)
        self.add_volume_rendering(hemorrhage_volume, color=(1.0, 0.0, 0.0),
                                  opacity_points=[(0, 0.0), (128, 0.3), (255, 0.8)])

        skull_volume = self.create_volume_data(skull_data, (width, height, 20), vtk.VTK_UNSIGNED_CHAR)
        self.create_3d_from_skull(skull_volume)

        brain_volume = self.create_volume_data(brain_tissue_data, (width, height, 20), vtk.VTK_UNSIGNED_CHAR)
        self.create_3d_from_brain_tissue(brain_volume)

        csf_volume = self.create_volume_data(csf_data, (width, height, 20), vtk.VTK_UNSIGNED_CHAR)
        self.add_volume_rendering(csf_volume, color=(0.0, 0.0, 1.0), opacity_points=[(0, 0.0), (128, 0.2), (255, 0.5)])

        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(vtk.vtkImageData().SetDimensions(width, height, depth))
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddActor(outline_actor)

        self.display_hemorrhage_volume(hemorrhage_volume_data)

        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def display_hemorrhage_volume(self, volume_data):
        """Calculate and display hemorrhage volume in cm³."""
        non_zero_voxels = np.sum(volume_data > 0)
        volume_cm3 = non_zero_voxels / 1000.0
        volume_text = f"Hemorrhage Volume: {volume_cm3:.2f} cm³"
        self.volume_text_actor.SetInput(volume_text)
        self.volume_text_actor.GetTextProperty().SetColor(0.3, 0.3, 0.3)
        self.volume_text_actor.GetTextProperty().SetFontSize(18)
        self.volume_text_actor.SetPosition(10, 10)
        self.renderer.AddActor(self.volume_text_actor)

    def create_volume_data(self, data, dims, scalar_type):
        """Helper to create VTK volume data from 2D/3D image data."""
        width, height, depth = dims
        volume = vtk.vtkImageData()
        volume.SetDimensions(width, height, depth)
        volume.SetSpacing(1.0, 1.0, 1.0)
        volume.SetOrigin(0.0, 0.0, 0.0)
        volume.AllocateScalars(scalar_type, 1)
        if len(data.shape) == 2:
            volume_data = np.zeros(dims, dtype=np.uint8 if scalar_type == vtk.VTK_UNSIGNED_CHAR else np.int16)
            for i in range(depth):
                volume_data[:, :, i] = data.T
        else:
            volume_data = data
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    volume.SetScalarComponentFromFloat(x, y, z, 0, volume_data[x, y, z])
        return volume

    def add_volume_rendering(self, volume, color, opacity_points):
        """Helper to add volume rendering with specified color and opacity."""
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        opacity_func = vtk.vtkPiecewiseFunction()
        for value, opacity in opacity_points:
            opacity_func.AddPoint(value, opacity)
        volume_property.SetScalarOpacity(opacity_func)
        color_func = vtk.vtkColorTransferFunction()
        for value, _ in opacity_points:
            color_func.AddRGBPoint(value, *color)
        volume_property.SetColor(color_func)
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(volume)
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(volume_mapper)
        volume_actor.SetProperty(volume_property)
        self.renderer.AddVolume(volume_actor)

    def create_3d_from_skull(self, skull_volume):
        """Create a 3D surface model of the skull using Marching Cubes."""
        mc_skull = vtk.vtkMarchingCubes()
        mc_skull.SetInputData(skull_volume)
        mc_skull.SetValue(0, 128)
        skull_mapper = vtk.vtkPolyDataMapper()
        skull_mapper.SetInputConnection(mc_skull.GetOutputPort())
        skull_mapper.ScalarVisibilityOff()
        skull_actor = vtk.vtkActor()
        skull_actor.SetMapper(skull_mapper)
        skull_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        skull_actor.GetProperty().SetOpacity(0.5)
        self.renderer.AddActor(skull_actor)

    def create_3d_from_brain_tissue(self, brain_volume):
        """Create a 3D volume model of brain tissue with opacity-based gradient."""
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(0, 0.0)
        opacity_func.AddPoint(20, 0.1)
        opacity_func.AddPoint(35, 0.3)
        opacity_func.AddPoint(50, 0.5)
        opacity_func.AddPoint(255, 0.5)
        volume_property.SetScalarOpacity(opacity_func)
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(20, 0.9, 0.7, 0.7)
        color_func.AddRGBPoint(35, 0.8, 0.6, 0.6)
        color_func.AddRGBPoint(50, 0.7, 0.5, 0.5)
        color_func.AddRGBPoint(255, 0.7, 0.5, 0.5)
        volume_property.SetColor(color_func)
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(brain_volume)
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(volume_mapper)
        volume_actor.SetProperty(volume_property)
        self.renderer.AddVolume(volume_actor)

    def create_3d_from_volume(self, volume_data):
        """Create a 3D representation from a 3D volume with custom transfer function."""
        self.renderer.RemoveAllViewProps()
        if volume_data is None or len(volume_data.shape) != 3:
            return
        height, width, depth = volume_data.shape
        volume = vtk.vtkImageData()
        volume.SetDimensions(width, height, depth)
        volume.SetSpacing(1.0, 1.0, 1.0)
        volume.SetOrigin(0.0, 0.0, 0.0)
        volume.AllocateScalars(vtk.VTK_SHORT, 1)
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    volume.SetScalarComponentFromFloat(x, y, z, 0, volume_data[y, x, z])
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        compositeOpacity = vtk.vtkPiecewiseFunction()
        compositeOpacity.AddPoint(-1000, 0.0)
        compositeOpacity.AddPoint(0, 0.1)
        compositeOpacity.AddPoint(20, 0.2)
        compositeOpacity.AddPoint(50, 0.5)
        compositeOpacity.AddPoint(100, 0.6)
        compositeOpacity.AddPoint(300, 0.9)
        volumeProperty.SetScalarOpacity(compositeOpacity)
        colorFunc = vtk.vtkColorTransferFunction()
        colorFunc.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        colorFunc.AddRGBPoint(0, 0.0, 0.0, 1.0)
        colorFunc.AddRGBPoint(20, 0.8, 0.6, 0.6)
        colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
        colorFunc.AddRGBPoint(100, 1.0, 0.5, 0.5)
        colorFunc.AddRGBPoint(300, 1.0, 1.0, 1.0)
        volumeProperty.SetColor(colorFunc)
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(volume)
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(volume_mapper)
        volume_actor.SetProperty(volumeProperty)
        self.renderer.AddVolume(volume_actor)
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(volume)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddActor(outline_actor)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def shutdown(self):
        """Clean up VTK interactor when closing the application."""
        self.vtkWidget.GetRenderWindow().Finalize()
        self.interactor.TerminateApp()

class BrainCTViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_volume = None
        self.current_hemorrhage_volume = None
        self.current_entropy_result = None
        self.current_slice_index = 0
        self.w_level = 40
        self.w_width = 120
        self.current_axial_slice = None
        self.current_coronal_slice = None
        self.current_sagittal_slice = None
        self.current_folder = ""
        self.thumbnails = []
        self.torch_model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('ICH Preprocessing Viewer')
        self.setGeometry(100, 100, 1600, 900)
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.file_list = QListWidget()
        self.file_list.setFixedWidth(200)
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.itemClicked.connect(self.load_selected_nifti)
        main_layout.addWidget(self.file_list)
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setFixedWidth(120)
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QVBoxLayout()
        self.thumbnail_layout.setSpacing(1)
        self.thumbnail_layout.setContentsMargins(2, 2, 2, 2)
        self.thumbnail_layout.setAlignment(Qt.AlignTop)
        self.thumbnail_container.setLayout(self.thumbnail_layout)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        main_layout.addWidget(self.thumbnail_scroll)
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout()
        viewer_widget.setLayout(viewer_layout)
        viewer_layout.setSpacing(5)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(viewer_widget)
        button_layout = QHBoxLayout()
        viewer_layout.addLayout(button_layout)
        self.open_nifti_button = QPushButton('Open NIfTI Folder')
        self.open_nifti_button.clicked.connect(self.open_nifti_folder)
        button_layout.addWidget(self.open_nifti_button)
        self.view_button = QPushButton('View')
        self.view_button.clicked.connect(self.show_axial_coronal_sagittal)
        button_layout.addWidget(self.view_button)
        self.export_button = QPushButton('Export Images')
        self.export_button.clicked.connect(self.export_images)
        button_layout.addWidget(self.export_button)
        self.batch_export_button = QPushButton('Batch Export')
        self.batch_export_button.clicked.connect(self.batch_export)
        button_layout.addWidget(self.batch_export_button)
        self.csv_export_button = QPushButton('Export CSV')
        self.csv_export_button.clicked.connect(self.export_csv_stats)
        button_layout.addWidget(self.csv_export_button)
        self.model_button = QPushButton('Load Model')
        self.model_button.clicked.connect(self.load_torch_model)
        button_layout.addWidget(self.model_button)
        self.inference_button = QPushButton('Run Inference')
        self.inference_button.clicked.connect(self.run_model_inference)
        button_layout.addWidget(self.inference_button)
        self.tab_widget = QTabWidget()
        viewer_layout.addWidget(self.tab_widget)
        self.tab_2d = QWidget()
        self.tab_widget.addTab(self.tab_2d, "2D View")
        layout_2d = QVBoxLayout(self.tab_2d)
        layout_2d.setSpacing(5)
        layout_2d.setContentsMargins(5, 5, 5, 5)
        self.controls_box = QGroupBox("DICOM Controls")
        controls_layout = QFormLayout()
        self.controls_box.setLayout(controls_layout)
        layout_2d.addWidget(self.controls_box)
        slice_layout = QHBoxLayout()
        self.slice_dial = QDial()
        self.slice_dial.setMinimum(0)
        self.slice_dial.setMaximum(0)
        self.slice_dial.setNotchesVisible(True)
        self.slice_dial.valueChanged.connect(self.update_slice)
        slice_layout.addWidget(self.slice_dial)
        self.slice_label = QLabel("Slice: 0 / 0")
        slice_layout.addWidget(self.slice_label)
        controls_layout.addRow("Slice:", slice_layout)
        level_layout = QHBoxLayout()
        self.level_dial = QDial()
        self.level_dial.setMinimum(-1000)
        self.level_dial.setMaximum(1000)
        self.level_dial.setValue(self.w_level)
        self.level_dial.setNotchesVisible(True)
        self.level_dial.valueChanged.connect(self.update_window)
        level_layout.addWidget(self.level_dial)
        self.level_label = QLabel(f"WL: {self.w_level}")
        level_layout.addWidget(self.level_label)
        controls_layout.addRow("Window Level:", level_layout)
        width_layout = QHBoxLayout()
        self.width_dial = QDial()
        self.width_dial.setMinimum(1)
        self.width_dial.setMaximum(2000)
        self.width_dial.setValue(self.w_width)
        self.width_dial.setNotchesVisible(True)
        self.width_dial.valueChanged.connect(self.update_window)
        width_layout.addWidget(self.width_dial)
        self.width_label = QLabel(f"WW: {self.w_width}")
        width_layout.addWidget(self.width_label)
        controls_layout.addRow("Window Width:", width_layout)
        self.controls_box.setVisible(False)
        processing_panel = QWidget()
        processing_layout = QHBoxLayout()
        processing_layout.setAlignment(Qt.AlignCenter)
        processing_layout.setSpacing(2)
        processing_layout.setContentsMargins(2, 2, 2, 2)
        processing_panel.setLayout(processing_layout)
        self.original_panel = self.create_image_panel("Original Image")
        self.skull_removed_panel = self.create_image_panel("Skull Removed")
        self.entropy_panel = self.create_image_panel("Entropy Threshold")
        self.prediction_panel = self.create_image_panel("Model Prediction")
        processing_layout.addWidget(self.original_panel)
        processing_layout.addWidget(self.skull_removed_panel)
        processing_layout.addWidget(self.entropy_panel)
        processing_layout.addWidget(self.prediction_panel)
        layout_2d.addWidget(processing_panel)
        self.views_panel = QWidget()
        self.views_layout = QHBoxLayout(self.views_panel)
        self.views_layout.setAlignment(Qt.AlignCenter)
        self.views_layout.setSpacing(2)
        self.views_layout.setContentsMargins(2, 2, 2, 2)
        self.views_panel.setVisible(True)
        layout_2d.addWidget(self.views_panel)
        self.axial_view_panel = self.create_image_panel("Axial")
        self.coronal_view_panel = self.create_image_panel("Coronal")
        self.sagittal_view_panel = self.create_image_panel("Sagittal")
        self.views_layout.addWidget(self.axial_view_panel)
        self.views_layout.addWidget(self.coronal_view_panel)
        self.views_layout.addWidget(self.sagittal_view_panel)
        self.tab_3d = QWidget()
        self.tab_widget.addTab(self.tab_3d, "3D Visualization")
        layout_3d = QVBoxLayout(self.tab_3d)
        self.vtk_widget = VTKWidget()
        layout_3d.addWidget(self.vtk_widget)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        viewer_layout.addWidget(self.progress_bar)
        self.watermark_label = QLabel("copyright © by Huang You-Ting", main_widget)
        self.watermark_label.setObjectName("watermark")
        self.watermark_label.setGeometry(1450, 20, 150, 30)
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(0.5)
        self.watermark_label.setGraphicsEffect(opacity_effect)
        self.watermark_label.raise_()
        self.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #aaaaaa;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                border: 1px solid #888888;
            }
            QPushButton:pressed {
                background-color: #b0b0b0;
                color: #ffffff;
                border: 1px solid #666666;
            }
            QDial {
                background-color: #f0f0f0;
                width: 50px;
                height: 50px;
            }
            QDial::handle {
                background-color: #666666;
                border: 2px solid #444444;
                border-radius: 8px;
                width: 16px;
                height: 16px;
            }
            QDial::handle:hover {
                background-color: #888888;
                border: 2px solid #666666;
            }
            QDial::handle:pressed {
                background-color: #555555;
            }
            QListWidget {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 2px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #b0b0b0;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-bottom: none;
                padding: 8px 16px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border: 1px solid #aaaaaa;
                border-bottom: 1px solid #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #d0d0d0;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #f8f8f8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-size: 14px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QWidget#imagePanel {
                border: 1px solid #dddddd;
                border-radius: 3px;
                background-color: #f0f0f0;
                padding: 2px;
            }
            QLabel#titleLabel, QLabel#thumbTitleLabel {
                font-size: 12px;
                font-weight: bold;
                padding: 2px;
            }
            QLabel#imageLabel, QLabel#thumbImageLabel {
                font-size: 14px;
                margin: 0px;
            }
            QScrollArea {
                border: 1px solid #cccccc;
                background-color: #f0f0f0;
            }
            QWidget#thumbnailPanel {
                background-color: #f0f0f0;
                padding: 2px;
            }
            QLabel#thumbImageLabel[selected="true"] {
                border: 2px solid #4CAF50;
                border-radius: 3px;
            }
            QLabel#watermark {
                color: #666666;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.original_panel.setObjectName("imagePanel")
        self.skull_removed_panel.setObjectName("imagePanel")
        self.entropy_panel.setObjectName("imagePanel")
        self.prediction_panel.setObjectName("imagePanel")
        self.axial_view_panel.setObjectName("imagePanel")
        self.coronal_view_panel.setObjectName("imagePanel")
        self.sagittal_view_panel.setObjectName("imagePanel")
        self.thumbnail_container.setObjectName("thumbnailPanel")
        self.setCentralWidget(main_widget)

    def create_image_panel(self, title):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        panel.setLayout(layout)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setFixedSize(512, 512)
        image_label.setObjectName("imageLabel")
        layout.addWidget(image_label)
        return panel

    def generate_thumbnails(self):
        if self.current_volume is None:
            return
        for label, _ in self.thumbnails:
            self.thumbnail_layout.removeWidget(label)
            label.deleteLater()
        self.thumbnails = []
        num_slices = self.current_volume.shape[2]
        slice_step = max(1, num_slices // 20)
        progress_step = 20.0 / max(1, num_slices // slice_step)
        for i in range(0, num_slices, slice_step):
            slice_img = self.current_volume[:, :, i].copy()
            slice_img = window_ct(slice_img, self.w_level, self.w_width).astype(np.uint8)
            slice_img = cv2.resize(slice_img, (100, 100), interpolation=cv2.INTER_LINEAR)
            height, width = slice_img.shape
            bytes_per_line = width
            q_img = QImage(slice_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            thumb_panel = QWidget()
            thumb_layout = QVBoxLayout()
            thumb_layout.setSpacing(2)
            thumb_layout.setContentsMargins(2, 2, 2, 2)
            thumb_panel.setLayout(thumb_layout)
            title_label = QLabel(f"Slice {i}")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setObjectName("thumbTitleLabel")
            thumb_layout.addWidget(title_label)
            image_label = ClickableLabel(i)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(100, 100)
            image_label.setPixmap(pixmap)
            image_label.setObjectName("thumbImageLabel")
            thumb_layout.addWidget(image_label)
            self.thumbnail_layout.addWidget(thumb_panel)
            self.thumbnails.append((image_label, i))
            self.progress_bar.setValue(int(self.progress_bar.value() + progress_step))
            QApplication.processEvents()
        self.update_thumbnail_highlight()

    def update_thumbnail_highlight(self):
        for label, slice_index in self.thumbnails:
            label.setProperty("selected", slice_index == self.current_slice_index)
            label.style().unpolish(label)
            label.style().polish(label)

    def on_thumbnail_clicked(self, slice_index):
        if 0 <= slice_index < self.current_volume.shape[2]:
            self.current_slice_index = slice_index
            self.slice_dial.setValue(slice_index)
            self.update_slice(slice_index)

    def open_nifti_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Open NIfTI Folder', '')
        if folder_path:
            self.current_folder = folder_path
            self.file_list.clear()
            nifti_files = [f for f in os.listdir(folder_path) if f.endswith(('.nii', '.nii.gz'))]
            if not nifti_files:
                print("No NIfTI files found in the selected folder")
                return
            self.file_list.addItems(nifti_files)
            for label, _ in self.thumbnails:
                self.thumbnail_layout.removeWidget(label)
                label.deleteLater()
            self.thumbnails = []
            self.current_volume = None
            self.current_hemorrhage_volume = None
            self.current_entropy_result = None
            self.current_axial_slice = None
            self.current_coronal_slice = None
            self.current_sagittal_slice = None
            self.controls_box.setVisible(False)
            self.vtk_widget.renderer.RemoveAllViewProps()
            self.vtk_widget.renderer.AddActor(self.vtk_widget.text_actor)
            self.vtk_widget.vtkWidget.GetRenderWindow().Render()
            for panel in [self.original_panel, self.skull_removed_panel, self.entropy_panel,
                          self.prediction_panel, self.axial_view_panel, self.coronal_view_panel,
                          self.sagittal_view_panel]:
                image_label = panel.layout().itemAt(1).widget()
                image_label.clear()
            self.torch_model = None

    def load_selected_nifti(self, item):
        if not self.current_folder or not item:
            return
        file_path = os.path.join(self.current_folder, item.text())
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        self.progress_bar.setValue(20)
        volume = read_nifti_file(file_path)
        if volume is None:
            print("Error loading NIfTI file or invalid format")
            self.progress_bar.setVisible(False)
            return
        self.progress_bar.setValue(40)
        self.current_volume = volume
        num_slices = volume.shape[2]
        self.current_hemorrhage_volume = np.zeros((512, 512, num_slices), dtype=np.uint8)
        self.slice_dial.setMaximum(num_slices - 1)
        self.w_level = 40
        self.w_width = 120
        self.current_slice_index = 0
        self.level_dial.setValue(self.w_level)
        self.width_dial.setValue(self.w_width)
        self.slice_dial.setValue(self.current_slice_index)
        self.level_label.setText(f"WL: {self.w_level}")
        self.width_label.setText(f"WW: {self.w_width}")
        self.slice_label.setText(f"Slice: {self.current_slice_index + 1} / {num_slices}")
        self.controls_box.setVisible(True)
        self.progress_bar.setValue(60)
        self.update_slice(self.current_slice_index)
        self.progress_bar.setValue(80)
        self.vtk_widget.create_3d_from_volume(volume)
        self.generate_thumbnails()
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

    def update_slice(self, slice_index):
        if self.current_volume is None or slice_index >= self.current_volume.shape[2]:
            return
        self.current_slice_index = slice_index
        current_slice = self.current_volume[:, :, slice_index].copy()
        windowed_slice = window_ct(current_slice, self.w_level, self.w_width)
        img = windowed_slice.astype(np.uint8)
        self.process_slice(img)
        self.slice_label.setText(f"Slice: {slice_index + 1} / {self.current_volume.shape[2]}")
        self.update_thumbnail_highlight()
        self.show_axial_coronal_sagittal()
        if self.torch_model:
            self.run_model_inference()

    def update_window(self):
        self.w_level = self.level_dial.value()
        self.w_width = self.width_dial.value()
        self.level_label.setText(f"WL: {self.w_level}")
        self.width_label.setText(f"WW: {self.w_width}")
        if self.current_volume is not None:
            self.current_hemorrhage_volume = None
            self.update_slice(self.current_slice_index)
            self.generate_thumbnails()

    def process_slice(self, img):
        skull_removed, brain_mask, skull_mask = remove_skull(img)
        hemorrhage_region = np.zeros_like(img, dtype=np.uint8)
        hemorrhage_region[(img >= 50) & (img <= 75)] = img[(img >= 50) & (img <= 75)]
        hemorrhage_mask = apply_entropy_and_morphology(hemorrhage_region)
        self.current_hemorrhage_volume[:, :, self.current_slice_index] = hemorrhage_mask
        self.current_entropy_result = hemorrhage_mask
        brain_tissue_mask = segment_brain_tissue(img)
        csf_mask = segment_csf(img)
        self.display_image(img, self.original_panel)
        self.display_image(skull_removed, self.skull_removed_panel)
        self.display_image(hemorrhage_mask, self.entropy_panel)
        self.vtk_widget.create_3d_from_binary(self.current_hemorrhage_volume, skull_mask, brain_tissue_mask, csf_mask)

    def display_image(self, img, panel):
        image_label = panel.layout().itemAt(1).widget()
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        height, width = img.shape
        bytes_per_line = width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        image_label.setPixmap(pixmap)

    def batch_export(self):
        if self.current_volume is None or self.current_hemorrhage_volume is None:
            print("No volume data loaded to export")
            return
        save_dir = QFileDialog.getExistingDirectory(self, 'Select Directory to Save Files', '')
        if not save_dir:
            print("No directory selected for saving files")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        num_slices = self.current_volume.shape[2]
        progress_step = 50.0 / num_slices
        for i in range(num_slices):
            slice_img = self.current_volume[:, :, i].copy()
            slice_img = window_ct(slice_img, self.w_level, self.w_width).astype(np.uint8)
            slice_path = os.path.join(save_dir, f"slice_{i + 1}.png")
            cv2.imwrite(slice_path, slice_img)
            self.progress_bar.setValue(int(self.progress_bar.value() + progress_step))
            QApplication.processEvents()
        hemorrhage_sitk = sitk.GetImageFromArray(self.current_hemorrhage_volume.astype(np.uint8))
        hemorrhage_path = os.path.join(save_dir, "hemorrhage_volume.nii.gz")
        sitk.WriteImage(hemorrhage_sitk, hemorrhage_path)
        print(f"Hemorrhage volume saved to {hemorrhage_path}")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

    def export_csv_stats(self):
        if self.current_hemorrhage_volume is None:
            print("No hemorrhage volume data available")
            return
        save_path = QFileDialog.getSaveFileName(self, 'Save CSV File', '', 'CSV Files (*.csv)')[0]
        if not save_path:
            return
        data = []
        total_volume = 0
        for i in range(self.current_hemorrhage_volume.shape[2]):
            slice_mask = self.current_hemorrhage_volume[:, :, i]
            area_mm2 = np.sum(slice_mask > 0)  # 1 pixel = 1 mm²
            volume_mm3 = area_mm2 * 1  # 1 slice = 1 mm thick
            total_volume += volume_mm3
            volume_cm3 = total_volume / 1000.0
            data.append({
                'Slice': i + 1,
                'Area_mm2': area_mm2,
                'Cumulative_Volume_cm3': volume_cm3
            })
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"CSV stats saved to {save_path}")

    def load_torch_model(self):
        model_path = QFileDialog.getOpenFileName(self, 'Select PyTorch Model', '', 'PyTorch Files (*.pth)')[0]
        if not model_path:
            return
        try:
            self.torch_model = torch.load(model_path, map_location=torch.device('cpu'))
            self.torch_model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.torch_model = None

    def run_model_inference(self):
        if self.torch_model is None:
            print("No model loaded")
            return
        if self.current_volume is None:
            print("No volume data loaded")
            return
        slice_img = self.current_volume[:, :, self.current_slice_index].copy()
        slice_img = window_ct(slice_img, self.w_level, self.w_width).astype(np.uint8)
        img = cv2.resize(slice_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_tensor = self.transform(img).unsqueeze(0)
        try:
            with torch.no_grad():
                output = self.torch_model(img_tensor)
                if isinstance(output, dict):
                    output = output['out']
                elif isinstance(output, (list, tuple)):
                    output = output[0]
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                pred = (pred > 0.5).astype(np.uint8) * 255
            self.display_image(pred, self.prediction_panel)
            print("Inference completed and displayed")
        except Exception as e:
            print(f"Error during inference: {e}")

    def export_images(self):
        if self.current_volume is None:
            print("No volume data loaded to export")
            return
        save_dir = QFileDialog.getExistingDirectory(self, 'Select Directory to Save Images', '')
        if not save_dir:
            print("No directory selected for saving images")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        self.progress_bar.setValue(25)
        original_img = self.current_volume[:, :, self.current_slice_index].copy()
        original_img = window_ct(original_img, self.w_level, self.w_width).astype(np.uint8)
        original_path = os.path.join(save_dir, f"original_slice_{self.current_slice_index + 1}.png")
        cv2.imwrite(original_path, original_img)
        print(f"Original image saved to {original_path}")
        self.progress_bar.setValue(50)
        if self.current_axial_slice is not None:
            axial_path = os.path.join(save_dir, f"axial_slice_{self.current_slice_index + 1}.png")
            cv2.imwrite(axial_path, self.current_axial_slice)
            print(f"Axial view saved to {axial_path}")
        else:
            print("Axial view not available for export")
        self.progress_bar.setValue(75)
        if self.current_coronal_slice is not None:
            coronal_resized = cv2.resize(self.current_coronal_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
            coronal_path = os.path.join(save_dir, "coronal_slice.png")
            cv2.imwrite(coronal_path, coronal_resized)
            print(f"Coronal view saved to {coronal_path}")
        else:
            print("Coronal view not available for export")
        self.progress_bar.setValue(90)
        if self.current_sagittal_slice is not None:
            sagittal_resized = cv2.resize(self.current_sagittal_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
            sagittal_path = os.path.join(save_dir, "sagittal_slice.png")
            cv2.imwrite(sagittal_path, sagittal_resized)
            print(f"Sagittal view saved to {sagittal_path}")
        else:
            print("Sagittal view not available for export")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

    def show_axial_coronal_sagittal(self):
        if self.current_volume is None:
            print("No volume data loaded to display views")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        height, width, depth = self.current_volume.shape
        axial_slice_idx = self.current_slice_index
        coronal_slice_idx = width // 2
        sagittal_slice_idx = height // 2
        self.progress_bar.setValue(20)
        axial_slice = self.current_volume[:, :, axial_slice_idx].copy()
        coronal_slice = np.rot90(self.current_volume[:, coronal_slice_idx, :], k=1)
        coronal_slice = np.flipud(coronal_slice)
        sagittal_slice = np.rot90(self.current_volume[sagittal_slice_idx, :, :], k=1)
        sagittal_slice = np.fliplr(sagittal_slice)
        self.progress_bar.setValue(40)
        axial_slice = window_ct(axial_slice, self.w_level, self.w_width).astype(np.uint8)
        coronal_slice = window_ct(coronal_slice, self.w_level, self.w_width).astype(np.uint8)
        sagittal_slice = window_ct(sagittal_slice, self.w_level, self.w_width).astype(np.uint8)
        self.current_axial_slice = axial_slice
        self.current_coronal_slice = coronal_slice
        self.current_sagittal_slice = sagittal_slice
        self.progress_bar.setValue(60)
        self.display_image(axial_slice, self.axial_view_panel)
        self.display_image(coronal_slice, self.coronal_view_panel)
        self.display_image(sagittal_slice, self.sagittal_view_panel)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

    def closeEvent(self, event):
        self.vtk_widget.shutdown()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BrainCTViewer()
    window.show()
    sys.exit(app.exec_())