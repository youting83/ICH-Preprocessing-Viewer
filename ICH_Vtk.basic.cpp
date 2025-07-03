#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>
#include <QTabWidget>
#include <QDial>
#include <QGroupBox>
#include <QFormLayout>
#include <QProgressBar>
#include <QListWidget>
#include <QScrollArea>
#include <QTextEdit>
#include <QImage>
#include <QPixmap>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

// Placeholder for LibTorch (uncomment if used)
// #include <torch/torch.h>

class ClickableLabel : public QLabel {
    Q_OBJECT
public:
    ClickableLabel(int sliceIndex, QWidget* parent = nullptr) : QLabel(parent), sliceIndex_(sliceIndex) {}
signals:
    void clicked(int sliceIndex);
protected:
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            emit clicked(sliceIndex_);
        }
    }
private:
    int sliceIndex_;
};

void windowCt(Eigen::MatrixXi& img, int wLevel = 40, int wWidth = 120) {
    double wMin = wLevel - wWidth / 2.0;
    double wMax = wLevel + wWidth / 2.0;
    for (int i = 0; i < img.rows(); ++i) {
        for (int j = 0; j < img.cols(); ++j) {
            double val = img(i, j);
            val = (val - wMin) * (255.0 / (wMax - wMin));
            val = std::max(0.0, std::min(255.0, val));
            img(i, j) = static_cast<int>(val);
        }
    }
}

double calculateEntropy(const std::vector<int>& hist, int totalPixels) {
    if (totalPixels == 0) return 0.0;
    double entropy = 0.0;
    for (int val : hist) {
        if (val > 0) {
            double prob = static_cast<double>(val) / totalPixels;
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}

cv::Mat thresholdEntropy(const cv::Mat& img) {
    cv::Mat imgU8;
    img.convertTo(imgU8, CV_8U);
    std::vector<int> hist(256, 0);
    int totalPixels = 0;
    for (int i = 0; i < imgU8.rows; ++i) {
        for (int j = 0; j < imgU8.cols; ++j) {
            int val = imgU8.at<uchar>(i, j);
            if (val > 0) {
                hist[val]++;
                totalPixels++;
            }
        }
    }
    if (totalPixels == 0) {
        return cv::Mat::zeros(img.size(), CV_8U);
    }
    double maxEntropy = 0.0;
    int optimalThreshold = 0;
    for (int t = 1; t < 255; ++t) {
        int backgroundSum = 0;
        int foregroundSum = 0;
        for (int i = 0; i < t; ++i) backgroundSum += hist[i];
        for (int i = t; i < 256; ++i) foregroundSum += hist[i];
        if (backgroundSum == 0 || foregroundSum == 0) continue;
        double backgroundEntropy = calculateEntropy(std::vector<int>(hist.begin(), hist.begin() + t), backgroundSum);
        double foregroundEntropy = calculateEntropy(std::vector<int>(hist.begin() + t, hist.end()), foregroundSum);
        double combinedEntropy = backgroundEntropy + foregroundEntropy;
        if (combinedEntropy > maxEntropy) {
            maxEntropy = combinedEntropy;
            optimalThreshold = t;
        }
    }
    cv::Mat binary;
    cv::threshold(imgU8, binary, optimalThreshold, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    return binary;
}

cv::Mat applyEntropyAndMorphology(const cv::Mat& img) {
    cv::Mat imgU8;
    img.convertTo(imgU8, CV_8U);
    cv::Mat binary;
    try {
        binary = thresholdEntropy(imgU8);
    } catch (const std::exception& e) {
        qDebug() << "Error in entropy thresholding:" << e.what() << ". Using Otsu instead.";
        cv::threshold(imgU8, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    return binary;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> removeSkull(const cv::Mat& img) {
    cv::Mat imgU8;
    img.convertTo(imgU8, CV_8U);
    cv::Mat thresh, skullThresh;
    cv::threshold(imgU8, thresh, 100, 255, cv::THRESH_BINARY);
    cv::threshold(imgU8, skullThresh, 200, 255, cv::THRESH_BINARY);
    cv::Mat brainMask;
    cv::bitwise_and(thresh, ~skullThresh, brainMask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(brainMask, brainMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(brainMask, brainMask, cv::MORPH_OPEN, kernel);
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(brainMask, labels, stats, centroids, 8);
    if (numLabels > 1) {
        int largestLabel = 1;
        int maxArea = stats.at<int>(1, cv::CC_STAT_AREA);
        for (int i = 2; i < numLabels; ++i) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) > maxArea) {
                maxArea = stats.at<int>(i, cv::CC_STAT_AREA);
                largestLabel = i;
            }
        }
        brainMask.setTo(0);
        brainMask.setTo(255, labels == largestLabel);
    }
    cv::Mat skullMask = skullThresh;
    cv::morphologyEx(skullMask, skullMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(skullMask, skullMask, cv::MORPH_OPEN, kernel);
    cv::Mat skullRemoved;
    cv::bitwise_and(imgU8, imgU8, skullRemoved, brainMask);
    return {skullRemoved, brainMask, skullMask};
}

cv::Mat segmentBrainTissue(const cv::Mat& img) {
    cv::Mat brainTissue = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat imgU8;
    img.convertTo(imgU8, CV_8U);
    for (int i = 0; i < imgU8.rows; ++i) {
        for (int j = 0; j < imgU8.cols; ++j) {
            if (imgU8.at<uchar>(i, j) >= 20 && imgU8.at<uchar>(i, j) <= 50) {
                brainTissue.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(brainTissue, brainTissue, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(brainTissue, brainTissue, cv::MORPH_OPEN, kernel);
    return brainTissue;
}

cv::Mat segmentCsf(const cv::Mat& img) {
    cv::Mat csf = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat imgU8;
    img.convertTo(imgU8, CV_8U);
    for (int i = 0; i < imgU8.rows; ++i) {
        for (int j = 0; j < imgU8.cols; ++j) {
            if (imgU8.at<uchar>(i, j) >= 0 && imgU8.at<uchar>(i, j) <= 15) {
                csf.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(csf, csf, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(csf, csf, cv::MORPH_OPEN, kernel);
    return csf;
}

struct HistogramStats {
    double mean;
    double std;
    int min;
    int max;
    double entropy;
};

HistogramStats computeHistogramStats(const cv::Mat& img) {
    cv::Mat imgU8;
    img.convertTo(imgU8, CV_8U);
    std::vector<int> hist(256, 0);
    int totalPixels = 0;
    for (int i = 0; i < imgU8.rows; ++i) {
        for (int j = 0; j < imgU8.cols; ++j) {
            int val = imgU8.at<uchar>(i, j);
            if (val > 0) {
                hist[val]++;
                totalPixels++;
            }
        }
    }
    if (totalPixels == 0) {
        return {0.0, 0.0, 0, 0, 0.0};
    }
    double mean = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] > 0) {
            mean += i * hist[i];
        }
    }
    mean /= totalPixels;
    double variance = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] > 0) {
            variance += hist[i] * std::pow(i - mean, 2);
        }
    }
    double std = std::sqrt(variance / totalPixels);
    int minVal = 255;
    int maxVal = 0;
    for (int i = 0; i < imgU8.rows; ++i) {
        for (int j = 0; j < imgU8.cols; ++j) {
            int val = imgU8.at<uchar>(i, j);
            if (val > 0) minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }
    double entropy = calculateEntropy(hist, totalPixels);
    return {mean, std, minVal, maxVal, entropy};
}

using ImageType = itk::Image<short, 3>;
using ImageReaderType = itk::ImageFileReader<ImageType>;

std::tuple<Eigen::Tensor<short, 3>, Eigen::Matrix4d, ImageType::Pointer> readNiftiFile(const std::string& filePath) {
    try {
        ImageReaderType::Pointer reader = ImageReaderType::New();
        reader->SetFileName(filePath);
        reader->Update();
        ImageType::Pointer image = reader->GetOutput();
        ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        Eigen::Tensor<short, 3> volume(size[1], size[0], size[2]);
        itk::ImageRegionIterator<ImageType> iterator(image, image->GetLargestPossibleRegion());
        for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator) {
            ImageType::IndexType idx = iterator.GetIndex();
            volume(idx[1], idx[0], idx[2]) = iterator.Get();
        }
        Eigen::Matrix4d affine;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                affine(i, j) = image->GetDirection()(i, j);
            }
        }
        return {volume, affine, image};
    } catch (const itk::ExceptionObject& e) {
        qDebug() << "Error reading NIfTI file:" << e.what();
        return {Eigen::Tensor<short, 3>(), Eigen::Matrix4d::Zero(), nullptr};
    }
}

class BrainCTViewer : public QMainWindow {
    Q_OBJECT
public:
    BrainCTViewer(QWidget* parent = nullptr) : QMainWindow(parent) {
        initUI();
    }
private:
    Eigen::Tensor<short, 3> currentVolume_;
    Eigen::Tensor<unsigned char, 3> currentHemorrhageVolume_;
    cv::Mat currentEntropyResult_;
    int currentSliceIndex_ = 0;
    int wLevel_ = 40;
    int wWidth_ = 120;
    cv::Mat currentAxialSlice_;
    cv::Mat currentCoronalSlice_;
    cv::Mat currentSagittalSlice_;
    QString currentFolder_;
    std::vector<std::pair<ClickableLabel*, int>> thumbnails_;
    Eigen::Matrix4d niftiAffine_;
    ImageType::Pointer niftiImage_;
    // Placeholder for LibTorch model
    // std::shared_ptr<torch::jit::script::Module> torchModel_;

    QListWidget* fileList_;
    QScrollArea* thumbnailScroll_;
    QWidget* thumbnailContainer_;
    QVBoxLayout* thumbnailLayout_;
    QTabWidget* tabWidget_;
    QWidget* tab2D_;
    QGroupBox* controlsBox_;
    QDial* sliceDial_;
    QLabel* sliceLabel_;
    QDial* levelDial_;
    QLabel* levelLabel_;
    QDial* widthDial_;
    QLabel* widthLabel_;
    QPushButton* openNiftiButton_;
    QPushButton* viewButton_;
    QPushButton* exportButton_;
    QPushButton* exportMaskButton_;
    QPushButton* batchExportButton_;
    QPushButton* csvExportButton_;
    QPushButton* modelButton_;
    QPushButton* inferenceButton_;
    QWidget* originalPanel_;
    QWidget* skullRemovedPanel_;
    QWidget* entropyPanel_;
    QWidget* predictionPanel_;
    QWidget* axialViewPanel_;
    QWidget* coronalViewPanel_;
    QWidget* sagittalViewPanel_;
    QTextEdit* histText_;
    QProgressBar* progressBar_;

    void initUI() {
        setWindowTitle("ICH 2D Preprocessing Viewer");
        resize(1600, 900);
        QWidget* mainWidget = new QWidget(this);
        QHBoxLayout* mainLayout = new QHBoxLayout(mainWidget);
        mainLayout->setSpacing(5);
        mainLayout->setContentsMargins(5, 5, 5, 5);
        fileList_ = new QListWidget(mainWidget);
        fileList_->setFixedWidth(200);
        fileList_->setSelectionMode(QAbstractItemView::SingleSelection);
        connect(fileList_, &QListWidget::itemClicked, this, &BrainCTViewer::loadSelectedNifti);
        mainLayout->addWidget(fileList_);
        thumbnailScroll_ = new QScrollArea(mainWidget);
        thumbnailScroll_->setFixedWidth(120);
        thumbnailScroll_->setWidgetResizable(true);
        thumbnailContainer_ = new QWidget();
        thumbnailLayout_ = new QVBoxLayout(thumbnailContainer_);
        thumbnailLayout_->setSpacing(1);
        thumbnailLayout_->setContentsMargins(2, 2, 2, 2);
        thumbnailLayout_->setAlignment(Qt::AlignTop);
        thumbnailScroll_->setWidget(thumbnailContainer_);
        mainLayout->addWidget(thumbnailScroll_);
        QWidget* viewerWidget = new QWidget(mainWidget);
        QVBoxLayout* viewerLayout = new QVBoxLayout(viewerWidget);
        viewerLayout->setSpacing(5);
        viewerLayout->setContentsMargins(0, 0, 0, 0);
        mainLayout->addWidget(viewerWidget);
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        viewerLayout->addLayout(buttonLayout);
        openNiftiButton_ = new QPushButton("Open NIfTI Folder", viewerWidget);
        connect(openNiftiButton_, &QPushButton::clicked, this, &BrainCTViewer::openNiftiFolder);
        buttonLayout->addWidget(openNiftiButton_);
        viewButton_ = new QPushButton("View", viewerWidget);
        connect(viewButton_, &QPushButton::clicked, this, &BrainCTViewer::showAxialCoronalSagittal);
        buttonLayout->addWidget(viewButton_);
        exportButton_ = new QPushButton("Export Images", viewerWidget);
        connect(exportButton_, &QPushButton::clicked, this, &BrainCTViewer::exportImages);
        buttonLayout->addWidget(exportButton_);
        exportMaskButton_ = new QPushButton("Export Mask (.nii.gz)", viewerWidget);
        connect(exportMaskButton_, &QPushButton::clicked, this, &BrainCTViewer::exportNiftiMask);
        buttonLayout->addWidget(exportMaskButton_);
        batchExportButton_ = new QPushButton("Batch Export", viewerWidget);
        connect(batchExportButton_, &QPushButton::clicked, this, &BrainCTViewer::batchExport);
        buttonLayout->addWidget(batchExportButton_);
        csvExportButton_ = new QPushButton("Export CSV", viewerWidget);
        connect(csvExportButton_, &QPushButton::clicked, this, &BrainCTViewer::exportCsvStats);
        buttonLayout->addWidget(csvExportButton_);
        modelButton_ = new QPushButton("Load Model", viewerWidget);
        connect(modelButton_, &QPushButton::clicked, this, &BrainCTViewer::loadTorchModel);
        buttonLayout->addWidget(modelButton_);
        inferenceButton_ = new QPushButton("Run Inference", viewerWidget);
        connect(inferenceButton_, &QPushButton::clicked, this, &BrainCTViewer::runModelInference);
        buttonLayout->addWidget(inferenceButton_);
        tabWidget_ = new QTabWidget(viewerWidget);
        viewerLayout->addWidget(tabWidget_);
        tab2D_ = new QWidget();
        tabWidget_->addTab(tab2D_, "2D View");
        QVBoxLayout* layout2D = new QVBoxLayout(tab2D_);
        layout2D->setSpacing(5);
        layout2D->setContentsMargins(5, 5, 5, 5);
        controlsBox_ = new QGroupBox("DICOM Controls", tab2D_);
        QFormLayout* controlsLayout = new QFormLayout(controlsBox_);
        layout2D->addWidget(controlsBox_);
        QHBoxLayout* sliceLayout = new QHBoxLayout();
        sliceDial_ = new QDial();
        sliceDial_->setMinimum(0);
        sliceDial_->setMaximum(0);
        sliceDial_->setNotchesVisible(true);
        connect(sliceDial_, &QDial::valueChanged, this, &BrainCTViewer::updateSlice);
        sliceLayout->addWidget(sliceDial_);
        sliceLabel_ = new QLabel("Slice: 0 / 0");
        sliceLayout->addWidget(sliceLabel_);
        controlsLayout->addRow("Slice:", sliceLayout);
        QHBoxLayout* levelLayout = new QHBoxLayout();
        levelDial_ = new QDial();
        levelDial_->setMinimum(-1000);
        levelDial_->setMaximum(1000);
        levelDial_->setValue(wLevel_);
        levelDial_->setNotchesVisible(true);
        connect(levelDial_, &QDial::valueChanged, this, &BrainCTViewer::updateWindow);
        levelLayout->addWidget(levelDial_);
        levelLabel_ = new QLabel(QString("WL: %1").arg(wLevel_));
        levelLayout->addWidget(levelLabel_);
        controlsLayout->addRow("Window Level:", levelLayout);
        QHBoxLayout* widthLayout = new QHBoxLayout();
        widthDial_ = new QDial();
        widthDial_->setMinimum(1);
        widthDial_->setMaximum(2000);
        widthDial_->setValue(wWidth_);
        widthDial_->setNotchesVisible(true);
        connect(widthDial_, &QDial::valueChanged, this, &BrainCTViewer::updateWindow);
        widthLayout->addWidget(widthDial_);
        widthLabel_ = new QLabel(QString("WW: %1").arg(wWidth_));
        widthLayout->addWidget(widthLabel_);
        controlsLayout->addRow("Window Width:", widthLayout);
        controlsBox_->setVisible(false);
        QWidget* processingPanel = new QWidget(tab2D_);
        QHBoxLayout* processingLayout = new QHBoxLayout(processingPanel);
        processingLayout->setAlignment(Qt::AlignCenter);
        processingLayout->setSpacing(2);
        processingLayout->setContentsMargins(2, 2, 2, 2);
        originalPanel_ = createImagePanel("Original Image");
        skullRemovedPanel_ = createImagePanel("Skull Removed");
        entropyPanel_ = createImagePanel("Hemorrhage Mask");
        predictionPanel_ = createImagePanel("Model Prediction");
        processingLayout->addWidget(originalPanel_);
        processingLayout->addWidget(skullRemovedPanel_);
        processingLayout->addWidget(entropyPanel_);
        processingLayout->addWidget(predictionPanel_);
        layout2D->addWidget(processingPanel);
        QWidget* viewsPanel = new QWidget(tab2D_);
        QHBoxLayout* viewsLayout = new QHBoxLayout(viewsPanel);
        viewsLayout->setAlignment(Qt::AlignCenter);
        viewsLayout->setSpacing(2);
        viewsLayout->setContentsMargins(2, 2, 2, 2);
        axialViewPanel_ = createImagePanel("Axial");
        coronalViewPanel_ = createImagePanel("Coronal");
        sagittalViewPanel_ = createImagePanel("Sagittal");
        viewsLayout->addWidget(axialViewPanel_);
        viewsLayout->addWidget(coronalViewPanel_);
        viewsLayout->addWidget(sagittalViewPanel_);
        layout2D->addWidget(viewsPanel);
        QGroupBox* histogramPanel = new QGroupBox("Histogram Analysis", tab2D_);
        QVBoxLayout* histLayout = new QVBoxLayout(histogramPanel);
        histText_ = new QTextEdit();
        histText_->setReadOnly(true);
        histText_->setFixedHeight(100);
        histLayout->addWidget(histText_);
        layout2D->addWidget(histogramPanel);
        progressBar_ = new QProgressBar(viewerWidget);
        progressBar_->setRange(0, 100);
        progressBar_->setValue(0);
        progressBar_->setVisible(false);
        viewerLayout->addWidget(progressBar_);
        setCentralWidget(mainWidget);
        setStyleSheet(R"(
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
                font-size: 12px;
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
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 5px;
            }
        )");
        originalPanel_->setObjectName("imagePanel");
        skullRemovedPanel_->setObjectName("imagePanel");
        entropyPanel_->setObjectName("imagePanel");
        predictionPanel_->setObjectName("imagePanel");
        axialViewPanel_->setObjectName("imagePanel");
        coronalViewPanel_->setObjectName("imagePanel");
        sagittalViewPanel_->setObjectName("imagePanel");
        thumbnailContainer_->setObjectName("thumbnailPanel");
    }

    QWidget* createImagePanel(const QString& title) {
        QWidget* panel = new QWidget();
        QVBoxLayout* layout = new QVBoxLayout(panel);
        layout->setSpacing(2);
        layout->setContentsMargins(2, 2, 2, 2);
        QLabel* titleLabel = new QLabel(title, panel);
        titleLabel->setAlignment(Qt::AlignCenter);
        titleLabel->setObjectName("titleLabel");
        layout->addWidget(titleLabel);
        QLabel* imageLabel = new QLabel(panel);
        imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setFixedSize(512, 512);
        imageLabel->setObjectName("imageLabel");
        layout->addWidget(imageLabel);
        return panel;
    }

    void displayImage(const cv::Mat& img, QWidget* panel) {
        QLabel* imageLabel = qobject_cast<QLabel*>(panel->layout()->itemAt(1)->widget());
        cv::Mat displayImg;
        if (img.size() != cv::Size(512, 512)) {
            cv::resize(img, displayImg, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);
        } else {
            displayImg = img.clone();
        }
        QImage qImg(displayImg.data, displayImg.cols, displayImg.rows, displayImg.step, QImage::Format_Grayscale8);
        imageLabel->setPixmap(QPixmap::fromImage(qImg));
    }

private slots:
    void openNiftiFolder() {
        QString folderPath = QFileDialog::getExistingDirectory(this, "Open NIfTI Folder", "");
        if (!folderPath.isEmpty()) {
            currentFolder_ = folderPath;
            fileList_->clear();
            QDir dir(currentFolder_);
            QStringList niftiFiles = dir.entryList({"*.nii", "*.nii.gz"}, QDir::Files);
            if (niftiFiles.isEmpty()) {
                qDebug() << "No NIfTI files found in the selected folder";
                return;
            }
            fileList_->addItems(niftiFiles);
            for (const auto& pair : thumbnails_) {
                thumbnailLayout_->removeWidget(pair.first);
                delete pair.first;
            }
            thumbnails_.clear();
            currentVolume_ = Eigen::Tensor<short, 3>();
            currentHemorrhageVolume_ = Eigen::Tensor<unsigned char, 3>();
            currentEntropyResult_ = cv::Mat();
            currentAxialSlice_ = cv::Mat();
            currentCoronalSlice_ = cv::Mat();
            currentSagittalSlice_ = cv::Mat();
            niftiAffine_ = Eigen::Matrix4d::Zero();
            niftiImage_ = nullptr;
            controlsBox_->setVisible(false);
            for (QWidget* panel : {originalPanel_, skullRemovedPanel_, entropyPanel_, predictionPanel_,
                                  axialViewPanel_, coronalViewPanel_, sagittalViewPanel_}) {
                qobject_cast<QLabel*>(panel->layout()->itemAt(1)->widget())->clear();
            }
            histText_->clear();
            // torchModel_.reset();
        }
    }

    void loadSelectedNifti(QListWidgetItem* item) {
        if (currentFolder_.isEmpty() || !item) return;
        QString filePath = QDir(currentFolder_).filePath(item->text());
        progressBar_->setVisible(true);
        progressBar_->setValue(0);
        QApplication::processEvents();
        progressBar_->setValue(20);
        auto [volume, affine, image] = readNiftiFile(filePath.toStdString());
        if (volume.dimension(0) == 0) {
            qDebug() << "Error loading NIfTI file or invalid format";
            progressBar_->setVisible(false);
            return;
        }
        progressBar_->setValue(40);
        currentVolume_ = volume;
        niftiAffine_ = affine;
        niftiImage_ = image;
        int numSlices = volume.dimension(2);
        currentHemorrhageVolume_ = Eigen::Tensor<unsigned char, 3>(volume.dimension(0), volume.dimension(1), numSlices);
        currentHemorrhageVolume_.setZero();
        sliceDial_->setMaximum(numSlices - 1);
        wLevel_ = 40;
        wWidth_ = 120;
        currentSliceIndex_ = 0;
        levelDial_->setValue(wLevel_);
        widthDial_->setValue(wWidth_);
        sliceDial_->setValue(currentSliceIndex_);
        levelLabel_->setText(QString("WL: %1").arg(wLevel_));
        widthLabel_->setText(QString("WW: %1").arg(wWidth_));
        sliceLabel_->setText(QString("Slice: %1 / %2").arg(currentSliceIndex_ + 1).arg(numSlices));
        controlsBox_->setVisible(true);
        progressBar_->setValue(60);
        updateSlice(currentSliceIndex_);
        progressBar_->setValue(80);
        generateThumbnails();
        progressBar_->setValue(100);
        progressBar_->setVisible(false);
    }

    void generateThumbnails() {
        if (currentVolume_.dimension(0) == 0) return;
        for (const auto& pair : thumbnails_) {
            thumbnailLayout_->removeWidget(pair.first);
            delete pair.first;
        }
        thumbnails_.clear();
        int numSlices = currentVolume_.dimension(2);
        int sliceStep = std::max(1, numSlices / 20);
        double progressStep = 20.0 / std::max(1, numSlices / sliceStep);
        for (int i = 0; i < numSlices; i += sliceStep) {
            Eigen::MatrixXi slice = currentVolume_.chip(i, 2);
            windowCt(slice, wLevel_, wWidth_);
            cv::Mat sliceImg(slice.rows(), slice.cols(), CV_8U);
            for (int r = 0; r < slice.rows(); ++r) {
                for (int c = 0; c < slice.cols(); ++c) {
                    sliceImg.at<uchar>(r, c) = static_cast<uchar>(slice(r, c));
                }
            }
            cv::resize(sliceImg, sliceImg, cv::Size(100, 100), 0, 0, cv::INTER_LINEAR);
            QImage qImg(sliceImg.data, sliceImg.cols, sliceImg.rows, sliceImg.step, QImage::Format_Grayscale8);
            QWidget* thumbPanel = new QWidget();
            QVBoxLayout* thumbLayout = new QVBoxLayout(thumbPanel);
            thumbLayout->setSpacing(2);
            thumbLayout->setContentsMargins(2, 2, 2, 2);
            QLabel* titleLabel = new QLabel(QString("Slice %1").arg(i), thumbPanel);
            titleLabel->setAlignment(Qt::AlignCenter);
            titleLabel->setObjectName("thumbTitleLabel");
            thumbLayout->addWidget(titleLabel);
            ClickableLabel* imageLabel = new ClickableLabel(i, thumbPanel);
            imageLabel->setAlignment(Qt::AlignCenter);
            imageLabel->setFixedSize(100, 100);
            imageLabel->setPixmap(QPixmap::fromImage(qImg));
            imageLabel->setObjectName("thumbImageLabel");
            thumbLayout->addWidget(imageLabel);
            thumbnailLayout_->addWidget(thumbPanel);
            thumbnails_.emplace_back(imageLabel, i);
            connect(imageLabel, &ClickableLabel::clicked, this, &BrainCTViewer::onThumbnailClicked);
            progressBar_->setValue(progressBar_->value() + static_cast<int>(progressStep));
            QApplication::processEvents();
        }
        updateThumbnailHighlight();
    }

    void updateThumbnailHighlight() {
        for (const auto& pair : thumbnails_) {
            pair.first->setProperty("selected", pair.second == currentSliceIndex_);
            pair.first->style()->unpolish(pair.first);
            pair.first->style()->polish(pair.first);
        }
    }

    void onThumbnailClicked(int sliceIndex) {
        if (sliceIndex >= 0 && sliceIndex < currentVolume_.dimension(2)) {
            currentSliceIndex_ = sliceIndex;
            sliceDial_->setValue(sliceIndex);
            updateSlice(sliceIndex);
        }
    }

    void updateSlice(int sliceIndex) {
        if (currentVolume_.dimension(0) == 0 || sliceIndex >= currentVolume_.dimension(2)) return;
        currentSliceIndex_ = sliceIndex;
        Eigen::MatrixXi currentSlice = currentVolume_.chip(sliceIndex, 2);
        windowCt(currentSlice, wLevel_, wWidth_);
        cv::Mat img(currentSlice.rows(), currentSlice.cols(), CV_8U);
        for (int r = 0; r < currentSlice.rows(); ++r) {
            for (int c = 0; c < currentSlice.cols(); ++c) {
                img.at<uchar>(r, c) = static_cast<uchar>(currentSlice(r, c));
            }
        }
        processSlice(img);
        sliceLabel_->setText(QString("Slice: %1 / %2").arg(sliceIndex + 1).arg(currentVolume_.dimension(2)));
        updateThumbnailHighlight();
        showAxialCoronalSagittal();
    }

    void updateWindow() {
        wLevel_ = levelDial_->value();
        wWidth_ = widthDial_->value();
        levelLabel_->setText(QString("WL: %1").arg(wLevel_));
        widthLabel_->setText(QString("WW: %1").arg(wWidth_));
        if (currentVolume_.dimension(0) != 0) {
            currentHemorrhageVolume_ = Eigen::Tensor<unsigned char, 3>(
                currentVolume_.dimension(0), currentVolume_.dimension(1), currentVolume_.dimension(2));
            currentHemorrhageVolume_.setZero();
            updateSlice(currentSliceIndex_);
            generateThumbnails();
        }
    }

    void processSlice(const cv::Mat& img) {
        auto [skullRemoved, brainMask, skullMask] = removeSkull(img);
        cv::Mat hemorrhageRegion = cv::Mat::zeros(img.size(), CV_8U);
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                if (img.at<uchar>(i, j) >= 50 && img.at<uchar>(i, j) <= 75) {
                    hemorrhageRegion.at<uchar>(i, j) = img.at<uchar>(i, j);
                }
            }
        }
        cv::Mat hemorrhageMask = applyEntropyAndMorphology(hemorrhageRegion);
        for (int i = 0; i < hemorrhageMask.rows; ++i) {
            for (int j = 0; j < hemorrhageMask.cols; ++j) {
                currentHemorrhageVolume_(i, j, currentSliceIndex_) = hemorrhageMask.at<uchar>(i, j);
            }
        }
        currentEntropyResult_ = hemorrhageMask;
        cv::Mat brainTissueMask = segmentBrainTissue(img);
        cv::Mat csfMask = segmentCsf(img);
        displayImage(img, originalPanel_);
        displayImage(skullRemoved, skullRemovedPanel_);
        displayImage(hemorrhageMask, entropyPanel_);
        HistogramStats stats = computeHistogramStats(img);
        QString histText = QString("Slice %1:\nMean Intensity: %2\nStd Deviation: %3\nMin Intensity: %4\nMax Intensity: %5\nEntropy: %6")
            .arg(currentSliceIndex_ + 1)
            .arg(stats.mean, 0, 'f', 2)
            .arg(stats.std, 0, 'f', 2)
            .arg(stats.min)
            .arg(stats.max)
            .arg(stats.entropy, 0, 'f', 2);
        histText_->setText(histText);
    }

    void exportImages() {
        if (currentVolume_.dimension(0) == 0) {
            qDebug() << "No volume data loaded to export";
            return;
        }
        QString saveDir = QFileDialog::getExistingDirectory(this, "Select Directory to Save Images", "");
        if (saveDir.isEmpty()) {
            qDebug() << "No directory selected for saving images";
            return;
        }
        progressBar_->setVisible(true);
        progressBar_->setValue(0);
        QApplication::processEvents();
        progressBar_->setValue(25);
        Eigen::MatrixXi originalImg = currentVolume_.chip(currentSliceIndex_, 2);
        windowCt(originalImg, wLevel_, wWidth_);
        cv::Mat origImg(originalImg.rows(), originalImg.cols(), CV_8U);
        for (int r = 0; r < originalImg.rows(); ++r) {
            for (int c = 0; c < originalImg.cols(); ++c) {
                origImg.at<uchar>(r, c) = static_cast<uchar>(originalImg(r, c));
            }
        }
        QString originalPath = QDir(saveDir).filePath(QString("original_slice_%1.png").arg(currentSliceIndex_ + 1));
        cv::imwrite(originalPath.toStdString(), origImg);
        qDebug() << "Original image saved to" << originalPath;
        progressBar_->setValue(50);
        if (!currentAxialSlice_.empty()) {
            QString axialPath = QDir(saveDir).filePath(QString("axial_slice_%1.png").arg(currentSliceIndex_ + 1));
            cv::imwrite(axialPath.toStdString(), currentAxialSlice_);
            qDebug() << "Axial view saved to" << axialPath;
        } else {
            qDebug() << "Axial view not available for export";
        }
        progressBar_->setValue(75);
        if (!currentCoronalSlice_.empty()) {
            cv::Mat coronalResized;
            cv::resize(currentCoronalSlice_, coronalResized, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);
            QString coronalPath = QDir(saveDir).filePath("coronal_slice.png");
            cv::imwrite(coronalPath.toStdString(), coronalResized);
            qDebug() << "Coronal view saved to" << coronalPath;
        } else {
            qDebug() << "Coronal view not available for export";
        }
        progressBar_->setValue(90);
        if (!currentSagittalSlice_.empty()) {
            cv::Mat sagittalResized;
            cv::resize(currentSagittalSlice_, sagittalResized, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);
            QString sagittalPath = QDir(saveDir).filePath("sagittal_slice.png");
            cv::imwrite(sagittalPath.toStdString(), sagittalResized);
            qDebug() << "Sagittal view saved to" << sagittalPath;
        } else {
            qDebug() << "Sagittal view not available for export";
        }
        progressBar_->setValue(100);
        progressBar_->setVisible(false);
    }

    void batchExport() {
        if (currentVolume_.dimension(0) == 0 || currentHemorrhageVolume_.dimension(0) == 0) {
            qDebug() << "No volume data loaded to export";
            return;
        }
        QString saveDir = QFileDialog::getExistingDirectory(this, "Select Directory to Save Files", "");
        if (saveDir.isEmpty()) {
            qDebug() << "No directory selected for saving files";
            return;
        }
        progressBar_->setVisible(true);
        progressBar_->setValue(0);
        int numSlices = currentVolume_.dimension(2);
        double progressStep = 50.0 / numSlices;
        for (int i = 0; i < numSlices; ++i) {
            Eigen::MatrixXi slice = currentVolume_.chip(i, 2);
            windowCt(slice, wLevel_, wWidth_);
            cv::Mat sliceImg(slice.rows(), slice.cols(), CV_8U);
            for (int r = 0; r < slice.rows(); ++r) {
                for (int c = 0; c < slice.cols(); ++c) {
                    sliceImg.at<uchar>(r, c) = static_cast<uchar>(slice(r, c));
                }
            }
            QString slicePath = QDir(saveDir).filePath(QString("slice_%1.png").arg(i + 1));
            cv::imwrite(slicePath.toStdString(), sliceImg);
            progressBar_->setValue(progressBar_->value() + static_cast<int>(progressStep));
            QApplication::processEvents();
        }
        using MaskImageType = itk::Image<unsigned char, 3>;
        MaskImageType::Pointer maskImage = MaskImageType::New();
        MaskImageType::SizeType size;
        size[0] = currentHemorrhageVolume_.dimension(1);
        size[1] = currentHemorrhageVolume_.dimension(0);
        size[2] = currentHemorrhageVolume_.dimension(2);
        MaskImageType::IndexType start;
        start.Fill(0);
        MaskImageType::RegionType region(start, size);
        maskImage->SetRegions(region);
        maskImage->Allocate();
        itk::ImageRegionIterator<MaskImageType> iterator(maskImage, maskImage->GetLargestPossibleRegion());
        for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator) {
            MaskImageType::IndexType idx = iterator.GetIndex();
            iterator.Set(currentHemorrhageVolume_(idx[1], idx[0], idx[2]));
        }
        QString hemorrhagePath = QDir(saveDir).filePath("hemorrhage_volume.nii.gz");
        using WriterType = itk::ImageFileWriter<MaskImageType>;
        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(hemorrhagePath.toStdString());
        writer->SetInput(maskImage);
        writer->Update();
        qDebug() << "Hemorrhage volume saved to" << hemorrhagePath;
        progressBar_->setValue(100);
        progressBar_->setVisible(false);
    }

    void exportCsvStats() {
        if (currentHemorrhageVolume_.dimension(0) == 0) {
            qDebug() << "No hemorrhage volume data available";
            return;
        }
        QString savePath = QFileDialog::getSaveFileName(this, "Save CSV File", "", "CSV Files (*.csv)");
        if (savePath.isEmpty()) return;
        std::ofstream csvFile(savePath.toStdString());
        csvFile << "Slice,Area_mm2,Cumulative_Volume_cm3\n";
        double totalVolume = 0.0;
        for (int i = 0; i < currentHemorrhageVolume_.dimension(2); ++i) {
            int areaMm2 = 0;
            for (int r = 0; r < currentHemorrhageVolume_.dimension(0); ++r) {
                for (int c = 0; c < currentHemorrhageVolume_.dimension(1); ++c) {
                    if (currentHemorrhageVolume_(r, c, i) > 0) {
                        areaMm2++;
                    }
                }
            }
            double volumeMm3 = areaMm2 * 1.0; // 1 mm slice thickness
            totalVolume += volumeMm3;
            double volumeCm3 = totalVolume / 1000.0;
            csvFile << (i + 1) << "," << areaMm2 << "," << volumeCm3 << "\n";
        }
        csvFile.close();
        qDebug() << "CSV stats saved to" << savePath;
    }

    void loadTorchModel() {
        QString modelPath = QFileDialog::getOpenFileName(this, "Select PyTorch Model", "", "PyTorch Files (*.pth)");
        if (modelPath.isEmpty()) return;
        try {
            // Placeholder: Load model using LibTorch
            // torchModel_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(modelPath.toStdString()));
            // torchModel_->eval();
            qDebug() << "Model loaded from" << modelPath;
        } catch (const std::exception& e) {
            qDebug() << "Error loading model:" << e.what();
            // torchModel_.reset();
        }
    }

    void runModelInference() {
        // if (!torchModel_) {
        qDebug() << "No model loaded or LibTorch not integrated";
        return;
        // }
        if (currentVolume_.dimension(0) == 0) {
            qDebug() << "No volume data loaded";
            return;
        }
        Eigen::MatrixXi slice = currentVolume_.chip(currentSliceIndex_, 2);
        windowCt(slice, wLevel_, wWidth_);
        cv::Mat img(slice.rows(), slice.cols(), CV_8U);
        for (int r = 0; r < slice.rows(); ++r) {
            for (int c = 0; c < slice.cols(); ++c) {
                img.at<uchar>(r, c) = static_cast<uchar>(slice(r, c));
            }
        }
        cv::resize(img, img, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);
        // Placeholder: Convert to tensor and run inference
        // torch::Tensor imgTensor = torch::from_blob(img.data, {1, 1, img.rows, img.cols}, torch::kByte);
        // imgTensor = imgTensor.to(torch::kFloat32) / 255.0;
        // imgTensor = (imgTensor - 0.5) / 0.5;
        // std::vector<torch::jit::IValue> inputs = {imgTensor};
        // torch::Tensor output = torchModel_->forward(inputs).toTensor();
        // output = torch::sigmoid(output).squeeze();
        // cv::Mat pred(512, 512, CV_8U);
        // memcpy(pred.data, output.data_ptr<float>(), 512 * 512 * sizeof(float));
        // pred = (pred > 0.5) * 255;
        cv::Mat pred = cv::Mat::zeros(512, 512, CV_8U); // Dummy output
        displayImage(pred, predictionPanel_);
        qDebug() << "Inference completed and displayed (placeholder)";
    }

    void exportNiftiMask() {
        if (currentHemorrhageVolume_.dimension(0) == 0) {
            qDebug() << "No hemorrhage volume data available to export";
            return;
        }
        QString savePath = QFileDialog::getSaveFileName(this, "Save Hemorrhage Mask", "", "NIfTI Files (*.nii.gz)");
        if (savePath.isEmpty()) {
            qDebug() << "No file selected for saving mask";
            return;
        }
        if (!savePath.endsWith(".nii.gz")) {
            savePath += ".nii.gz";
        }
        progressBar_->setVisible(true);
        progressBar_->setValue(0);
        QApplication::processEvents();
        progressBar_->setValue(50);
        using MaskImageType = itk::Image<unsigned char, 3>;
        MaskImageType::Pointer maskImage = MaskImageType::New();
        MaskImageType::SizeType size;
        size[0] = currentHemorrhageVolume_.dimension(1);
        size[1] = currentHemorrhageVolume_.dimension(0);
        size[2] = currentHemorrhageVolume_.dimension(2);
        MaskImageType::IndexType start;
        start.Fill(0);
        MaskImageType::RegionType region(start, size);
        maskImage->SetRegions(region);
        maskImage->Allocate();
        itk::ImageRegionIterator<MaskImageType> iterator(maskImage, maskImage->GetLargestPossibleRegion());
        for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator) {
            MaskImageType::IndexType idx = iterator.GetIndex();
            iterator.Set(currentHemorrhageVolume_(idx[1], idx[0], idx[2]));
        }
        using WriterType = itk::ImageFileWriter<MaskImageType>;
        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(savePath.toStdString());
        writer->SetInput(maskImage);
        writer->Update();
        progressBar_->setValue(100);
        progressBar_->setVisible(false);
        qDebug() << "Hemorrhage mask saved to" << savePath;
    }

    void showAxialCoronalSagittal() {
        if (currentVolume_.dimension(0) == 0) {
            qDebug() << "No volume data loaded to display views";
            return;
        }
        progressBar_->setVisible(true);
        progressBar_->setValue(0);
        QApplication::processEvents();
        int height = currentVolume_.dimension(0);
        int width = currentVolume_.dimension(1);
        int depth = currentVolume_.dimension(2);
        int axialSliceIdx = currentSliceIndex_;
        int coronalSliceIdx = width / 2;
        int sagittalSliceIdx = height / 2;
        progressBar_->setValue(20);
        Eigen::MatrixXi axialSlice = currentVolume_.chip(axialSliceIdx, 2);
        Eigen::MatrixXi coronalSlice = currentVolume_.chip(coronalSliceIdx, 1);
        Eigen::MatrixXi sagittalSlice = currentVolume_.chip(sagittalSliceIdx, 0);
        // Rotate and flip coronal
        Eigen::MatrixXi coronalRotated(coronalSlice.cols(), coronalSlice.rows());
        for (int i = 0; i < coronalSlice.rows(); ++i) {
            for (int j = 0; j < coronalSlice.cols(); ++j) {
                coronalRotated(j, coronalSlice.rows() - 1 - i) = coronalSlice(i, j);
            }
        }
        // Rotate and flip sagittal
        Eigen::MatrixXi sagittalRotated(sagittalSlice.cols(), sagittalSlice.rows());
        for (int i = 0; i < sagittalSlice.rows(); ++i) {
            for (int j = 0; j < sagittalSlice.cols(); ++j) {
                sagittalRotated(sagittalSlice.cols() - 1 - j, i) = sagittalSlice(i, j);
            }
        }
        progressBar_->setValue(40);
        windowCt(axialSlice, wLevel_, wWidth_);
        windowCt(coronalRotated, wLevel_, wWidth_);
        windowCt(sagittalRotated, wLevel_, wWidth_);
        currentAxialSlice_ = cv::Mat(axialSlice.rows(), axialSlice.cols(), CV_8U);
        for (int r = 0; r < axialSlice.rows(); ++r) {
            for (int c = 0; c < axialSlice.cols(); ++c) {
                currentAxialSlice_.at<uchar>(r, c) = static_cast<uchar>(axialSlice(r, c));
            }
        }
        currentCoronalSlice_ = cv::Mat(coronalRotated.rows(), coronalRotated.cols(), CV_8U);
        for (int r = 0; r < coronalRotated.rows(); ++r) {
            for (int c = 0; c < coronalRotated.cols(); ++c) {
                currentCoronalSlice_.at<uchar>(r, c) = static_cast<uchar>(coronalRotated(r, c));
            }
        }
        currentSagittalSlice_ = cv::Mat(sagittalRotated.rows(), sagittalRotated.cols(), CV_8U);
        for (int r = 0; r < sagittalRotated.rows(); ++r) {
            for (int c = 0; c < sagittalRotated.cols(); ++c) {
                currentSagittalSlice_.at<uchar>(r, c) = static_cast<uchar>(sagittalRotated(r, c));
            }
        }
        progressBar_->setValue(60);
        displayImage(currentAxialSlice_, axialViewPanel_);
        displayImage(currentCoronalSlice_, coronalViewPanel_);
        displayImage(currentSagittalSlice_, sagittalViewPanel_);
        progressBar_->setValue(100);
        progressBar_->setVisible(false);
    }
};

#include "main.moc"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    BrainCTViewer window;
    window.show();
    return app.exec();
}