# Road Map Extraction from Satellite Images

This repository contains the implementation of a deep learning-based system for **road map extraction** from satellite images, developed as part of my master's thesis at Shahid Beheshti University. The primary objective of this project is to create a highly accurate and efficient model capable of detecting road networks in remote sensing imagery, specifically designed for routing and rescue operations in areas not covered by traditional map services like Google Maps.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Overview

The project focuses on extracting road maps from satellite images using advanced deep learning techniques. The proposed system, named **RoadFormer**, is built on a combination of traditional Convolutional Neural Networks (CNNs) and modern transformer-based architectures like the **Swin Transformer**. The core objective is to leverage these models for road segmentation tasks in both urban and rural areas with varying terrain conditions.

The **RoadFormer** model addresses the challenges of satellite image segmentation by incorporating techniques such as:
- **Dilated Convolutions**: To increase the receptive field without losing spatial resolution.
- **Separable Convolutions**: To improve computational efficiency and reduce overfitting.
- **Swin Transformer**: For efficient multi-scale feature extraction.

## Model Architecture

The architecture of **RoadFormer** can be divided into three key components:

1. **Encoder**: 
   - Based on the **Swin Transformer**, the encoder extracts multi-scale features from the input satellite images.
   - The Swin Transformer enables the model to focus on different regions of the image at different scales, capturing both local and global context.

2. **Bottleneck**:
   - The bottleneck layer uses **dilated convolutions** to capture larger spatial dependencies while maintaining computational efficiency. Dilated convolutions expand the receptive field without increasing the number of parameters, making the model more effective in detecting long, continuous road structures.

3. **Decoder**:
   - The decoder utilizes **separable convolutions** to upsample the features and refine the segmented road maps. By splitting the depthwise and pointwise convolutions, the decoder can reconstruct accurate pixel-level details while keeping the number of computations low.

The model architecture is designed to balance **accuracy** and **efficiency**, making it well-suited for real-world applications such as disaster management, urban planning, and transportation systems.

## Datasets

This project utilizes two well-known datasets for training and evaluation:

- **DeepGlobe**: A large-scale dataset designed for satellite image road extraction. It contains high-resolution images paired with ground truth road labels.
- **Massachusetts Road Dataset**: Another dataset that focuses on road detection in aerial images. This dataset contains various urban and suburban scenes, making it a good benchmark for the road extraction task.

Both datasets include a diverse range of road types, from highways to narrow rural roads, providing the necessary data diversity for training robust models.

## Installation

To set up the environment for running the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/mAbbaspour/thesis-SBU.git
   cd thesis-SBU
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets (DeepGlobe, Massachusetts) and place them in the appropriate directories.

4. Ensure you have the following Python packages installed:
   - `torch`
   - `numpy`
   - `opencv-python`
   - `matplotlib`
   - `tqdm`

## Usage

### Preprocessing

Before training, the input satellite images must be preprocessed. Run the preprocessing script to resize, normalize, and augment the data:

```bash
python preprocess.py --dataset deepglobe
```

### Training

To train the **RoadFormer** model on the DeepGlobe dataset, use the following command:

```bash
python train.py --model roadformer --dataset deepglobe --epochs 50 --batch_size 16
```

You can adjust the number of epochs and batch size based on your computational resources.

### Evaluation

After training, evaluate the model on the test set:

```bash
python evaluate.py --model roadformer --dataset deepglobe
```

This will generate performance metrics such as Intersection over Union (IoU) and F1-score.

### Visualization

To visualize the road segmentation results on a sample test image, use the visualization script:

```bash
python visualize.py --model roadformer --dataset massachusetts --image_path path_to_image
```

This script will output a visual comparison between the ground truth and the predicted road map.

## Results

The results of the **RoadFormer** model show significant improvements over baseline methods such as **U-Net** and **PSPNet**, particularly in terms of accuracy and IoU scores. Below is a performance comparison across different models:

| Model        | Dataset      | IoU   | F1-Score |
|--------------|--------------|-------|----------|
| RoadFormer   | DeepGlobe    | 85.4% | 89.1%    |
| U-Net        | DeepGlobe    | 81.7% | 86.3%    |
| PSPNet       | Massachusetts| 82.1% | 87.2%    |
| DeepLabV3    | DeepGlobe    | 83.5% | 88.0%    |

The **RoadFormer** model excels in handling complex terrains, particularly in rural and forested areas, where traditional models struggle.

## Future Work

While the results are promising, there is still room for improvement. Some potential future directions include:

- **Handling Complex Terrains**: Further improve the model’s ability to extract roads in diverse terrains such as **mountainous** and **forested** areas, particularly in regions like **Iran**, where such features are prevalent.
- **Real-Time Integration**: Integrating the system with live satellite data streams for real-time road extraction and updating of maps.
- **Low-Resolution Images**: Optimize the model for performance on lower-resolution images to reduce computational costs and broaden its applicability in real-time scenarios.

## Acknowledgements

This project was developed as part of my master's thesis under the supervision of **Dr. Kheradpisheh** and **Dr. Hajiabolhassan** at Shahid Beheshti University. I would like to express my deep gratitude to both of my supervisors for their invaluable guidance and support throughout the development of this system. I also appreciate the constructive feedback from my colleagues and mentors, which significantly contributed to the success of this work.
```
