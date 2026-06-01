# GalaxEye Binary Change Detection (EO-SAR)

## Project Title & Description
**Siamese UNet for Imbalanced Binary Change Detection using EO-SAR Images**

This repository contains the solution for the GalaxEye Satellite AI Research Intern assessment. The task involves performing pixel-level binary change detection (0 = No-Change, 1 = Change) given co-registered pre-event (EO) and post-event (SAR) image pairs.

The solution implements a **Siamese UNet** architecture with weight-shared ResNet-34 encoders to extract multi-modal features, absolute difference-based skip fusion, and a combined BCE+Dice loss function to tackle severe class imbalance (the dataset exhibits a 58:1 ratio of no-change to change pixels).

## Requirements
- Python 3.11+
- See `requirements.txt` for all pinned dependencies.

## Environment Setup
Run the following commands to create a virtual environment and install dependencies:

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt
```

## Dataset Structure
Place the provided dataset inside a folder named `Dataset/` at the root of the repository. The expected directory layout is:

```
Galaxyeyeai/
├── Dataset/
│   ├── train/
│   │   ├── pre-event/
│   │   ├── post-event/
│   │   └── target/
│   ├── val/
│   │   ├── pre-event/
│   │   ├── post-event/
│   │   └── target/
│   └── test/
│       ├── pre-event/
│       ├── post-event/
│       └── target/
├── src/
├── config.yaml
├── train.py
├── eval.py
└── ...
```

*Note: The script automatically handles the mandatory label remapping (`0/1` -> `0`, `2/3` -> `1`).*

## Training
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7354c57f-3de3-4269-b7ae-854214503d35" />

To train the model from scratch, execute:
```bash
python train.py --config config.yaml
```
Checkpoints will be saved automatically to `checkpoints/best.pth` and `checkpoints/last.pth`. TensorBoard logs are stored in `runs/`.

## Evaluation
To evaluate the model on the test data (or any split) and generate metrics + visualisations, run:
```bash
python eval.py --data_path Dataset/test --weights checkpoints/best.pth
```
Results will be saved inside the `results/` folder, including JSON metrics, confusion matrices, and qualitative image comparisons.

## Model Weights

- Link: [Hugging Face Repository](https://huggingface.co/rishii100/binary-change-detection)

## Results


| Split      | IoU     | Precision | Recall  | F1 Score |
|------------|---------|-----------|---------|----------|
| Validation | 24.74%  | 26.45%    | 79.29%  | 39.67%   |
| Test       | 3.50%   | 4.19%     | 17.37%  | 6.75%    |

## Citation / References
- **UNet:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015.
- **Siamese Networks for CD:** Daudt et al., "Fully Convolutional Siamese Networks for Change Detection", ICIP 2018.
- **Albumentations:** Buslaev et al., "Albumentations: fast and flexible image augmentations", Information 2020.
- **Timm:** Ross Wightman, "PyTorch Image Models", GitHub.
