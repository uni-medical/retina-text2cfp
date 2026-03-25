# 🔬 RetinaLogos Training Setup


## 📁 Directory Structure

```
├── configs/
│   ├── train/
│   │   ├── train.sh                 # Training script
│   │   └── data_example.yaml        # Training data configuration file
│   └── inference/
│       ├── inference.sh             # Inference script
│       └── example_captions.txt     # Input text examples file
├── example_data/
│   ├── example_ver1.json           # Training dataset json file
│   ├── example_ver2.json           # Training dataset json file
│   └── images/                     # Folder for Image data
├── results/                        # Output results (Automatically Generated)
└── environment_RetinaLogos.yml # Environment configuration
```

## 🛠️ Environment Setup

```bash
# Create conda environment
conda env create -f environment_RetinaLogos.yml

# Activate environment
conda activate RetinaLogos
```

## 📄 Data Format

Training data in JSON format (A few EyePACS dataset examples):

```json
{
    "image": "example_data/images/27597_right.jpeg",
    "id": "23702",
    "width": 2592,
    "height": 3888,
    "caption": "Detailed medical description of the fundus photograph..."
}
```

## 🚀 Training

```bash
# Run training
bash configs/train/train.sh
```

### ⚙️ Training Parameters

- Batch size: 8
- Learning rate: 1e-6
- Max steps: 400,000
- Image size: 512x512
- Precision: bf16

## 🔮 Inference

```bash
# Run inference (supports multi-GPU)
bash configs/inference/inference.sh
```

The script uses `configs/inference/example_captions.txt` as input text and generates images to `./results/inference_results/`. You can customize the input text file by editing the caption file path in the inference script.

### 🎨 Custom Inference

Edit `configs/inference/example_captions.txt` with your descriptions, with each description on a separate line in the txt file:

```text
This fundus photograph captures a detailed view of the retina, showcasing its intricate structure and various landmarks. The fundus photograph reveals a clear view of the retinal surface, showcasing key structures of the eye.
```

## 📤 Output

- Training checkpoints: `./results/training/`
- Generated images: `./results/inference_results/`

## 📥 Model Checkpoints

Model checkpoints can be downloaded from the following link:

**Checkpoint Download Link**: [Model]([https://1drv.ms/f/c/b53a195b16c6fb0e/ErVc9gdh9ldOueaqnVoxs68BhQe5xtCxlAvg-VDs02uCDA?e=hsnnYR](https://1drv.ms/f/c/b53a195b16c6fb0e/IgC1XPYHYfZXTrnmqp1aMbOvASYMhYUA-FmJQ38U4SvAYyo?e=2Q4gMx))

After downloading the checkpoint files, please place them in the appropriate directory and update the model path configuration in the inference script.
