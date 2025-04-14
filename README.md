# SimGen

A simple implementation of image generation with multiple conditions.

## Environment Setup

```bash
conda create -n simgen python=3.8
conda activate simgen
pip install -r requirements.txt
```

## Data Preparation

The demo data is already included in the repository under `example_data/` directory. The data structure is:

```
example_data/
├── real_img/          # Target images
├── depth/             # Depth images
├── seg/               # Segmentation images
└── text_label.json    # Text descriptions for each image
```

## Model Weights

1. Download the pre-trained weights:
```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt
```

2. Prepare the model weights:
```bash
bash prepare_weights.sh
```

## Training

The training process consists of three stages:

1. Train ImgDiff (Image Generation):
```bash
bash train.sh
```

2. Train CondDiff (Condition Generation):
```bash
bash train_t2i.sh
```

3. Train Full SimGen (Cascade):
   - First, train CondDiff using step 2
   - Then, update the checkpoint path in `configs/cascade_v21_sn.yaml`
   - Finally, run:
```bash
bash train_cas.sh
```

## Evaluation

1. Visualize Results:
   - Use `web_page.py` for visualizing with conditions:
```bash
python web_page.py --image-folder <output_dir> --condition-num <num_conditions>
```
   - Use `web.py` for visualizing without conditions:
```bash
python web.py --image-folder <output_dir>
```

2. Calculate Metrics:
   - Pixel Diversity:
```bash
python pixel_diversity.py --image-folder <output_dir>
```
   - FID Score (using [pytorch-fid](https://github.com/mseitzer/pytorch-fid)):
```bash
pip install pytorch-fid
python -m pytorch_fid <real_images_dir> <generated_images_dir>
```

## Directory Structure

```
.
├── configs/           # Configuration files
├── ckpts/            # Model checkpoints
├── example_data/     # Demo data
├── output/           # Training outputs
├── src/              # Source code
│   ├── train/        # Training code
│   └── test/         # Testing code
└── utils/            # Utility scripts
```

## Configuration

The main configuration files are:
- `configs/local_v21_sn.yaml`: Local condition model (ImgDiff)
- `configs/t2i_v21_v.yaml`: Text-to-image model (CondDiff)
- `configs/cascade_v21_sn.yaml`: Cascade model (Full SimGen)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
