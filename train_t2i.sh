CUDA_VISIBLE_DEVICES=0 python src/train/train.py --config-path ./configs/t2i_v21_v.yaml \
    --logdir <output_dir> \
    --batch-size 12 \
    --gpus 1 \
    --resume <model_ckpt_path>