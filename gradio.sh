export GRADIO_TEMP_DIR="<temp_dir>"
python src/test/test_sd21.py --config-path ./configs/local_v21_sn.yaml \
    --resume <model_ckpt_path>