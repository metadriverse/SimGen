import sys
if './' not in sys.path:
	sys.path.append('./')
from utils.share import *
import utils.config as config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
from pytorch_lightning import seed_everything
import argparse

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
from annotator.uniformer import UniformerDetector
from annotator.content import ContentDetector

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler


def process(depth_image, seg_image, sn_depth_image, sn_seg_image, sn_img_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution_h, image_resolution_w, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength):
    
    seed_everything(seed)

    if depth_image is not None:
        anchor_image = depth_image
    elif seg_image is not None:
        anchor_image = seg_image
    elif sn_depth_image is not None:
        anchor_image = sn_depth_image
    elif sn_seg_image is not None:
        anchor_image = sn_seg_image
    elif sn_img_image is not None:
        anchor_image = sn_img_image
    elif content_image is not None:
        anchor_image = content_image
    else:
        anchor_image = np.zeros((image_resolution_h, image_resolution_w, 3)).astype(np.uint8)
    H, W, C = resize_image(HWC3(anchor_image), image_resolution_h).shape

    with torch.no_grad():
        if depth_image is not None:
            depth_image = cv2.resize(depth_image, (W, H))
            depth_image = depth_image
            use_depth = True
        else:
            depth_image = np.zeros((H, W, C)).astype(np.uint8)
            use_depth = False
        if seg_image is not None:
            seg_image = cv2.resize(seg_image, (W, H))
            seg_image = seg_image
            use_seg = True
        else:
            seg_image = np.zeros((H, W, C)).astype(np.uint8)
            use_seg = False
        if sn_depth_image is not None:
            sn_depth_image = cv2.resize(sn_depth_image, (W, H))
            sn_depth_image = sn_depth_image
            use_sn_depth = True
        else:
            sn_depth_image = np.zeros((H, W, C)).astype(np.uint8)
            use_sn_depth = False
        if sn_seg_image is not None:
            sn_seg_image = cv2.resize(sn_seg_image, (W, H))
            sn_seg_image = sn_seg_image
            use_sn_seg = True      
        else:
            sn_seg_image = np.zeros((H, W, C)).astype(np.uint8)
            use_sn_seg = False
        if sn_img_image is not None:
            sn_img_image = cv2.resize(sn_img_image, (W, H))
            sn_img_image = sn_img_image
        else:
            sn_img_image = np.zeros((H, W, C)).astype(np.uint8)
        if content_image is not None:
            content_emb = apply_content(content_image)
        else:
            content_emb = np.zeros((768))

        if use_depth or use_seg:
            detected_maps_list = [depth_image, seg_image]
        else:
            detected_maps_list = [sn_depth_image, sn_seg_image]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, global_strength=global_strength)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]

    return [results, detected_maps_list]

def main():
    parser = argparse.ArgumentParser(description='SimGen Demo')
    parser.add_argument('--config-path', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()

    global model, ddim_sampler
    model = create_model(args.config_path).cpu()
    model.load_state_dict(load_state_dict(args.resume, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## SimGen Demo")
        with gr.Row():
            sn_depth_image = gr.Image(source='upload', type="numpy", label='sn_depth')
            sn_seg_image = gr.Image(source='upload', type="numpy", label='sn_seg')
            sn_img_image = gr.Image(source='upload', type="numpy", label='sn_img')
        with gr.Row():
            depth_image = gr.Image(source='upload', type="numpy", label='depth')
            seg_image = gr.Image(source='upload', type="numpy", label='seg')
            content_image = gr.Image(source='upload', type="numpy", label='content')
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
        with gr.Row():
            run_button = gr.Button(label="Run")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=6, step=1)
                    image_resolution_h = gr.Slider(label="Image Height", minimum=256, maximum=768, value=448, step=64)
                    image_resolution_w = gr.Slider(label="Image Width", minimum=256, maximum=768, value=768, step=64)
                    strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1, step=0.01)
                    global_strength = gr.Slider(label="Global Strength", minimum=0, maximum=2, value=1, step=0.01)
                    low_threshold = gr.Slider(label="Canny Low Threshold", minimum=1, maximum=255, value=100, step=1)
                    high_threshold = gr.Slider(label="Canny High Threshold", minimum=1, maximum=255, value=200, step=1)
                    value_threshold = gr.Slider(label="Hough Value Threshold (MLSD)", minimum=0.01, maximum=2.0, value=0.1, step=0.01)
                    distance_threshold = gr.Slider(label="Hough Distance Threshold (MLSD)", minimum=0.01, maximum=20.0, value=0.1, step=0.01)
                    alpha = gr.Slider(label="Alpha", minimum=0.1, maximum=20.0, value=6.2, step=0.01)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=42, step=1)
                    eta = gr.Number(label="Eta (DDIM)", value=0.0)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                          value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        
        with gr.Row():
            image_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        with gr.Row():
            cond_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        

        ips = [depth_image, seg_image, sn_depth_image, sn_seg_image, sn_img_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution_h, image_resolution_w, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength]
        run_button.click(fn=process, inputs=ips, outputs=[image_gallery, cond_gallery])


    block.launch(server_name='0.0.0.0')

if __name__ == "__main__":
    main()
