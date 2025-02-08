import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from simgen.annotator.util import resize_image, HWC3
from simgen.ldm.util import instantiate_from_config
from simgen.models.ddim_hacked import DDIMSampler
from simgen.models.util import create_model, load_state_dict


class Pipeline(ABC):
    """ Abstract class for a generation pipeline """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self):
        pass


@dataclass
class SimGenOutput:
    """ Dataclass for SimGen Output """
    images: List[PIL.Image.Image]


class SimGenPipeline(Pipeline):
    """ Class for SimGen Generation Pipeline """
    def __init__(self, torch_dtype=torch.float32, device_map="cuda", load_first_stage: bool = True, **kwargs):
        super().__init__()
        local_path = snapshot_download(repo_id="SichengMo-UCLA/SimGen")
        print(f"Model downloaded to {local_path}")
        config_path = os.path.join(local_path, "config.yaml")
        ckpt_path = os.path.join(local_path, "simgen_v2.1.ckpt")
        model = create_model(config_path).cpu()
        model.load_state_dict(load_state_dict(ckpt_path, location=device_map))
        self.model = model.cuda()
        self.dtype = torch_dtype
        self.ddim_sampler = DDIMSampler(model)
        self.split_back_ground = False
        self.load_first_stage = load_first_stage
        if load_first_stage:
            first_cond_model_config = OmegaConf.load(config_path).first_cond_config
            first_cond_model = instantiate_from_config(first_cond_model_config).cpu()
            cond_diff_path = os.path.join(local_path, "conddiff.ckpt")
            first_cond_model.load_state_dict(load_state_dict(cond_diff_path, location=device_map))
            self.first_cond_model = first_cond_model
        else:
            self.first_cond_model = None

    def split_foreground(self, depth_image, seg_image):
        COLOR_DICT_CITY = np.array(
            [
                [128, 64, 128],  # 0: 'road'
                [232, 35, 244],  # 1: 'sidewalk'
                [70, 70, 70],  # 2: 'building'
                [156, 102, 102],  # 3: 'wall'
                [153, 153, 190],  # 4: 'fence'
                [153, 153, 153],  # 5: 'pole'
                [30, 170, 250],  # 6: 'traffic light'
                [0, 220, 220],  # 7: 'traffic sign'
                [35, 142, 107],  # 8: 'vegetation'
                [152, 251, 152],  # 9: 'terrain'
                [180, 130, 70],  # 10: 'sky'
                [60, 20, 220],  # 11: 'person'
                [0, 0, 255],  # 12: 'rider'
                [142, 0, 0],  # 13: 'car'
                [70, 0, 0],  # 14: 'truck'
                [100, 60, 0],  # 15: 'bus'
                [100, 80, 0],  # 16: 'train'
                [230, 0, 0],  # 17: 'motorcycle'
                [32, 11, 119],  # 18: 'bicycle'
                [189, 176, 55],  # 19: 'cross work'
                [255, 255, 255]  # 20: 'lane line'
            ]
        )
        if depth_image is None or seg_image is None:
            return [None, None, None, None], None
        fore_depth, back_depth, fore_seg, back_seg = depth_image, depth_image, seg_image, seg_image
        COLOR_DICT_CITY[:, [2, 0]] = COLOR_DICT_CITY[:, [0, 2]]
        colormap = COLOR_DICT_CITY
        colormap_reshaped = colormap.reshape((1, -1, 3))
        seg_image = (seg_image * 255).astype(np.uint8)
        distances = np.linalg.norm(seg_image[:, :, np.newaxis, :] - colormap_reshaped, axis=-1)
        index_matrix = np.argmin(distances, axis=-1)

        mask = np.zeros((seg_image.shape[0], seg_image.shape[1], 1), dtype=np.uint8)
        foreground = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        mask[..., 0] = np.isin(index_matrix, foreground).astype(np.uint8)

        ori_mask = mask.copy()
        fore_depth = np.where(mask == 0, 0, fore_depth)
        back_depth = np.where(mask == 1, 0, back_depth)
        fore_seg = np.where(mask == 0, 0, fore_seg)
        back_seg = np.where(mask == 1, 0, back_seg)
        mask = np.logical_or(index_matrix == 19, index_matrix == 20).astype(np.uint8)
        fore_seg[mask == 1] = [128 / 255.0, 64 / 255.0, 128 / 255.0]

        return [fore_depth, fore_seg, back_depth, back_seg], ori_mask

    def mix_cond(self, back_cond, fore_cond):
        if back_cond is None:
            return fore_cond
        H, W, _ = fore_cond.shape
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        fore_cond = (fore_cond * 255).astype(np.uint8)
        back_cond = (back_cond * 255).astype(np.uint8)
        mask[..., 0] = np.isin(fore_cond[..., 0], [0]).astype(np.uint8) & np.isin(fore_cond[..., 1], [0]).astype(
            np.uint8
        ) & np.isin(fore_cond[..., 2], [0]).astype(np.uint8)
        back_cond = np.where(mask == 0, 0, back_cond)
        return np.maximum(fore_cond, back_cond).astype(np.float32) / 255.0

    def _process(
        self, sn_depth_image, sn_seg_image, prompt, num_samples, image_resolution_h, image_resolution_w, ddim_steps,
        strength, scale, seed, eta
    ):
        seed_everything(seed)
        if sn_depth_image is not None and isinstance(sn_depth_image, PIL.Image.Image):
            sn_depth_image = np.array(sn_depth_image).astype(np.uint8)
        if sn_seg_image is not None and isinstance(sn_seg_image, PIL.Image.Image):
            sn_seg_image = np.array(sn_seg_image).astype(np.uint8)
        if sn_depth_image is not None:
            anchor_image = sn_depth_image
        elif sn_seg_image is not None:
            anchor_image = sn_seg_image
        else:
            anchor_image = np.zeros((image_resolution_h, image_resolution_w, 3)).astype(np.uint8)
        H, W, C = resize_image(HWC3(anchor_image), image_resolution_h).shape

        with torch.no_grad():
            if sn_depth_image is not None:
                sn_depth_image = cv2.resize(sn_depth_image, (W, H))
                sn_depth_image = sn_depth_image
            else:
                sn_depth_image = np.zeros((H, W, C)).astype(np.uint8)
            if sn_seg_image is not None:
                sn_seg_image = cv2.resize(sn_seg_image, (W, H))
                sn_seg_image = sn_seg_image
            else:
                sn_seg_image = np.zeros((H, W, C)).astype(np.uint8)

            detected_maps_list = [
                sn_depth_image,
                sn_seg_image,
            ]
            self.split_back_ground = False

            def pre_process(condition):
                condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                condition = cv2.resize(
                    condition,
                    dsize=(image_resolution_w, image_resolution_h),
                    fx=1,
                    fy=1,
                    interpolation=cv2.INTER_LINEAR
                )
                condition = condition.astype(np.float32) / 255.0
                return condition

            detected_maps_list = [pre_process(cond) for cond in detected_maps_list]
            mask = np.zeros((H, W, 1))
            if self.split_back_ground:
                # you can chose to remove background (for simulator conditions) and/or customize backgrounds
                conds, mask = self.split_foreground(detected_maps_list[0], detected_maps_list[1])
                detected_maps_list = [self.mix_cond(None, conds[i]) for i in range(len(conds) // 2)]

            detected_maps = np.concatenate(detected_maps_list, axis=2)

            local_control = torch.from_numpy(detected_maps.copy()).float().cuda()
            local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)

            global_control = torch.from_numpy(np.zeros((768))).float().cuda().clone()
            global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

            image = torch.zeros((num_samples, H, W, 3)).cuda()
            anno = [prompt for _ in range(num_samples)]
            mask = torch.from_numpy(mask.copy()).cuda()
            mask = torch.stack([mask for _ in range(num_samples)], dim=0)

            input_dict = dict(
                jpg=image,
                txt=anno,
                local_conditions=local_control,
                global_conditions=global_control,
                mask=mask,
                name="testing pipeline"
            )

            # convert dtype using autocast
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                return_log = self.model.log_images(
                    input_dict,
                    N=num_samples,
                    n_row=1,
                    sample=False,
                    ddim_steps=ddim_steps,
                    ddim_eta=eta,
                    unconditional_guidance_scale=scale,
                    first_cond_model=self.first_cond_model,
                    strength=strength,
                )

            results = return_log[f"samples_cfg_scale_{scale:.2f}"]

            results = results.permute(0, 2, 3, 1).clamp(-1.0, 1.0).to(torch.float32)
            results = (results + 1.0) / 2.0 * 255.0
            results = results.detach().cpu().numpy().astype(np.uint8)
            x_results = [results[i] for i in range(num_samples)]

            local_cond_samples = return_log['local_control'].detach().permute(0, 2, 3, 1)
            local_cond_samples = (local_cond_samples + 1.0) / 2.0 * 255.0
            local_cond_samples = local_cond_samples.detach().cpu().squeeze(0).numpy().astype(np.uint8)
            x_results.append(local_cond_samples[:, :, 3])
            x_results.append(local_cond_samples[:, :, 3:])

        return [x_results, detected_maps_list]

    def __call__(
        self,
        depth_image: PIL.Image.Image | None = None,
        seg_image: PIL.Image.Image | None = None,
        content_image: PIL.Image.Image | None = None,
        prompt: str | None = None,
        seed: int = 2024,
        image_width: int = 768,
        image_height: int = 448,
        num_images_pre_prompt: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        conddiff_strength: float = 0.6,
    ) -> List[PIL.Image.Image] | SimGenOutput:
        # Cond-diff Strength is the stength for SDEdit stength

        images = self._process(
            sn_depth_image=depth_image,
            sn_seg_image=seg_image,
            prompt=prompt,
            num_samples=num_images_pre_prompt,
            image_resolution_h=image_height,
            image_resolution_w=image_width,
            ddim_steps=num_inference_steps,
            strength=conddiff_strength,
            scale=guidance_scale,
            seed=seed,
            eta=0.0,
        )[0]

        return SimGenOutput(images=images)
