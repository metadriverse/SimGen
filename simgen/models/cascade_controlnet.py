import einops
import torch

from einops import rearrange, repeat
from torchvision.utils import make_grid

from simgen.ldm.models.diffusion.ddpm import LatentDiffusion
from simgen.ldm.util import log_txt_as_img, instantiate_from_config
from simgen.ldm.models.diffusion.ddim import DDIMSampler
import numpy as np

COLOR_DICT_CITY = np.array(
    [
        [0, 0, 0],  # 0: 'ignore'
        [128, 64, 128],  # 1: 'road'
        [232, 35, 244],  # 2: 'sidewalk'
        [70, 70, 70],  # 3: 'building'
        [156, 102, 102],  # 4: 'wall'
        [153, 153, 190],  # 5: 'fence'
        [153, 153, 153],  # 6: 'pole'
        [30, 170, 250],  # 7: 'traffic light'
        [0, 220, 220],  # 8: 'traffic sign'
        [35, 142, 107],  # 9: 'vegetation'
        [152, 251, 152],  # 10: 'terrain'
        [180, 130, 70],  # 11: 'sky'
        [60, 20, 220],  # 12: 'person'
        [0, 0, 255],  # 13: 'rider'
        [142, 0, 0],  # 14: 'car'
        [70, 0, 0],  # 15: 'truck'
        [100, 60, 0],  # 16: 'bus'
        [100, 80, 0],  # 17: 'train'
        [230, 0, 0],  # 18: 'motorcycle'
        [32, 11, 119],  # 19: 'bicycle'
        [189, 176, 55],  # 20: 'cross work'
        [255, 255, 255],  # 21: 'lane line'
    ]
)
COLOR_DICT_CITY[:, [2, 0]] = COLOR_DICT_CITY[:, [0, 2]]

COLOR_DICT_FORE = np.array(
    [
        [0, 0, 0],  # 0: 'ignore'
        [128, 64, 128],  # 1: 'road'
        [153, 153, 190],  # 2: 'fence'
        [60, 20, 220],  # 3: 'person'
        [142, 0, 0],  # 4: 'car'
        [70, 0, 0],  # 5: 'truck'
        [100, 60, 0],  # 6: 'bus'
        [230, 0, 0],  # 7: 'motorcycle'
        [32, 11, 119],  # 8: 'bicycle'
        [0, 0, 0],  # 9: 'ignore'
    ]
)

COLOR_DICT_FORE[:, [2, 0]] = COLOR_DICT_FORE[:, [0, 2]]


class UniControlNet(LatentDiffusion):
    def __init__(self, mode, local_control_config=None, global_control_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['local', 'global', 'uni']
        self.mode = mode
        if self.mode in ['local', 'uni']:
            self.local_adapter = instantiate_from_config(local_control_config)
            self.local_control_scales = [1.0] * 13
        if self.mode in ['global', 'uni']:
            self.global_adapter = instantiate_from_config(global_control_config)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        if len(batch['local_conditions']) != 0:
            local_conditions = batch['local_conditions']
            if bs is not None:
                local_conditions = local_conditions[:bs]
            local_conditions = local_conditions.to(self.device)
            local_conditions = einops.rearrange(local_conditions, 'b h w c -> b c h w')
            local_conditions = local_conditions.to(memory_format=torch.contiguous_format).float()
        else:
            local_conditions = torch.zeros(1, 1, 1, 1).to(self.device).to(memory_format=torch.contiguous_format).float()
        if len(batch['global_conditions']) != 0:
            global_conditions = batch['global_conditions']
            if bs is not None:
                global_conditions = global_conditions[:bs]
            global_conditions = global_conditions.to(self.device).to(memory_format=torch.contiguous_format).float()
        else:
            global_conditions = torch.zeros(1, 1).to(self.device).to(memory_format=torch.contiguous_format).float()
        if len(batch['mask']) != 0:
            mask = batch['mask']
            if bs is not None:
                mask = mask[:bs]
            mask = mask.to(self.device)
            mask = einops.rearrange(mask, 'b h w c -> b c h w')
            mask = mask.to(memory_format=torch.contiguous_format).float()
        else:
            mask = torch.zeros(1, 1, 1, 1).to(self.device).to(memory_format=torch.contiguous_format).float()

        return x, dict(
            c_crossattn=[c], local_control=[local_conditions], global_control=[global_conditions], mask=[mask]
        )

    def apply_model(self, x_noisy, t, cond, global_strength=1, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if self.mode in ['global', 'uni']:
            assert cond['global_control'][0] != None
            global_control = self.global_adapter(cond['global_control'][0])
            cond_txt = torch.cat([cond_txt, global_strength * global_control], dim=1)
        if self.mode in ['local', 'uni']:
            assert cond['local_control'][0] != None
            local_control = torch.cat(cond['local_control'], 1)
            local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, local_conditions=local_control)
            local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]

        if self.mode == 'global':
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)
        else:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        plot_denoise_rows=False,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        first_cond_model=None,
        split='val',
        strength=0.6,
        **kwargs
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat = c["local_control"][0][:N]
        c_global = c["global_control"][0][:N]
        mask = c["mask"][0][:N]
        c = c["c_crossattn"][0][:N]

        log["origin_depth"] = c_cat[:, :3, ...] * 2.0 - 1.0
        log["origin_seg"] = c_cat[:, 3:6, ...] * 2.0 - 1.0
        is_train = split == 'train'
        # is_train = True
        if not is_train:
            c_cat = self.split_cond(c_cat, mask)
            sn_img = self.encode_cond(c_cat)

            uc_cross = self.get_unconditional_conditioning(N)
            uc_full = {"c_crossattn": [uc_cross]}
            first_cond_model = first_cond_model.to(self.device)
            samples_cfg, _ = first_cond_model.sample_log(
                cond={"c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
                x_T=sn_img,
                strength=strength,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)

            c_cat = self.decode_cond(x_samples_cfg, c_cat)
            c_cat = self.mix_cond(c_cat, mask)

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["local_control"] = c_cat * 2.0 - 1.0

        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(
                cond={
                    "local_control": [c_cat],
                    "c_crossattn": [c],
                    "global_control": [c_global]
                },
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_global = torch.zeros_like(c_global)
            uc_full = {"local_control": [uc_cat], "c_crossattn": [uc_cross], "global_control": [uc_global]}
            samples_cfg, _ = self.sample_log(
                cond={
                    "local_control": [c_cat],
                    "c_crossattn": [c],
                    "global_control": [c_global]
                },
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def encode_cond(self, cond):

        cond = (cond[:, :6, ...] * 255).cpu().numpy().astype(int)
        cond = einops.rearrange(cond, 'b c h w -> b h w c')  # [0-1]
        output_cond = np.zeros((cond.shape[0], cond.shape[1], cond.shape[2], 3), dtype=int)

        output_cond[:, :, :, 0] = cond[:, :, :, 0]  # depth
        seg_image = cond[:, :, :, 3:6]

        # print(seg_image[0])

        # colormap = COLOR_DICT_CITY
        colormap = COLOR_DICT_FORE

        seg_image = einops.rearrange(seg_image, 'b h w c -> b (h w) c')
        colormap_reshaped = colormap.reshape((1, -1, 3))
        distances = np.linalg.norm(seg_image[:, :, np.newaxis, :] - colormap_reshaped, axis=-1)
        index_matrix = np.argmin(distances, axis=-1)
        index_matrix = einops.rearrange(index_matrix, 'b (h w) -> b h w', h=cond.shape[1])

        # index_matrix = np.where(index_matrix == 21, 0, index_matrix) # set ignore to road, will be fixed in next version

        # print(index_matrix[0])
        # input()

        # output_cond[:, :, :, 1] = index_matrix*12
        output_cond[:, :, :, 1] = (index_matrix + 1) * 28
        # output_cond = np.where(mask == 0, 0, output_cond)
        output_cond[:, :, :, 2] = 127

        output_cond = output_cond.astype(np.float32) / 255.0

        output_cond = output_cond * 2.0 - 1.0  # [-1, 1]
        output_cond = einops.rearrange(output_cond, 'b h w c -> b c h w')
        output_cond = torch.tensor(output_cond).to(self.device).to(memory_format=torch.contiguous_format).float()
        # output_cond = torch.tensor(output_cond).to(memory_format=torch.contiguous_format).float()
        return output_cond

    @torch.no_grad()
    def decode_cond(self, cond, c_cat):
        cond = cond.cpu().numpy().astype(np.float32)
        cond = einops.rearrange(cond, 'b c h w -> b h w c')
        cond = (cond + 1.0) / 2.0  # [0-1]
        cond = (cond * 255.0).astype(int)

        depth = cond[:, :, :, 0]
        depth = np.clip(depth, 0, 255)
        seg_index = cond[:, :, :, 1]
        seg_index = np.clip(seg_index, 0, 255)
        # seg_index[(seg_index > 253) | (seg_index < 0)] = 270
        # print(depth)
        # print(seg_index)
        # print((seg_index/13.0).astype(int))

        # colormap = COLOR_DICT_CITY
        colormap = COLOR_DICT_FORE
        # seg_index = np.round(seg_index.astype(float)/12.0).astype(int)
        seg_index = np.round(seg_index.astype(float) / 28.0 - 1).astype(int)

        seg_index = np.where(seg_index == 21, 0, seg_index)

        seg_image = colormap[seg_index]
        # seg_index = np.where(np.isin(seg_index, [10, 11, 12]), 12, seg_index)
        # seg_index = np.where(np.isin(seg_index, [13, 14]), 14, seg_index)
        # seg_index = np.where(np.isin(seg_index, [16, 17]), 16, seg_index)
        # seg_index = np.where(np.isin(seg_index, [18, 19]), 19, seg_index)

        output_cond = np.zeros((cond.shape[0], cond.shape[1], cond.shape[2], 6), dtype=int)
        output_cond[:, :, :, 0] = depth
        output_cond[:, :, :, 1] = depth
        output_cond[:, :, :, 2] = depth
        output_cond[:, :, :, 3:] = seg_image

        output_cond = output_cond.astype(np.float32) / 255.0
        output_cond = einops.rearrange(output_cond, 'b h w c -> b c h w')
        output_cond = torch.tensor(output_cond).to(self.device).to(memory_format=torch.contiguous_format).float()

        if c_cat.shape[1] > 6:
            output_cond = torch.cat([output_cond, c_cat[:, 6:, ...]], 1)
        return output_cond

    @torch.no_grad()
    def split_cond(self, cond, mask):

        cond = cond.cpu().numpy().astype(np.float32)
        cond = einops.rearrange(cond, 'b c h w -> b h w c')

        fore_depth, back_depth, fore_seg, back_seg = cond[:, :, :, :3].copy(), cond[:, :, :, :3].copy(
        ), cond[:, :, :, 3:6].copy(), cond[:, :, :, 3:6].copy()

        colormap = COLOR_DICT_CITY
        colormap_reshaped = colormap.reshape((1, -1, 3))
        seg_image = fore_seg.copy()
        seg_image = (seg_image * 255).astype(np.uint8)
        seg_image = einops.rearrange(seg_image, 'b h w c -> b (h w) c')
        distances = np.linalg.norm(seg_image[:, :, np.newaxis, :] - colormap_reshaped, axis=-1)
        index_matrix = np.argmin(distances, axis=-1)
        index_matrix = einops.rearrange(index_matrix, 'b (h w) -> b h w', h=cond.shape[1])

        # mask = np.zeros((cond.shape[0], cond.shape[1], cond.shape[2], 1), dtype=np.uint8)
        # foreground = [0,4,11,12,13,14,15,16,17,18,19,20]
        # mask[..., 0] = np.isin(index_matrix, foreground).astype(np.uint8)
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = einops.rearrange(mask, 'b c h w -> b h w c')

        fore_depth = np.where(mask == 0, 0, fore_depth)
        back_depth = np.where(mask == 1, 0, back_depth)
        fore_seg = np.where(mask == 0, 0, fore_seg)
        back_seg = np.where(mask == 1, 0, back_seg)

        if cond.shape[3] > 6:
            output_cond = np.concatenate([fore_depth, fore_seg, back_depth, back_seg, cond[:, :, :, 6:]], axis=3)
        else:
            output_cond = np.concatenate([fore_depth, fore_seg, back_depth, back_seg], axis=3)
        output_cond = einops.rearrange(output_cond, 'b h w c -> b c h w')
        output_cond = torch.tensor(output_cond).to(self.device).to(memory_format=torch.contiguous_format).float()

        return output_cond

    @torch.no_grad()
    def mix_cond(self, cond, mask):

        cond = cond.cpu().numpy().astype(np.float32)
        cond = einops.rearrange(cond, 'b c h w -> b h w c')

        fore_depth, fore_seg, back_depth, back_seg = cond[:, :, :, :3], cond[:, :, :, 3:6], cond[:, :, :,
                                                                                                 6:9], cond[:, :, :,
                                                                                                            9:12]

        # mask = np.zeros((cond.shape[0], cond.shape[1], cond.shape[2], 1), dtype=np.uint8)

        # fore_cond = fore_depth.copy()
        # back_cond = back_depth.copy()
        # fore_cond = (fore_cond*255).astype(np.uint8)
        # back_cond = (back_cond*255).astype(np.uint8)
        # mask[..., 0] = np.isin(fore_cond[..., 0], [0]).astype(np.uint8) & np.isin(fore_cond[..., 1], [0]).astype(np.uint8) & np.isin(fore_cond[..., 2], [0]).astype(np.uint8)

        mask = mask.cpu().numpy().astype(np.uint8)
        mask = einops.rearrange(mask, 'b c h w -> b h w c')

        back_depth = np.where(mask == 1, 0, back_depth)
        back_seg = np.where(mask == 1, 0, back_seg)

        output_depth = np.maximum(fore_depth, back_depth)
        output_seg = np.maximum(fore_seg, back_seg)

        if cond.shape[3] > 12:
            output_cond = np.concatenate([output_depth, output_seg, cond[:, :, :, 12:]], axis=3)
        else:
            output_cond = np.concatenate([output_depth, output_seg], axis=3)
        output_cond = einops.rearrange(output_cond, 'b h w c -> b c h w')
        output_cond = torch.tensor(output_cond).to(self.device).to(memory_format=torch.contiguous_format).float()

        return output_cond

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        if self.mode == 'global':
            h, w = 512, 512
        else:
            _, _, h, w = cond["local_control"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.mode in ['local', 'uni']:
            params += list(self.local_adapter.parameters())
        if self.mode in ['global', 'uni']:
            params += list(self.global_adapter.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cuda()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cpu()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
