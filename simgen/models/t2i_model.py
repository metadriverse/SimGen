import einops
import torch

from einops import rearrange, repeat
from torchvision.utils import make_grid

from simgen.ldm.models.diffusion.ddpm import LatentDiffusion
from simgen.ldm.util import log_txt_as_img
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


class T2IModel(LatentDiffusion):

    # def __init__(self, mode, local_control_config=None, global_control_config=None, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     assert mode in ['local', 'global', 'uni']
    #     self.mode = mode
    #     if self.mode in ['local', 'uni']:
    #         self.local_adapter = instantiate_from_config(local_control_config)
    #         self.local_control_scales = [1.0] * 13
    #     if self.mode in ['global', 'uni']:
    #         self.global_adapter = instantiate_from_config(global_control_config)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        ori_img = batch['ori_img']
        sn_img = batch['sn_img']
        if bs is not None:
            ori_img = ori_img[:bs]
            sn_img = sn_img[:bs]
        ori_img = ori_img.to(self.device)
        ori_img = einops.rearrange(ori_img, 'b h w c -> b c h w')
        ori_img = ori_img.to(memory_format=torch.contiguous_format).float()
        sn_img = sn_img.to(self.device)
        sn_img = einops.rearrange(sn_img, 'b h w c -> b c h w')
        sn_img = sn_img.to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], ori_img=[ori_img], sn_img=[sn_img])

    def apply_model(self, x_noisy, t, cond, global_strength=1, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)
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
        strength=0.75,
        **kwargs
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        ori_img = c["ori_img"][0][:N]
        sn_img = c["sn_img"][0][:N]
        c = c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        # reconstruction = self.decode_first_stage(z)
        # log['origin_depth'] = reconstruction[:,:3,...]
        # log['origin_seg'] = reconstruction[:,3:,...]
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        log["ori_img"] = ori_img
        # log["sn_img"] = sn_img
        log["local_control"] = self.decode_cond(sn_img)  # [-1,1]
        # log["local_control"] = sn_img

        # res = log["reconstruction"].cpu().numpy()
        # print(np.min(res), np.max(res), np.mean(res))

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
                cond={"c_crossattn": [c]}, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_full = {"c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(
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
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            decoded_img = self.decode_cond(x_samples_cfg, flip=False)
            # decoded_img = x_samples_cfg

            log['output_depth'] = decoded_img[:, :3, ...]
            log['output_seg'] = decoded_img[:, 3:, ...]

        return log

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        decoded_z = self.first_stage_model.decode(z)
        return decoded_z

    @torch.no_grad()
    def encode_first_stage(self, x):
        encoded_x = self.first_stage_model.encode(x)
        return encoded_x

    @torch.no_grad()
    def decode_cond(self, cond, flip=True):

        # cond = torch.clamp(cond, -1.0, 1.0)
        cond = cond.cpu().numpy().astype(np.float32)
        cond = einops.rearrange(cond, 'b c h w -> b h w c')
        # print(np.min(cond), np.max(cond))
        cond = (cond + 1.0) / 2.0  # [0-1]
        cond = (cond * 255.0).astype(int)
        # print(np.min(cond), np.max(cond))
        # print(cond)

        depth = cond[:, :, :, 0]
        # print(np.min(depth), np.max(depth), np.mean(depth))
        depth = np.clip(depth, 0, 255)
        seg_index = cond[:, :, :, 1]
        # print(np.min(seg_index), np.max(seg_index), np.mean(seg_index))
        seg_index = np.clip(seg_index, 0, 255)
        std = cond[:, :, :, 2]
        # print(np.min(std), np.max(std), np.mean(std))
        # seg_index[(seg_index > 253) | (seg_index < 0)] = 270
        # print(depth)
        # print(seg_index)
        # print((seg_index/13.0).astype(int))
        # colormap = COLOR_DICT_CITY
        colormap = COLOR_DICT_FORE
        # seg_index = np.round(seg_index.astype(float)/12.0).astype(int)
        seg_index = np.round(seg_index.astype(float) / 28.0 - 1).astype(int)

        # seg_index = np.where(seg_index == 21, 0, seg_index)

        # # seg_index = np.where(np.isin(seg_index, [1, 2, 3]), 1, seg_index)
        # # seg_index = np.where(np.isin(seg_index, [4, 5, 6, 7, 8, 9]), 5, seg_index)
        # seg_index = np.where(np.isin(seg_index, [10, 11, 12]), 12, seg_index)
        # seg_index = np.where(np.isin(seg_index, [13, 14]), 14, seg_index)
        # seg_index = np.where(np.isin(seg_index, [16, 17]), 16, seg_index)
        # seg_index = np.where(np.isin(seg_index, [18, 19]), 19, seg_index)

        seg_image = colormap[seg_index]

        output_cond = np.zeros((cond.shape[0], cond.shape[1], cond.shape[2], 6), dtype=int)
        output_cond[:, :, :, 0] = depth
        output_cond[:, :, :, 1] = depth
        output_cond[:, :, :, 2] = depth
        output_cond[:, :, :, 3:] = seg_image

        output_cond = output_cond.astype(np.float32) / 255.0
        output_cond = (output_cond * 2.0) - 1.0
        output_cond = einops.rearrange(output_cond, 'b h w c -> b c h w')
        output_cond = torch.tensor(output_cond).to(self.device).to(memory_format=torch.contiguous_format).float()

        return output_cond

    @torch.no_grad()
    def sample_log(
        self, cond, batch_size, ddim, ddim_steps, eta, x_T, strength, unconditional_guidance_scale,
        unconditional_conditioning, **kwargs
    ):
        ddim_sampler = DDIMSampler(self)
        # _, _, h, w = cond.shape
        h, w = 896 // 2, 1536 // 2
        shape = (self.channels, h // 8, w // 8)

        init_image = x_T
        init_latent = self.get_first_stage_encoding(self.encode_first_stage(init_image))

        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        # strength = 0.75
        t_enc = int(strength * ddim_steps)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        x_T = ddim_sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))

        c = cond
        samples = ddim_sampler.decode(
            x_T,
            c,
            t_enc,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )

        # x_samples = self.decode_first_stage(samples)

        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, x_T=x_T, **kwargs)
        return samples, None
        # return samples, intermediates

    # @torch.no_grad()
    # def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
    #     ddim_sampler = DDIMSampler(self)
    #     if self.mode == 'global':
    #         h, w = 512, 512
    #     else:
    #         _, _, h, w = cond["local_control"][0].shape
    #     shape = (self.channels, h // 8, w // 8)
    #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
    #     return samples, intermediates

    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     params = []
    #     if self.mode in ['local', 'uni']:
    #         params += list(self.local_adapter.parameters())
    #     if self.mode in ['global', 'uni']:
    #         params += list(self.global_adapter.parameters())
    #     if not self.sd_locked:
    #         params += list(self.model.diffusion_model.output_blocks.parameters())
    #         params += list(self.model.diffusion_model.out.parameters())
    #     opt = torch.optim.AdamW(params, lr=lr)
    #     return opt

    # def low_vram_shift(self, is_diffusing):
    #     if is_diffusing:
    #         self.model = self.model.cuda()
    #         if self.mode in ['local', 'uni']:
    #             self.local_adapter = self.local_adapter.cuda()
    #         if self.mode in ['global', 'uni']:
    #             self.global_adapter = self.global_adapter.cuda()
    #         self.first_stage_model = self.first_stage_model.cpu()
    #         self.cond_stage_model = self.cond_stage_model.cpu()
    #     else:
    #         self.model = self.model.cpu()
    #         if self.mode in ['local', 'uni']:
    #             self.local_adapter = self.local_adapter.cpu()
    #         if self.mode in ['global', 'uni']:
    #             self.global_adapter = self.global_adapter.cpu()
    #         self.first_stage_model = self.first_stage_model.cuda()
    #         self.cond_stage_model = self.cond_stage_model.cuda()
