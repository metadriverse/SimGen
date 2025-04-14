import sys
if './' not in sys.path:
	sys.path.append('./')
	
from omegaconf import OmegaConf
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger

from ldm.util import instantiate_from_config
from models.util import load_state_dict
from models.logger import ImageLogger
import cv2
import random
import numpy as np
import os
from util import *

parser = argparse.ArgumentParser(description='Uni-ControlNet Training')
parser.add_argument('--config-path', type=str, default='./configs/local_v15.yaml')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--training-steps', type=int, default=5e4)
parser.add_argument('--resume-path', type=str, default='./ckpt/init_local.ckpt')
parser.add_argument('--logdir', type=str, default='./log_local/')
parser.add_argument('--log-freq', type=int, default=1)
parser.add_argument('--sd-locked', type=bool, default=True)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--img', type=str, default=None)
parser.add_argument('--depth', type=str, default=None)
parser.add_argument('--seg', type=str, default=None)
parser.add_argument('--text', type=str, default=None)
args = parser.parse_args()

from torch.utils.data import Dataset

from pick_dataset import UniDataset

class CustomDataset(UniDataset):
    def __init__(self, img, depth, seg, text):
        
        super().__init__()
        self.img = img
        self.depth = depth
        self.seg = seg
        self.text = text

    def __getitem__(self, index):

        data_info = self.data_infos[index]
        img_path, img_conditions = list(data_info.items())[0]
        if self.img is not None:
            img_path = self.img
        image = cv2.imread(img_path)
        image = cv2.resize(image, dsize=self.dsize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        image = (image.astype(np.float32) / 127.5) - 1.0

        if self.text is not None:
            img_conditions["text"] = self.text
        if self.depth is not None:
            img_conditions['depth'] = self.depth
        if self.seg is not None:
            img_conditions['seg'] = self.seg
        if self.text is not None:
            img_conditions['text'] = self.text
        anno = img_conditions["text"]

        local_files = []
        for local_type in self.local_type_list:
            local_files.append(img_conditions[local_type])
        global_files = []
        for global_type in self.global_type_list:
            global_files.append(img_conditions[global_type])

        local_conditions = []
        for local_file in local_files:
            if local_file == "BLANK":
                local_conditions.append(None)
            else:
                condition = cv2.imread(local_file)
                condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                condition = cv2.resize(condition, dsize=self.dsize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                condition = condition.astype(np.float32) / 255.0
                local_conditions.append(condition)

        local_conditions = local_conditions[:2]

        local_conditions[0] = np.ones_like(local_conditions[0]) * np.max(local_conditions[0]) - local_conditions[0]

        global_conditions = []
        for global_file in global_files:
            condition = np.zeros((self.dsize[1], self.dsize[0], 3), dtype=np.float32)
            global_conditions.append(condition)

        index = random.choice([i for i, x in enumerate(local_conditions) if x is not None])

        for i in range(len(local_conditions)):
            if local_conditions[i] is None:
                local_conditions[i] = np.zeros(local_conditions[index].shape, dtype=np.float32)

        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions, mask=np.zeros((local_conditions[index].shape[0], local_conditions[index].shape[1], 1), dtype=np.float32), name=img_path)
    
    def __len__(self):
        return len(self.data_infos)

def main():

    config_path = args.config_path
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_steps = args.training_steps
    resume_path = args.resume_path
    default_logdir = args.logdir
    logger_freq = args.log_freq
    sd_locked = args.sd_locked
    num_workers = args.num_workers
    gpus = args.gpus
    img, depth, seg, text = args.img, args.depth, args.seg, args.text

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config['model'])
    time = config['time'] if 'time' in config else 1

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    # dataset = instantiate_from_config(config['data'])
    # dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    # val_dataset = instantiate_from_config(config['val_data'])
    # val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    if 'first_cond_config' in config:
        first_cond_model = instantiate_from_config(config['first_cond_config'])
        first_cond_model.load_state_dict(load_state_dict(config['first_cond_config']['first_cond_ckpt'], location='cpu'))
        logger = ImageLogger(batch_frequency=logger_freq, num_local_conditions=(config['model']['params']['local_control_config']['params']['local_channels']//3), max_images=6, log_images_kwargs={'N':batch_size*time, 'first_cond_model':first_cond_model})
    else:
        first_cond_model = None
        logger = ImageLogger(batch_frequency=logger_freq, num_local_conditions=(config['model']['params']['local_control_config']['params']['local_channels']//3), max_images=batch_size, is_save=False, log_images_kwargs={'N':batch_size*time})

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=logger_freq,
        save_top_k=-1,
    )
        
    trainer = pl.Trainer(
        gpus=gpus,
        callbacks=[logger, checkpoint_callback], 
        default_root_dir=default_logdir,
        max_steps=training_steps,
        val_check_interval=logger_freq,
    )
    # trainer.fit(model,
    #     dataloader, 
    #     val_dataloader,
    # )
    while True:
        # dataset = instantiate_from_config(config['data'])
        custom_dataset = CustomDataset(img, depth, seg, text)
        dataloader = DataLoader(custom_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
        trainer.validate(model, dataloader)


if __name__ == '__main__':
    main()