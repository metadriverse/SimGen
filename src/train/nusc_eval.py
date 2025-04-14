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
args = parser.parse_args()


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
        dataset = instantiate_from_config(config['data'])
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
        trainer.validate(model, dataloader)


if __name__ == '__main__':
    main()