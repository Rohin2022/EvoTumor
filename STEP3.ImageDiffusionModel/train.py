from re import I
import sys, os
sys.path.append(os.getcwd())
from ddpm import Unet3D, GaussianDiffusion, Trainer
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import os
from ddpm.unet import UNet
import time
from dataset.dataloader import get_loader

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels, # image (1) and tumor mask (1)
            out_dim=cfg.model.out_dim,
            num_continuous_conditioners=10,
            num_organs=9
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()

    train_dataloader, train_sampler, dataset_size = get_loader(cfg.dataset)

    val_dataloader=None

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataloader,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
    )

    if cfg.model.load_milestone:
        trainer.load(-1) # load the latest checkpoint

    trainer.train()


if __name__ == '__main__':
    run()
