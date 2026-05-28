import os
import sys
import glob
sys.path.append(os.getcwd())
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from vq_gan_3d.model import VQGAN
from callbacks import ImageLogger, VideoLogger
import hydra
from omegaconf import DictConfig, open_dict
from dataset.dataloader import get_loader
import argparse
import logging
logging.disable(logging.WARNING)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total', total_num/(1024*1024.0), 'Trainable', trainable_num/(1024*1024.0))
    return {'Total': total_num, 'Trainable': trainable_num}


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig, args=None):
    pl.seed_everything(cfg.model.seed)

    cfg.model.default_root_dir = os.path.abspath(cfg.model.default_root_dir)

    train_dataloader, _, _ = get_loader(cfg.dataset)
    val_dataloader=None

    # automatically adjust learning rate
    base_lr = cfg.model.lr

    with open_dict(cfg):
        cfg.model.lr = 1 * (1/8.) * (2/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)

    model = VQGAN(cfg)
    get_parameter_number(model)
    save_step = 500
    callbacks = []
    callbacks.append(ModelCheckpoint(every_n_train_steps=save_step,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=300, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=100, max_images=4, clamp=True))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # load the most recent checkpoint file
    # load the most recent checkpoint file
    ckpt_path = None
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        if cfg.model.resume:
            log_folder = 'version_' + str(cfg.model.resume_version)
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            if os.path.exists(ckpt_folder):
                ckpts = sorted(
                    glob.glob(os.path.join(ckpt_folder, '*.ckpt')),
                    key=os.path.getmtime
                )
                if ckpts:
                    ckpt_path = ckpts[-1]
                    print(f'Resuming from: {ckpt_path}')
                else:
                    print(f'No checkpoints found in {ckpt_folder}')
            else:
                print(f'Checkpoint folder not found: {ckpt_folder}')
        else:
            log_folder = ckpt_file = ''
            version_id_used = step_used = 0
            for folder in os.listdir(base_dir):
                version_id = int(folder.split('_')[1])
                if version_id > version_id_used:
                    version_id_used = version_id
                    log_folder = 'version_' + str(version_id_used + 1)

    if cfg.model.pretrained_checkpoint is not None:
        # Note: Call it on the class (VQGAN) and assign it to the 'model' variable
        model = VQGAN.load_from_checkpoint(cfg.model.pretrained_checkpoint, cfg=cfg)
        print('load pretrained model:', cfg.model.pretrained_checkpoint)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.model.gpus,
        strategy="ddp" if cfg.model.gpus > 1 else "auto",
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        sync_batchnorm=True,
        precision=cfg.model.precision,
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    run()
