import os
from processor.trainers.do_train_person_reid import Person_Reid
from metrics.r1_mAP import R1_mAP, R1_mAP_reranking, R1_mAP_extend
from models import *
from dataset_processing import make_data_loader
from utils.opt import *
import utils.utils as util
from losses import *
from solver import make_optimizer, make_scheduler
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor


############################# Main ##################################################################################
def train(cfg):
    # print(cfg["MODEL"])
    checkpoint_path = None if cfg["CHECKPOINT_PATH"] == "" else cfg["CHECKPOINT_PATH"]
    print(checkpoint_path)
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    model = build_model(cfg, num_classes)

    loss_fn = make_loss(cfg, num_classes)
    optimizer = make_optimizer(cfg["SOLVER"], model)
    scheduler = make_scheduler(cfg["SOLVER"], optimizer)
    evaluator = R1_mAP(num_query=num_query, max_rank=50, feat_norm="yes", is_cross_cam = True)

    early_stop_callback = EarlyStopping(
        monitor="Val_mAP",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        every_n_epochs=1,
        monitor="Val_mAP",
        mode="max",
        filename="{epoch}-{Val_mAP:.5f}-{Val_CMC@rank1:.5f}-{Val_CMC@rank5:.5f}",
    )

    person_reid = Person_Reid(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        cfg=cfg,
    )
    if checkpoint_path is not None:
        person_reid = person_reid.load_from_checkpoint(
            checkpoint_path,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluator=evaluator,
            cfg=cfg,
        )
    person_reid.scheduler = make_scheduler(cfg["SOLVER"], optimizer)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        accelerator=cfg["ACCELERATOR"],
        devices=[0],
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=cfg["SOLVER"]["MAX_EPOCHS"],
        check_val_every_n_epoch=cfg["SOLVER"]["EVAL_PERIOD"],
        num_sanity_val_steps=-1,  # -1 for validation all, 0 for not validation
        # resume_from_checkpoint=checkpoint_path,
        default_root_dir=cfg["OUTPUT_DIR"] + '/reid/',
        enable_progress_bar=False,  # TQDM
        # replace_sampler_ddp=False, # set it to false when multiple gpus
        # reload_dataloaders_every_n_epochs = 15
    )
    print(cfg["SOLVER"])

    trainer.fit(
        person_reid,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    #############################  Config processing  ##################################################################################
    cfg = util.load_defaults(["configs/dataset.yaml", "configs/transformer.yaml"])

    # set seed
    util.set_seed(cfg["SOLVER"]["SEED"])
    #############################################################

    train(cfg)