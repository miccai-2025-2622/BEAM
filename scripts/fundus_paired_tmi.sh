#!/bin/bash

python src/train.py \
    model.learning_rate=1.5e-5 \
    model.lora_rank_vae=8 \
    model.lora_rank_unet=128 \
    logger.wandb.name="train_paired_1.5e-5lr_unetlora_128_hfc_tmi_prompt_full_modeling" \
    task_name="train_paired_1.5e-5lr_unetlora_128_hfc_tmi_prompt_full_modeling" \
    trainer.max_epochs=30 \
    trainer.devices=8 \
    data.type="TMI"
