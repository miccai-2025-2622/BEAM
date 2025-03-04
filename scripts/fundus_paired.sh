#!/bin/bash

python src/train.py logger.wandb.name="train_paired_noD_5e-6lr" task_name="train_paired_noD_5e-6lr" trainer.max_epochs=25
