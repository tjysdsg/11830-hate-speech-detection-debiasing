#!/usr/bin/env bash

# training on Founta using a debiased distilbert
# some layers are frozen

current_seed=42
data_dir=data
echo "Data dir: ${data_dir}"

python run_model.py \
  --do_train --do_lower_case \
  --freeze \
  --data_dir ${data_dir} \
  --bert_model context-debias/debiased_models/42/dbert \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir runs/founta_embed_dbert \
  --seed ${current_seed} \
  --task_name fou
