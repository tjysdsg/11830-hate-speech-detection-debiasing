#!/bin/bash

# training on Toxigen data without regularization
current_seed=42
data_dir=data
echo "Data dir: ${data_dir}"

python run_model.py \
  --do_train --do_lower_case \
  --data_dir ${data_dir} \
  --valid_step 200 \
  --bert_model distilbert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir runs/tox_dbert \
  --seed ${current_seed} \
  --task_name tox
