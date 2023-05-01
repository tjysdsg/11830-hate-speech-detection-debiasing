#!/bin/bash

# training with regularizing SOC explanations
current_seed=42
reg_strength=0.1
data_dir=data

echo "Data dir: ${data_dir}"
echo "reg_strength is ${reg_strength}"

soc_options="--hiex_add_itself --reg_explanations --nb_range 5 --sample_n 5 --negative_weight 0.1 --reg_strength ${reg_strength}"

python run_model.py \
  --do_train --do_lower_case \
  --data_dir ${data_dir} \
  --bert_model distilbert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --early_stop 5 \
  --output_dir runs/founta_soc \
  --seed ${current_seed} \
  --task_name fou \
  ${soc_options}
