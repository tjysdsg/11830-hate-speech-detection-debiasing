#!/bin/bash

# test on Toxigen data
checkpoint_dir=
data_dir=data

echo "Checkpoint dir: ${checkpoint_dir}"
echo "Data dir: ${data_dir}"

if [ -z "${checkpoint_dir}" ]; then
  echo "Please edit this file to specify checkpoint dir"
  exit 2
fi

python run_model.py \
  --do_eval \
  --do_lower_case \
  --data_dir ${data_dir} \
  --bert_model NA \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --output_dir "${checkpoint_dir}" \
  --task_name tox
