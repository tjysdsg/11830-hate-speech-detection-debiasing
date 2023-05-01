#!/usr/bin/env bash

pred=     # runs/founta_dbert/eval_details_2000_dev_fou.txt
out_dir=  # runs/founta_dbert/

if [ -z "${pred}" ] || [ -z "${out_dir}" ]; then
  echo "Please edit this file to specify the prediction file and the output directory."
  exit 2
fi

python eval_bias_founta.py \
  --data data/founta_test.csv \
  --word-list data/word_based_bias_list.csv \
  --prediction ${pred} \
  --out-dir ${out_dir}
