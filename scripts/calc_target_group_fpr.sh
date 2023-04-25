#!/usr/bin/env bash

for d in founta_dbert founta_dbert_soc founta_embed_dbert founta_embed_dbert_soc founta_embed_dbert_soc_finetune_tox; do
  python target_group_fpr.py runs/${d}/eval_details_0_dev_tox.txt runs/${d}/toxigen_fpr.csv
done
