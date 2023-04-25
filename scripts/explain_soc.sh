#!/usr/bin/env bash

for d in founta_dbert founta_dbert_soc founta_embed_dbert founta_embed_dbert_soc founta_embed_dbert_soc_finetune_tox; do
  python run_model.py \
    --do_eval --explain \
    --do_lower_case --data_dir data \
    --bert_model distilbert-base-uncased \
    --max_seq_length 128 --eval_batch_size 1 --train_batch_size 16 \
    --gradient_accumulation_steps 2 --learning_rate 2e-5 --num_train_epochs 20 \
    --early_stop 5 \
    --seed 42 \
    --hiex_add_itself --reg_explanations --nb_range 5 --sample_n 5 --negative_weight 0.1 --reg_strength 0.1 \
    --task_name tox \
    --algo soc \
    --hiex_idxs data/toxigen_soc_line_numbers.json \
    --output_dir runs/${d} \
    --output_filename soc_word_level_explain.txt
done
