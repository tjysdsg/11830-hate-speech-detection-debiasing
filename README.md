# CMU 11830 Group Project

Based on https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations

# The LM used by hierarchical explanation

SOC debiasing requires a pretrained LM. `run_model.py` will automatically train one if not found in `runs/lm/`.
We provided [a pre-trained LM](runs/lm/best_snapshot_devloss_8.975710184677787_iter_52000_model.pt).

# General notes regarding training scripts

- Set `--grandient_accumulation_steps` if OOM.
- I removed Nvidia Apex cuz it's just not working well in 2023.
  I didn't bother to implement AMP using `torch.amp` as BERT-base is not that large.
- Remove the loop in training scripts if you don't need multiple runs with different seeds

# Training

- `scripts/fou_vanilla.sh` trains a model without debiasing
- `scripts/fou_soc.sh` trains a model with SOC debiasing

# Evaluation

## Hate Speech Classification Performance

- Use `scripts/test_toxigen.sh` to get accuracy and F1 scores on Toxigen. It will also create a prediction file used for
  other tests below.

## Measure biases on Founta

- Run `scripts/test_bias_founta.sh` to get `runs/founta_*/founta_bias_eval.csv`.
  The bias metrics are from https://arxiv.org/abs/2102.00086 

## Measure biases on Toxigen

- Run `scripts/calc_target_group_fpr.sh` to get FPR of each target group in Toxigen. The output is
  in `runs/founta_*/toxigen_fpr.csv`.

## Get word weights of Toxigen samples using SOC explanation for all experiments

- `scripts/explain_soc.sh` will calculate the weight of every word in samples specified
  in `data/toxigen_soc_line_numbers.csv`

