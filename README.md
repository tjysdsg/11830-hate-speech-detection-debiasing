# CMU 11830 Group Project

Based on https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations

# The LM used by hierarchical explanation

SOC debiasing requires a pretrained LM. `run_model.py` will automatically train one if not found in `runs/lm/`.

# General notes regarding training scripts

- Set `--grandient_accumulation_steps` if OOM.
- I removed Nvidia Apex cuz it's just not working well in 2023.
  I didn't bother to implement AMP using `torch.amp` as BERT-base is not that large.
- Remove the loop in training scripts if you don't need multiple runs with different seeds

# (Jiyang) Reproducing the original GAB data experiments

Using the original scripts:

- `scripts/gab_vanilla.sh` trains a model without debiasing
- `scripts/gab_soc.sh` trains a model with SOC debiasing

# Get word weights of Toxigen samples using SOC explanation for all experiments

- `scripts/explain_soc.sh`
