import os
import torch
from bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformers import DistilBertForSequenceClassification


def forward_model(model, input_ids, input_mask, segment_ids):
    if isinstance(model, DistilBertForSequenceClassification):
        return model(input_ids, attention_mask=input_mask, labels=None).logits
    else:
        return model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=None).logits


def save_model(args, model, tokenizer):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)
