import os
import gc

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig, LlavaForConditionalGeneration
from fine_tune_metrics import tune_model

def print_model_parameters(model, message):
    """
    Prints the total number of parameters in the model.

    Args:
        model (nn.Module): The model.
        message (str): A custom message to display.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{message}: {total_params} parameters")
    return total_params


def prune_model_by_layers(model):
    """
    Prune the last layer of the model while keeping the last two layers intact.
    Args:
        model: The model to prune.
    """
    layers_ref = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
    layer_to_prune = len(layers_ref) - 3  

    if 0 <= layer_to_prune < len(layers_ref):
        print(f"Pruning layer {layer_to_prune}...")
        del layers_ref[layer_to_prune]
    else:
        print(f"Layer index {layer_to_prune} is out of range. Skipping.")

    print(f"Model pruned. Remaining layers: {len(layers_ref)}")


def prune_layers(args, iteration, base_dir, eval_metric, tune_eval=False, reference_probs=None, reference_logits=None, reference_representation=None):
    torch.cuda.empty_cache()
    
    print("Loading tokenizer from model path...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning during LLaVA model loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True
        )
    print("Model loaded.")

    len_before_pruning = len(model.language_model.model.layers) if hasattr(model, "language_model") else len(model.model.layers)

    prune_model_by_layers(model)

    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = len_before_pruning - 1
    else:
        model.config.num_hidden_layers = len_before_pruning - 1

    total_params_print = print_model_parameters(model, "Total parameters after pruning")

    if tune_eval:
        training_steps = 10
        if iteration != 1:
            training_steps = iteration * 10

        print(f"CUDA memory allocated before layer eval: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        eval = tune_model(model, processor, training_steps, eval_metric, reference_probs, reference_logits, reference_representation)

        model.to("cpu")
        del model, processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        model = None
        processor = None
        gc.collect()
        print(f"CUDA memory allocated after layer eval: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        return eval
    else:
        prune_model_dir = base_dir + '/' + f'prune_iteration_{iteration}'
        
        os.makedirs(prune_model_dir, exist_ok=True)
        model.save_pretrained(prune_model_dir)
        processor.save_pretrained(prune_model_dir)

        model.to("cpu")
        del model, processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        model = None
        processor = None
        gc.collect()    
