import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig, LlavaForConditionalGeneration
import random
import numpy as np
import pickle
import os
import gc
import argparse
from amp_criterion import measure_amp_heads_importance, measure_amp_mlps_importance, remove_all_hooks
from fine_tune_metrics import tune_model


def get_llm(model):
    """
    Return the inner language model if this is a LLaVA model; otherwise the model itself.
    """
    return getattr(model, "language_model", model)


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


def prune_model(model, tokenizer, final_pruning_ratio):
    """
    Prunes the model's attention heads and MLP units based on the specified pruning_ratio.
    The pruning is done using AMP (Activation Magnitude Pruning) to determine importance.

    Args:
        model (nn.Module): The language model to be pruned.
        final_pruning_ratio (float): The fraction of total parameters to prune (between 0 and 1).

    Returns:
        nn.Module: The pruned model.
    """
    total_attention_params = 0
    total_mlp_params = 0

    layers_ref = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
    for layer in layers_ref:
        total_attention_params += sum(p.numel() for p in layer.self_attn.parameters())
        total_mlp_params += sum(p.numel() for p in layer.mlp.parameters())

    total_params = total_attention_params + total_mlp_params

    actual_total_params = sum(p.numel() for p in model.parameters())

    pruning_ratio = (actual_total_params * final_pruning_ratio) / (total_params)

    llm = get_llm(model)
    amp_head_importances = measure_amp_heads_importance(
        llm,
        tokenizer,
        arch="llama",
        dataset_name="scienceqa",
        split="train",
        max_prompts=128,
        random_subset=True
    )
    remove_all_hooks()

    amp_mlps_importance = measure_amp_mlps_importance(
        llm,
        tokenizer,
        arch="llama",
        dataset_name="scienceqa",
        split="train",
        max_prompts=128,
        random_subset=True
    )
    remove_all_hooks()

    for layer_num, layer in enumerate(model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers):
        num_heads = layer.self_attn.num_heads
        num_prune_heads = round(pruning_ratio * num_heads)

        if num_prune_heads >= num_heads:
            num_prune_heads = num_heads - 1

        layer_heads_importance = amp_head_importances[layer_num]
        sorted_heads_indices = np.argsort(layer_heads_importance)

        heads_to_prune = sorted_heads_indices[:num_prune_heads]
        prune_attention_heads(layer.self_attn, heads_to_prune)

        intermediate_size = layer.mlp.gate_proj.out_features
        num_prune_mlp_units = round(intermediate_size*((num_heads / intermediate_size) * (pruning_ratio - (num_prune_heads / num_heads)) + pruning_ratio))

        if num_prune_mlp_units >= intermediate_size:
            num_prune_mlp_units = intermediate_size - 1
        
        layer_mlps_importance = amp_mlps_importance[layer_num]
        sorted_mlps_indices = np.argsort(layer_mlps_importance)

        mlp_units_to_prune = sorted_mlps_indices[:num_prune_mlp_units]
        prune_mlp_units(layer.mlp, mlp_units_to_prune)

    return model


def prune_attention_heads(attention_layer, heads_to_prune):
    """
    Prunes specified attention heads in the attention layer.

    Args:
        attention_layer (nn.Module): The attention layer to prune.
        heads_to_prune (list): Indices of the heads to prune.
    """
    num_heads = attention_layer.num_heads
    head_dim = attention_layer.head_dim

    heads_to_keep = sorted(set(range(num_heads)) - set(heads_to_prune))

    idxs_to_keep = []
    for head in heads_to_keep:
        idxs = list(range(head * head_dim, (head + 1) * head_dim))
        idxs_to_keep.extend(idxs)
    idxs_to_keep = sorted(idxs_to_keep)

    attention_layer.q_proj.weight = torch.nn.Parameter(
        attention_layer.q_proj.weight.data[idxs_to_keep, :]
    )

    attention_layer.k_proj.weight = torch.nn.Parameter(
        attention_layer.k_proj.weight.data[idxs_to_keep, :]
    )

    attention_layer.v_proj.weight = torch.nn.Parameter(
        attention_layer.v_proj.weight.data[idxs_to_keep, :]
    )

    attention_layer.o_proj.weight = torch.nn.Parameter(
        attention_layer.o_proj.weight.data[:, idxs_to_keep]
    )

    attention_layer.num_heads = len(heads_to_keep)
    attention_layer.hidden_size = attention_layer.num_heads * attention_layer.head_dim

    attention_layer.q_proj.out_features = attention_layer.hidden_size
    attention_layer.k_proj.out_features = attention_layer.hidden_size
    attention_layer.v_proj.out_features = attention_layer.hidden_size
    attention_layer.o_proj.in_features = attention_layer.hidden_size

    return


def prune_mlp_units(mlp_layer, units_to_prune):
    """
    Prunes specified units in the MLP layer.

    Args:
        mlp_layer (nn.Module): The MLP layer to prune.
        units_to_prune (list): Indices of the units to prune.
    """
    keep_idxs = sorted(set(range(mlp_layer.gate_proj.out_features)) - set(units_to_prune))

    mlp_layer.gate_proj.out_features = len(keep_idxs)
    mlp_layer.gate_proj.weight = torch.nn.Parameter(
        mlp_layer.gate_proj.weight.data[keep_idxs, :]
    )
 
    mlp_layer.up_proj.out_features = len(keep_idxs)
    mlp_layer.up_proj.weight = torch.nn.Parameter(
        mlp_layer.up_proj.weight.data[keep_idxs, :]
    )

    mlp_layer.down_proj.in_features = len(keep_idxs)
    mlp_layer.down_proj.weight = torch.nn.Parameter(
        mlp_layer.down_proj.weight.data[:, keep_idxs]
    )

    return


def prune_groups(args, iteration, base_dir, eval_metric, tune_eval=False, reference_probs=None, reference_logits=None, reference_representation=None):
    """
    Main function to load the model, perform pruning, update the configuration, and save the pruned model.
    """
    torch.cuda.empty_cache()
    gc.collect()
    
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

    print("Loading tokenizer from model path...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    
    torch.cuda.empty_cache()

    total_params = print_model_parameters(model, "Total parameters before pruning")
    layers_ref = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
    params_to_prune = sum(p.numel() for p in layers_ref[0].parameters())

    final_pruning_ratio = params_to_prune / total_params
    print("Iteration pruning ratio: ", final_pruning_ratio)
    model = prune_model(model, processor.tokenizer, final_pruning_ratio)

    layers_ref = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
    new_num_heads = layers_ref[0].self_attn.num_heads
    new_intermediate_size = layers_ref[0].mlp.gate_proj.out_features

    if hasattr(model.config, "text_config"):
        model.config.text_config.num_attention_heads = new_num_heads
        model.config.text_config.num_key_value_heads = new_num_heads
        model.config.text_config.intermediate_size = new_intermediate_size
    else:
        model.config.num_attention_heads = new_num_heads
        model.config.num_key_value_heads = new_num_heads
        model.config.intermediate_size = new_intermediate_size

    for layer in layers_ref:
        if hasattr(layer.self_attn, "num_key_value_heads"):
            layer.self_attn.num_key_value_heads = new_num_heads

    print(f"CUDA memory allocated after pruning: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    total_params_print = print_model_parameters(model, "Total parameters after pruning")
    if tune_eval:
        training_steps = 10
        if iteration != 1:
            training_steps = iteration * 10

        print(f"CUDA memory allocated before group eval: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        eval = tune_model(model, processor, training_steps, eval_metric, reference_probs, reference_logits, reference_representation)
        model.to("cpu")
        del model, processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        model = None
        processor = None
        gc.collect()
        print(f"CUDA memory allocated after group eval: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

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
