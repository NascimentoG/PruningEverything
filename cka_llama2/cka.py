import torch
import torch.nn.functional as F
from datasets import load_dataset
import random
import gc


def get_wikitext2_samples(num_samples=128):
    """
    Get random samples from wikitext2 dataset.
    Randomly samples until we get the specified number of non-empty samples.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = []

    while len(texts) < num_samples:
        sample = random.choice(dataset["text"])
        if sample.strip(): 
            texts.append(sample)
    print(f"Number of non-empty samples: {len(texts)}")
    return texts


def feature_space_linear_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor
) -> torch.Tensor:
    """
    Compute linear CKA between two feature matrices (n×d), optionally debiased.
    """
    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)

    k_xy = features_x.t().mm(features_y)
    k_xx = features_x.t().mm(features_x)
    k_yy = features_y.t().mm(features_y)

    dot_xy = torch.norm(k_xy) ** 2   
    norm_x = torch.norm(k_xx)                  
    norm_y = torch.norm(k_yy)                  

    cka = 1-(dot_xy / (norm_x * norm_y))
    dissimilarity = cka.detach().cpu().item()
    return dissimilarity


def compute_model_representation(model, tokenizer, device, num_samples=128): 
    model.eval()
    texts = get_wikitext2_samples(num_samples)

    representations = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            representations.append(outputs.hidden_states[-1][:, -1, :])
            del inputs, outputs
            torch.cuda.empty_cache()

    all_reprs = torch.cat(representations, dim=0).to(torch.float32)

    del representations
    gc.collect()

    return all_reprs


def compute_cka_from_reprs(repr_ref, repr_pruned):
    """
    Computes the CKA between two representations.
    """
    print(f"Reference representation shape: {repr_ref.shape}")
    print(f"Pruned representation shape: {repr_pruned.shape}")

    if repr_ref.shape != repr_pruned.shape:
        min_seq = min(repr_ref.shape[1], repr_pruned.shape[1])

        repr_ref = repr_ref[:, :min_seq, :]
        repr_pruned = repr_pruned[:, :min_seq, :]

    cka_dissimilarity = feature_space_linear_cka(repr_ref, repr_pruned)
    del repr_pruned
    gc.collect()
    return cka_dissimilarity
