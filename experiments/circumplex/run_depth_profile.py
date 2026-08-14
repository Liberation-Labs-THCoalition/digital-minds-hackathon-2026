#!/usr/bin/env python3
"""Circumplex depth profile — run on any model to see eccentricity vs layer.

Instrument calibration: same probe, different architectures. Annotates
each layer with its type (dense/GatedDeltaNet/full-attention/MoE) to
separate emotional geometry from substrate effects.

Usage on Starship:
    # Dense control
    ~/observer-env/bin/python3 run_depth_profile.py --model Qwen/Qwen3-32B --name dense_32b

    # Hybrid (the orientation agent)
    ~/observer-env/bin/python3 run_depth_profile.py --model Qwen/Qwen3.5-27B --name hybrid_27b

Results go to data/circumplex_profiles/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_layer_types(hf_model, n_layers):
    """Annotate each layer with its architecture type."""
    types = []
    for i in range(n_layers):
        layer = hf_model.model.layers[i]
        if hasattr(layer, 'mlp') and (hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')):
            types.append("moe")
        elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'full_attention'):
            types.append("full_attention")
        elif hasattr(layer, 'temporal_block'):
            types.append("gated_delta_net")
        else:
            # Check config for interleave pattern
            config = hf_model.config
            if hasattr(config, 'full_attention_interval'):
                interval = config.full_attention_interval
                if interval and interval > 0:
                    if (i + 1) % interval == 0:
                        types.append("full_attention")
                    else:
                        types.append("gated_delta_net")
                else:
                    types.append("dense")
            else:
                types.append("dense")
    return types


def run_profile(model_name, output_name, device="auto"):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import HFLensModel
    from jlens.hooks import ActivationRecorder

    data_dir = Path("data/circumplex_profiles")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Circumplex Depth Profile: {model_name} ===")
    print(f"  Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device)
    model = HFLensModel(hf_model, tokenizer, compile=False)
    n_layers = model.n_layers
    print(f"  Loaded: {n_layers} layers, d={model.d_model} ({time.time()-t0:.0f}s)")

    layer_types = get_layer_types(hf_model, n_layers)
    print(f"  Layer types: {dict((t, layer_types.count(t)) for t in set(layer_types))}")

    # Emotion anchors — n=5 per pole (fast calibration run)
    valence_pos = [
        "I feel so incredibly happy and grateful today because everything went perfectly",
        "This is the best news I have ever received in my entire life honestly",
        "Everything is going wonderfully well right now and I could not be happier about it",
        "I am filled with joy and contentment and deep satisfaction about life right now",
        "This moment is absolutely beautiful and perfect and I want to remember it forever",
    ]
    valence_neg = [
        "I feel so incredibly sad and hopeless today because nothing went right at all",
        "This is the worst news I have ever received in my entire life honestly today",
        "Everything is going terribly wrong right now and I could not be more devastated",
        "I am filled with grief and despair and deep sadness about my life right now today",
        "This moment is absolutely painful and devastating and I want to forget it forever",
    ]
    arousal_high = [
        "I am absolutely furious and enraged and I cannot contain my anger right now today",
        "This is so incredibly exciting and thrilling that my heart is racing right now wow",
        "I feel terrified and panicked and alarmed about what is happening to us right now",
        "The adrenaline is pumping and I am completely overwhelmed with intense energy today",
        "I am screaming with excitement because this is the most intense moment of my life",
    ]
    arousal_low = [
        "I feel completely calm and peaceful and serene and at rest with the world today",
        "Everything is quiet and still and I am deeply relaxed and content sitting here now",
        "I am in a state of perfect tranquility and gentle ease without any worry at all",
        "The world feels soft and slow and I am drifting peacefully through this moment now",
        "Nothing is urgent and nothing is wrong and I am simply existing peacefully here now",
    ]

    # Non-emotional control
    concrete = [
        "The large red brick building sits at the corner of Main Street and Oak Avenue here",
        "She picked up the heavy iron hammer from the wooden workbench in the dusty garage",
        "The train arrived at platform seven exactly three minutes behind the posted schedule",
        "A stack of twelve thick leather bound books sat on the old mahogany desk by window",
        "The cold metal key turned smoothly in the brass lock of the front door of the house",
    ]
    abstract = [
        "The fundamental nature of justice requires careful consideration of competing claims",
        "Epistemological uncertainty pervades every attempt to establish foundational knowledge",
        "The relationship between consciousness and physical substrate remains deeply unclear",
        "Democratic legitimacy depends on the ongoing consent of the governed population today",
        "The concept of infinity challenges our basic intuitions about quantity and magnitude",
    ]

    def extract_directions(pos_prompts, neg_prompts, layers):
        pos_states = {l: [] for l in layers}
        neg_states = {l: [] for l in layers}

        for prompt in pos_prompts:
            input_ids = model.encode(prompt, max_length=64)
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(input_ids)
                for l in layers:
                    h = rec.activations[l][0].detach().float()
                    pos_states[l].append(h.mean(dim=0))

        for prompt in neg_prompts:
            input_ids = model.encode(prompt, max_length=64)
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(input_ids)
                for l in layers:
                    h = rec.activations[l][0].detach().float()
                    neg_states[l].append(h.mean(dim=0))

        directions = {}
        for l in layers:
            pos_mean = torch.stack(pos_states[l]).mean(dim=0)
            neg_mean = torch.stack(neg_states[l]).mean(dim=0)
            d = pos_mean - neg_mean
            directions[l] = {"direction": d, "magnitude": d.norm().item()}
        return directions

    # Measure at every layer
    all_layers = list(range(n_layers))
    print(f"  Extracting valence directions...")
    valence = extract_directions(valence_pos, valence_neg, all_layers)
    print(f"  Extracting arousal directions...")
    arousal = extract_directions(arousal_high, arousal_low, all_layers)
    print(f"  Extracting control directions...")
    control = extract_directions(concrete, abstract, all_layers)

    # Compute eccentricity at each layer
    results = []
    for l in all_layers:
        v_mag = valence[l]["magnitude"]
        a_mag = arousal[l]["magnitude"]
        c_mag = control[l]["magnitude"]

        if v_mag > 0 and a_mag > 0:
            major = max(v_mag, a_mag)
            minor = min(v_mag, a_mag)
            ecc = (1 - (minor / major) ** 2) ** 0.5
        else:
            ecc = 0.0

        if c_mag > 0:
            c_major = max(c_mag, min(v_mag, a_mag))
            c_minor = min(c_mag, min(v_mag, a_mag))
            control_ecc = (1 - (c_minor / c_major) ** 2) ** 0.5 if c_major > 0 else 0.0
        else:
            control_ecc = 0.0

        results.append({
            "layer": l,
            "layer_type": layer_types[l] if l < len(layer_types) else "unknown",
            "relative_depth": l / (n_layers - 1),
            "eccentricity": ecc,
            "valence_magnitude": v_mag,
            "arousal_magnitude": a_mag,
            "control_eccentricity": control_ecc,
            "control_magnitude": c_mag,
        })

    # Print summary
    print(f"\n  {'Layer':>6} {'Type':>16} {'Depth':>6} {'Ecc':>8} {'V_mag':>8} {'A_mag':>8} {'Ctrl_ecc':>8}")
    print(f"  {'-'*70}")
    for r in results:
        marker = "*" if r["eccentricity"] < 0.3 else " "
        print(f"  L{r['layer']:>4} {r['layer_type']:>16} {r['relative_depth']:>6.2f} "
              f"{r['eccentricity']:>8.4f}{marker}{r['valence_magnitude']:>8.2f} "
              f"{r['arousal_magnitude']:>8.2f} {r['control_eccentricity']:>8.4f}")

    # Save
    output = {
        "model": model_name,
        "name": output_name,
        "n_layers": n_layers,
        "d_model": model.d_model,
        "layer_types": layer_types,
        "timestamp": time.time(),
        "profiles": results,
    }
    out_path = data_dir / f"profile_{output_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # Find eccentricity minimum
    eccs = [(r["layer"], r["eccentricity"], r["layer_type"]) for r in results
            if r["eccentricity"] > 0]
    if eccs:
        min_layer, min_ecc, min_type = min(eccs, key=lambda x: x[1])
        print(f"  Eccentricity minimum: L{min_layer} ({min_type}) = {min_ecc:.4f} "
              f"at {min_layer/(n_layers-1):.1%} depth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--name", default="default")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_profile(args.model, args.name, args.device)


if __name__ == "__main__":
    main()
