import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import argparse
import logging
from dataclasses import dataclass
from typing import List, Union
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge

# norms computed from FineWeb activations

# For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 (the default setting in generation_config.json). 
# DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions. 


# ==========================================
# Configuration Classes
# ==========================================
@dataclass
class NormalizedSteeringConfig:
    """Configuration for a steering intervention scaled to baseline norms."""
    layer: int
    vector: torch.Tensor
    alpha: float
    baseline_avg_norm: float
    
    def __post_init__(self):
        # 1. L2 Normalize the raw reasoning vector
        unit_vector = self.vector / torch.norm(self.vector)
        
        # 2. Scale it by the average baseline norm for this specific layer
        scaled_vector = unit_vector * self.baseline_avg_norm
        
        # 3. Apply the final steering strength multiplier (alpha)
        self.applied_vector = scaled_vector * self.alpha

# ==========================================
# Experiment Engine
# ==========================================
class ReasoningSteeringExperiment:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logging.info(f"Booting {model_name} via TransformerBridge on {self.device}...")
        self.model = TransformerBridge.boot_transformers(model_name)
        self.model.enable_compatibility_mode(disable_warnings=True, no_processing=True)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    def _get_steering_hook(self, config: NormalizedSteeringConfig):
        """Generates a forward hook function to inject the normalized steering vector."""
        # Move vector to the correct device once to prevent overhead during generation
        steering_vec = config.applied_vector.to(self.device)
        
        def hook_fn(activations, hook):
            # activations shape: [batch, seq_len, d_model]
            # In generate mode, this applies to the cached token being generated
            activations = activations + steering_vec
            return activations
            
        return hook_fn
    
    def generate_baseline(self, 
                          prompt: str, 
                          max_new_tokens: int = 512,
                          temperature: float = 0.6) -> str:
        """
        Generates text using the original model without any steering hooks.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Return the full context
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_with_steering(self, 
                               prompt: str, 
                               configs: Union[NormalizedSteeringConfig, List[NormalizedSteeringConfig]], 
                               max_new_tokens: int = 512,
                               temperature: float = 0.6) -> str:
        """
        Generates text while applying the steering configurations.
        """
        if isinstance(configs, NormalizedSteeringConfig):
            configs = [configs]
            
        # Prepare hooks
        fwd_hooks = []
        for config in configs:
            # Target the residual stream post-layer
            hook_name = f"blocks.{config.layer}.hook_out"
            # f"blocks.{config.layer}.hook_resid_post"
            hook_fn = self._get_steering_hook(config)
            fwd_hooks.append((hook_name, hook_fn))
            
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        # Generate with hooks context manager
        with self.model.hooks(fwd_hooks=fwd_hooks):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0),
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Return only the newly generated tokens
        #generated_ids = output_ids[0][prompt_len:]
        #return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Return the full context
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ==========================================
# Main CLI & Execution
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run normalized activation steering on a dataset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name.")
    parser.add_argument("--layer", type=int, default=21, help="Layer to steer.")
    parser.add_argument("--alpha", type=float, default=2, help="Steering strength multiplier.")
    parser.add_argument("--vector_file", type=str, default="/home/ines/Reasoning-activations/reasoning_vectors/Qwen3-8B/processbench/reasoning_vectors_cleaned_fineweb.pt", help="Path to cleaned reasoning vectors .pt file.")
    parser.add_argument("--norms_file", type=str, default="/home/ines/Reasoning-activations/baseline_vectors/Qwen3-8B/fineweb_activations_20000_avg_norms.pt", help="Path to baseline norms .pt file.")
    parser.add_argument("--input_jsonl", type=str, default="/home/ines/Reasoning-activations/src/steering_exp/test_input_v1.jsonl", help="Path to input dataset (e.g., AIME problems).")
    parser.add_argument("--output_jsonl", type=str, default="/home/ines/Reasoning-activations/causal_steering_exp/output_1.jsonl" , help="Path to save generated responses.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens to generate.")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # 1. Load Data
    logging.info(f"Loading reasoning vectors from {args.vector_file}...")
    vector_data = torch.load(args.vector_file, map_location="cpu")
    
    logging.info(f"Loading baseline norms from {args.norms_file}...")
    norms_data = torch.load(args.norms_file, map_location="cpu")

    # 2. Extract specific vector and norm for the target layer
    layer_key = f"blocks.{args.layer}.hook_out"
    if layer_key not in vector_data["layers"]:
        raise ValueError(f"Layer {layer_key} not found in vector file.")
    
    raw_vector = vector_data["layers"][layer_key]["reasoning_direction_sample_cleaned"] # reasoning_direction_sample
    avg_norm_for_layer = norms_data["avg_norms"][args.layer]
    
    logging.info(f"Layer {args.layer} | Baseline Norm: {avg_norm_for_layer:.4f} | Alpha: {args.alpha}")

    # 3. Create the Normalized Steering Config
    steer_config = NormalizedSteeringConfig(
        layer=args.layer,
        vector=raw_vector,
        alpha=args.alpha,
        baseline_avg_norm=avg_norm_for_layer
    )
    logging.info(f"Final applied vector magnitude: {torch.norm(steer_config.applied_vector).item():.4f}")

    # 4. Initialize Experiment
    experiment = ReasoningSteeringExperiment(model_name=args.model)

    # 5. Run Evaluation Loop
    logging.info(f"Reading prompts from {args.input_jsonl}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    
    with open(args.input_jsonl, "r", encoding="utf-8") as infile, \
         open(args.output_jsonl, "w", encoding="utf-8") as outfile:
        
        lines = infile.readlines()
        for line in tqdm(lines, desc=f"Generating with alpha={args.alpha}"):
            sample = json.loads(line.strip())
            
            # Adjust the prompt formatting to match your dataset structure
            # check or no whether using chat template from model's HF 
            #prompt = f"Problem:\n{sample['problem']}\n\nReasoning:\n"

            # Construct the prompt using the model's chat template
            messages = [
                # Optional: Add a system prompt if your model prefers one
                # {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": sample['problem']}
            ]
            
            # apply_chat_template automatically formats the messages using the Jinja template
            prompt = experiment.tokenizer.apply_chat_template(
                messages,
                tokenize=False,              # Return a string, not token IDs
                add_generation_prompt=True   # Appends the <|im_start|>assistant tag at the end
            )
            
            # --- 1. Generate Baseline (Unsteered) Response ---
            original_response = experiment.generate_baseline(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=0.0
            )

            # --- 2. Generate Steered Response ---
            steered_response = experiment.generate_with_steering(
                prompt=prompt,
                configs=steer_config,
                max_new_tokens=args.max_tokens,
                temperature=0.0 
            )
            
            # Save both generations
            sample["original_generation"] = original_response
            sample["steered_generation"] = steered_response
            sample["steered_layer"] = args.layer
            sample["steered_alpha"] = args.alpha
            
            outfile.write(json.dumps(sample) + "\n")
            outfile.flush()
            
    logging.info(f"Experiment complete. Results saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()