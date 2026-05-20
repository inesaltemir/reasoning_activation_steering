import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import html
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoModelForCausalLM, AutoTokenizer

# dot product h_i * v_{reasoning}, where h_i is the residual stream activation at token $i$

# ==========================================
# 1. Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen3-8B"
VECTORS_FILE = "holistic_reasoning_vectors.pt"
OUTPUT_HTML_FILE = "activation_visualization.html"

TARGET_LAYER = 24  
LAYER_KEY = f"blocks.{TARGET_LAYER}.hook_out" 

def main():
    # ==========================================
    # 2. Load Model, Tokenizer, and Vectors
    # ==========================================
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16
    )

    print(f"Loading reasoning vectors from {VECTORS_FILE}...")
    vectors_dict = torch.load(VECTORS_FILE)

    # Extract ALL THREE vectors, squeeze to 1D, and move to device
    layer_data = vectors_dict["layers"][LAYER_KEY]
    
    vec_raw = layer_data["holistic_reasoning_direction"].squeeze().to(model.device).to(torch.float16)
    vec_clean = layer_data["holistic_reasoning_direction_cleaned"].squeeze().to(model.device).to(torch.float16)
    vec_perfect = layer_data["mean_fully_correct_samples"].squeeze().to(model.device).to(torch.float16)
    vec_perfect_cleaned = layer_data["corr_reasoning_steering_vector_cleaned"].squeeze().to(model.device).to(torch.float16)
    

    # ==========================================
    # 3. Define the Sample Text
    # ==========================================
    sample_text_simple = """Problem: Solve the equation 3x + 5 = 14.
    Steps:
    1) Subtract 5 from both sides: 3x = 9.
    2) Divide by 3: x = 3.
    The final answer is correct."""

    sample_text_gpqa = """Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they be clearly resolved? 10^-4 ev """
    sample_text_processbench = """Find the smallest number $n$ such that there exist polynomials $f_{1}, f_{2}, \\ldots, f_{n}$ with rational coefficients satisfying\n\n$$\nx^{2}+7=f_{1}(x)^{2}+f_{2}(x)^{2}+\\cdots+f_{n}(x)^{2} .\n$$. To find the smallest number \\( n \\), we start by considering the given equation: \\( x^2 + 7 = f_1(x)^2 + f_2(x)^2 + \\cdots + f_n(x)^2 \\). Notice that \\( x^2 + 7 \\) is always greater than or equal to 7 for any real value of \\( x \\). We also observe that the right-hand side of the equation is a sum of squares of polynomials with rational coefficients. This means each term on the right-hand side must be non-negative. Now consider the case when \\( x = 0 \\). Plugging in \\( x = 0 \\) into the equation, we get: \\( 0^2 + 7 = f_1(0)^2 + f_2(0)^2 + \\cdots + f_n(0)^2 \\). This simplifies to \\( 7 = f_1(0)^2 + f_2(0)^2 + \\cdots + f_n(0)^2 \\). Since 7 is a prime number, the only way to express it as a sum of squares of integers is \\( 1^2 + 2^2 \\) (since 7 cannot be expressed as a single square). However, we are looking for polynomials with rational coefficients. In this case, we can express 7 as a sum of squares of two rational numbers: \\( (\\sqrt{7})^2 = (\\sqrt{7})^2 + 0^2 \\), but this does not satisfy our requirement of having rational coefficients for the polynomials. We need at least two non-zero terms. After trying different combinations, we find that 7 can be expressed as a sum of squares of three rational numbers: \\( (2)^2 + (1)^2 + (\\sqrt{2})^2 \\neq 7 \\), however, \\( (2)^2 + (1)^2 + (1)^2 + (1)^2 = 7 \\). However, we used integer coefficients here. Now we try with a simple expression like \\( x^2 \\) and constants. We see: \\( x^2 + 7 = (x^2 + 2)^2 + (-1)^2 + (-1)^2 + (-1)^2 \\). Comparing both sides, we see we have satisfied our initial condition. So \\( n = 4 \\) satisfies the equation. Therefore, the smallest number \\( n \\) is \\(\\boxed{4}\\)."""
 
    sample_text = """The box was light in my hands, addressed to Amanda Peters in neat printed labels. I twirled it once, then set it down on my coffee table with a satisfied pat. My thesis defense had gone perfectly yesterday, and I was still floating from it. Everything felt easy, manageable, fun. Even dealing with misdelivered mail seemed like a delightful little quest rather than an annoyance. I opened my laptop, fingers flying across the keyboard as I searched for Amanda on social media. There—Amanda Peters, moved to Denver six months ago. I sent her a message, adding a friendly emoji, offering to forward the package or hold it if she had someone picking up her stuff. Helping people felt natural when you felt this good about life."""

    # ==========================================
    # 4. Forward Pass with Hugging Face
    # ==========================================
    print("Running forward pass...")
    inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Shape: (seq_len, d_model)
    hidden_states = outputs.hidden_states[TARGET_LAYER][0] 

    # ==========================================
    # 5. Project onto ALL THREE Reasoning Vectors
    # ==========================================
    print("Projecting hidden states onto reasoning vectors...")
    act_raw = torch.matmul(hidden_states, vec_raw).cpu().float().numpy()
    act_clean = torch.matmul(hidden_states, vec_clean).cpu().float().numpy()
    act_perfect = torch.matmul(hidden_states, vec_perfect).cpu().float().numpy()
    act_perfect_cleaned = torch.matmul(hidden_states, vec_perfect_cleaned).cpu().float().numpy()
    

    # ==========================================
    # 6. Scale Activations
    # ==========================================
    # we dividE by the 99th percentile and not the absolute maximum
    # there is still 1% of the data that is larger than our scale factor
    # So clip them to interval [-1,1]
    # scale by 99th percentile instead of absolute maximum value to mitigate effect of rogue, massic meaningless spike in residual stream (random punctuation, BOS, whitespace)

    def scale_activations(activations):
        percentile_99 = np.percentile(np.abs(activations), 99)
        scale_factor = percentile_99 if percentile_99 > 1e-5 else 1.0 
        scaled = activations / scale_factor
        return np.clip(scaled, -1.0, 1.0)

    scaled_act_raw = scale_activations(act_raw)
    scaled_act_clean = scale_activations(act_clean)
    scaled_act_perfect = scale_activations(act_perfect)
    scaled_act_perfect_cleaned = scale_activations(act_perfect_cleaned)


    # ==========================================
    # 7. Generate Side-by-Side HTML Visualization
    # ==========================================
    print("Generating HTML visualization...")
    
    cmap = plt.get_cmap('coolwarm')
    gradient_colors = [mcolors.to_hex(cmap(i / 100.0)) for i in range(101)]
    css_gradient = f"linear-gradient(to right, {', '.join(gradient_colors)})"
    
    legend_html = f"""
    <div style="width: 400px; margin: 0 auto 30px auto; font-family: sans-serif;">
        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: bold; margin-bottom: 5px; color: #333;">
            <span>-1.0 (Opposite)</span>
            <span>0.0 (Neutral)</span>
            <span>+1.0 (Aligned)</span>
        </div>
        <div style="height: 20px; width: 100%; border-radius: 4px; background: {css_gradient}; border: 1px solid #aaa; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);"></div>
        <div style="font-size: 11px; color: #666; margin-top: 5px; text-align: center;">
            Color scale mapped to 99th percentile of vector activation values.
        </div>
    </div>
    """

    def generate_spans(tokens, activations):
        html_str = ""
        for token, act in zip(tokens, activations):
            norm_act = (act + 1) / 2
            rgba = cmap(norm_act)
            hex_color = mcolors.to_hex(rgba)
            
            if isinstance(token, bytes):
                token = token.decode('utf-8', errors='replace')
                
            clean_token = token.replace('Ġ', ' ').replace('Ċ', '\n')
            clean_token = html.escape(clean_token)
            
            if '\n' in clean_token:
                html_str += "<br>"
                clean_token = clean_token.replace('\n', '')
                if not clean_token: continue

            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.5 else "black"
            
            html_str += f"<span style='background-color: {hex_color}; color: {text_color}; padding: 0.5px 1px; margin: 0 1px; border-radius: 3px;' title='Activation: {act:.3f}'>{clean_token}</span>"
            
        return html_str

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    # Generate spans for all three
    spans_raw = generate_spans(tokens, scaled_act_raw)
    spans_clean = generate_spans(tokens, scaled_act_clean)
    spans_perfect = generate_spans(tokens, scaled_act_perfect)
    spans_perfect_cleaned = generate_spans(tokens, scaled_act_perfect_cleaned)

    # Construct the full HTML document with THREE columns
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Reasoning Vectors Visualization</title>
    <style>
        body {{ background-color: #f5f5f5; padding: 30px; font-family: sans-serif; }}
        .col {{ flex: 1; background: #fff; padding: 15px; border-radius: 5px; border: 1px solid #ddd; color: black; line-height: 1.6; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h3 {{ margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; font-size: 16px; }}
    </style>
</head>
<body>
    <h2 style="text-align: center; margin-bottom: 15px;">Reasoning Activation Vectors (Layer {TARGET_LAYER})</h2>
    
    {legend_html}
    
    <div style="display: flex; gap: 20px; font-family: monospace; font-size: 13px;">
        
        <div class="col">
            <h3>Mean correct samples</h3>
            {spans_perfect}
        </div>

        <div class="col">
            <h3>Mean correct samples fineweb de-confounded</h3>
            {spans_perfect_cleaned}
        </div>

        <div class="col">
            <h3>Raw Holistic Reasoning<br><span style="font-size:11px; font-weight:normal; color:#666;">(Perfect - Flawed)</span></h3>
            {spans_raw}
        </div>
        
        <div class="col">
            <h3>De-confounded Holistic<br><span style="font-size:11px; font-weight:normal; color:#666;">(FineWeb PCA removed)</span></h3>
            {spans_clean}
        </div>

        
    </div>
</body>
</html>
"""

    # ==========================================
    # 8. Save to File
    # ==========================================
    with open(OUTPUT_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"\n✅ Visualization successfully saved to '{OUTPUT_HTML_FILE}'.")

    # ==========================================
    # 9. Explicit GPU Cleanup
    # ==========================================
    print("Releasing model from GPU memory...")
    del model
    del vec_raw, vec_clean, vec_perfect, vec_perfect_cleaned
    del hidden_states
    torch.cuda.empty_cache()
    print("GPU memory released.")

if __name__ == "__main__":
    main()