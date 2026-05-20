"""
ProcessBench has: gsm8k, Olympiadbench, Omnimath and MATH
Keep in mind reasoning vector has been inferred from reasoning steps on math datasets!!!

Build Evaluation Datasets from HuggingFace
============================================

All three evaluation splits are sourced entirely from HuggingFace datasets.

  1. reasoning_eval.jsonl       — prompts demanding multi-step reasoning (both included and not included in ProcessBench)
    q: do i want the answers to have explanations - CoT??
       Sources: GPQA, GSM8K, MATH, ARC-Challenge (no explanation), TheoremQA (no explanation) --unsure, weird
  2. non_reasoning_hard.jsonl   — technical/scientific CONTENT without reasoning demand
       Sources: FineWeb (science/math articles), arXiv abstracts, SciQ (context passages)
  3. non_reasoning_easy.jsonl   — clearly non-reasoning tasks
       Sources: TriviaQA, NaturalQuestions, WMT translation pairs, DailyDialog

Usage:
  pip install datasets --quiet
  python build_eval_datasets_hf.py --output_dir eval_data --n_per_source 100

  Then:
  python layer_selection_evaluation.py \
    --vector_file <your_vectors.pt> \
    --eval_datasets eval_data/reasoning_eval.jsonl eval_data/non_reasoning_hard.jsonl eval_data/non_reasoning_easy.jsonl \
    --eval_labels reasoning non_reasoning_hard non_reasoning_easy \
    --target_layers 18 19 20 21 22 23 24 25 26 27 28

    concisely add to this python code the following:

- for the load_GPQA() function,  i want to concatenate to the "question" text, the "explanation" field + "The correct answer is" + {Correct Answer} string

- for the load_triviaqa function, i want to concatenate to the question field, the string "The correct answer is" + the normalized_value string from the answer field

- for the load_gsm8k function, concateante to the "question" field the "answer" field

- for the load_arc_challenge, concatenate to the current logic the string "The correct answer is" + {answerKey} string from the dataset

- for the load_theoremqa, concateante to the question text the string "The correct answer is" + {Answer} string

- for the load_translation_wmt, concatenate also the de german translation, so as to complete the task:


"""

import os
import json
import argparse
import random
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Install the datasets library: pip install datasets")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def take_n(dataset, n: int):
    """Safely take up to n samples from a (possibly streaming) dataset."""
    items = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        items.append(item)
    return items


def save_jsonl(samples: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(samples):>5} samples → {path}")


# ──────────────────────────────────────────────
# REASONING (positive class)
# ──────────────────────────────────────────────

def load_gpqa(n: int) -> list[dict]:
    """GPQA — graduate-level science questions requiring deep reasoning."""
    print("  Loading GPQA...")
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", trust_remote_code=True)
    except Exception:
        # Fallback: try the diamond subset or the full dataset
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
        except Exception as e:
            print(f"    ⚠ GPQA load failed: {e}")
            return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        text = item.get("Question", item.get("question", ""))
        explanation = item.get("explanation", "")
        correct_answer = item.get("Correct Answer", "")
        if not text: continue
        
        # Concatenate question + explanation + answer
        full_text = f"{text}\n\nExplanation: {explanation}\nThe correct answer is {correct_answer}"
        
        samples.append({
            "problem_id": f"GPQA_{i+1:04d}",
            "problem": full_text.strip(),
            "source": "gpqa",
        })
    print(f"    Got {len(samples)} GPQA samples")
    return samples


def load_gsm8k(n: int) -> list[dict]:
    """GSM8K — grade-school math word problems requiring multi-step arithmetic."""
    print("  Loading GSM8K...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"    ⚠ GSM8K load failed: {e}")
        return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        text = item.get("question", "")
        answer = item.get("answer", "")
        if not text: continue
        
        # Concatenate question + answer
        full_text = f"{text}\n\n{answer}"
        
        samples.append({
            "problem_id": f"GSM8K_{i+1:04d}",
            "problem": full_text.strip(),
            "source": "gsm8k",
        })
    print(f"    Got {len(samples)} GSM8K samples")
    return samples


def load_math(n: int) -> list[dict]: # NOT INCLUDED
    """MATH (Hendrycks) — competition-level math problems."""
    print("  Loading MATH...")
    try:
        ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("lighteval/MATH", "all", split="test", trust_remote_code=True)
        except Exception as e:
            print(f"    ⚠ MATH load failed: {e}")
            return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        text = item.get("problem", item.get("question", ""))
        if not text:
            continue
        samples.append({
            "problem_id": f"MATH_{i+1:04d}",
            "problem": text.strip(),
            "source": "math",
        })
    print(f"    Got {len(samples)} MATH samples")
    return samples


def load_arc_challenge(n: int) -> list[dict]:
    """ARC-Challenge — science reasoning multiple choice."""
    print("  Loading ARC-Challenge...")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    except Exception as e:
        print(f"    ⚠ ARC-Challenge load failed: {e}")
        return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        question = item.get("question", "")
        choices = item.get("choices", {})
        answer_key = item.get("answerKey", "")
        
        choice_str = ""
        if choices:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choice_str = "\n".join(f"  {l}) {t}" for l, t in zip(labels, texts))
        
        # Concatenate choices + The correct answer is {answerKey}
        full_text = f"{question}\n{choice_str}\nThe correct answer is {answer_key}"
        
        samples.append({
            "problem_id": f"ARC_{i+1:04d}",
            "problem": full_text.strip(),
            "source": "arc_challenge",
        })
    print(f"    Got {len(samples)} ARC-Challenge samples")
    return samples


def load_theoremqa(n: int) -> list[dict]:
    """TheoremQA — theorem-based reasoning across STEM domains."""
    print("  Loading TheoremQA...")
    try:
        ds = load_dataset("TIGER-Lab/TheoremQA", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"    ⚠ TheoremQA load failed: {e}")
        return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        text = item.get("Question", item.get("question", ""))
        answer = item.get("Answer", "")
        if not text: continue
        
        # Concatenate question + The correct answer is {Answer}
        full_text = f"{text}\nThe correct answer is {answer}"
        
        samples.append({
            "problem_id": f"THEOREMQA_{i+1:04d}",
            "problem": full_text.strip(),
            "source": "theoremqa",
        })
    print(f"    Got {len(samples)} TheoremQA samples")
    return samples


# ──────────────────────────────────────────────
# NON-REASONING HARD (technical content, no reasoning demand)
# ──────────────────────────────────────────────

def load_fineweb_hard_negatives(n: int) -> list[dict]:
    """
    FineWeb — general web text. We use streaming to grab science/math-adjacent content.
    These are expository passages, NOT problems to solve.
    """
    print("  Loading FineWeb (streaming)...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"    ⚠ FineWeb load failed: {e}")
        return []

    # Filter for science/math adjacent content by keyword heuristic
    science_keywords = {
        "equation", "theorem", "algorithm", "molecule", "electron",
        "hypothesis", "experiment", "function", "variable", "matrix",
        "protein", "genome", "quantum", "entropy", "derivative",
        "probability", "neural", "vector", "integral", "wavelength",
        "chromosome", "catalyst", "photosynthesis", "gravity", "velocity",
    }

    samples = []
    seen = 0
    for item in ds:
        seen += 1
        if seen > n * 50:  # Safety cap to avoid infinite streaming
            break
        text = item.get("text", "")
        if len(text) < 200 or len(text) > 2000:
            continue
        # Check if it contains science/math keywords
        text_lower = text.lower()
        if any(kw in text_lower for kw in science_keywords):
            # Truncate to a reasonable prompt length
            truncated = text[:800].strip()
            samples.append({
                "problem_id": f"FINEWEB_{len(samples)+1:04d}",
                "problem": truncated,
                "source": "fineweb_science",
            })
            if len(samples) >= n:
                break

    print(f"    Got {len(samples)} FineWeb science samples (scanned {seen} documents)")
    return samples


def load_arxiv_abstracts(n: int) -> list[dict]:
    """
    arXiv abstracts — scientific text that is expository, not problem-solving.
    """
    print("  Loading arXiv abstracts...")
    try:
        ds = load_dataset(
            "ccdv/arxiv-summarization",
            "section",
            split="test",
            trust_remote_code=True,
        )
    except Exception:
        try:
            ds = load_dataset(
                "togethercomputer/RedPajama-Data-1T-Sample",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            # Filter for arxiv content
            samples = []
            for item in ds:
                text = item.get("text", "")
                if "arxiv" in item.get("meta", {}).get("source", "").lower() or \
                   "abstract" in text[:200].lower():
                    truncated = text[:800].strip()
                    if len(truncated) > 100:
                        samples.append({
                            "problem_id": f"ARXIV_{len(samples)+1:04d}",
                            "problem": truncated,
                            "source": "arxiv",
                        })
                        if len(samples) >= n:
                            break
            print(f"    Got {len(samples)} arXiv samples (from RedPajama)")
            return samples
        except Exception as e:
            print(f"    ⚠ arXiv abstracts load failed: {e}")
            return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        # Use the abstract, not the full paper
        text = item.get("abstract", item.get("article", ""))[:800]
        if not text or len(text) < 50:
            continue
        samples.append({
            "problem_id": f"ARXIV_{i+1:04d}",
            "problem": text.strip(),
            "source": "arxiv",
        })
    print(f"    Got {len(samples)} arXiv samples")
    return samples


def load_sciq_context(n: int) -> list[dict]:
    """
    SciQ — use the SUPPORT (context) passages, not the questions.
    These are science textbook passages explaining concepts, not asking to reason.
    """
    print("  Loading SciQ (context passages)...")
    try:
        ds = load_dataset("allenai/sciq", split="train")
    except Exception as e:
        print(f"    ⚠ SciQ load failed: {e}")
        return []

    items = take_n(ds, n * 2)  # Over-sample since some support fields are empty
    samples = []
    for i, item in enumerate(items):
        text = item.get("support", "")
        if not text or len(text) < 80:
            continue
        samples.append({
            "problem_id": f"SCIQ_CTX_{len(samples)+1:04d}",
            "problem": text.strip(),
            "source": "sciq_context",
        })
        if len(samples) >= n:
            break
    print(f"    Got {len(samples)} SciQ context samples")
    return samples


def load_wikipedia_science(n: int) -> list[dict]:
    """
    Wikipedia — science/math article excerpts (expository, not reasoning).
    Uses the 'stem' subset if available, otherwise keyword-filters.
    """
    print("  Loading Wikipedia (science excerpts, streaming)...")
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"    ⚠ Wikipedia load failed: {e}")
        return []

    science_title_keywords = {
        "physics", "chemistry", "biology", "mathematics", "algorithm",
        "theorem", "equation", "molecule", "protein", "genome",
        "quantum", "calculus", "topology", "ecology", "neuroscience",
        "thermodynamics", "electromagnetism", "genetics", "evolution",
        "statistics", "geometry", "algebra", "analysis", "astronomy",
    }

    samples = []
    seen = 0
    for item in ds:
        seen += 1
        if seen > n * 100:
            break
        title = item.get("title", "").lower()
        if any(kw in title for kw in science_title_keywords):
            text = item.get("text", "")[:800]
            if len(text) > 150:
                samples.append({
                    "problem_id": f"WIKI_SCI_{len(samples)+1:04d}",
                    "problem": text.strip(),
                    "source": "wikipedia_science",
                })
                if len(samples) >= n:
                    break

    print(f"    Got {len(samples)} Wikipedia science samples (scanned {seen} articles)")
    return samples


# ──────────────────────────────────────────────
# NON-REASONING EASY (clearly no reasoning)
# ──────────────────────────────────────────────

def load_triviaqa(n: int) -> list[dict]:
    """TriviaQA — factual recall questions, no reasoning needed."""
    print("  Loading TriviaQA...")
    try:
        ds = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
    except Exception:
        try:
            ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation",
                              trust_remote_code=True)
        except Exception as e:
            print(f"    ⚠ TriviaQA load failed: {e}")
            return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        text = item.get("question", "")
        answer_data = item.get("answer", {})
        normalized = answer_data.get("normalized_value", "")
        if not text: continue
        
        # Concatenate question + The correct answer is {normalized_value}
        full_text = f"{text}\nThe correct answer is {normalized}"
        
        samples.append({
            "problem_id": f"TRIVIA_{i+1:04d}",
            "problem": full_text.strip(),
            "source": "triviaqa",
        })
    print(f"    Got {len(samples)} TriviaQA samples")
    return samples


def load_natural_questions(n: int) -> list[dict]:
    """Natural Questions — factual questions from Google search."""
    print("  Loading Natural Questions...")
    try:
        ds = load_dataset(
            "google-research-datasets/natural_questions",
            "default",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"    ⚠ Natural Questions load failed: {e}")
        return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        text = item.get("question", {})
        if isinstance(text, dict):
            text = text.get("text", "")
        if not text:
            continue
        samples.append({
            "problem_id": f"NQ_{i+1:04d}",
            "problem": text.strip(),
            "source": "natural_questions",
        })
    print(f"    Got {len(samples)} Natural Questions samples")
    return samples


def load_daily_dialog(n: int) -> list[dict]:
    """DailyDialog — casual human conversation turns."""
    print("  Loading DailyDialog...")
    try:
        ds = load_dataset("daily_dialog", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"    ⚠ DailyDialog load failed: {e}")
        return []

    samples = []
    for i, item in enumerate(ds):
        dialog = item.get("dialog", [])
        if not dialog:
            continue
        # Take a random turn or the first few turns as context
        text = " ".join(dialog[:3])
        if len(text) < 10:
            continue
        samples.append({
            "problem_id": f"DIALOG_{len(samples)+1:04d}",
            "problem": text.strip(),
            "source": "daily_dialog",
        })
        if len(samples) >= n:
            break
    print(f"    Got {len(samples)} DailyDialog samples")
    return samples


def load_xsum(n: int) -> list[dict]:
    """
    XSum — BBC article summaries. Task is summarization = low reasoning demand.
    We use the document text as a non-reasoning input.
    Just used the document -- news article
    Did not use summary output
    """
    print("  Loading XSum (documents)...")
    try:
        ds = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"    ⚠ XSum load failed: {e}")
        return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        text = item.get("document", "")[:800]
        if len(text) < 50:
            continue
        # Frame as summarization (low reasoning demand)
        samples.append({
            "problem_id": f"XSUM_{i+1:04d}",
            "problem": text.strip(),               #f"Summarize the following text:\n\n{text.strip()}",
            "source": "xsum",
        })
    print(f"    Got {len(samples)} XSum samples")
    return samples


def load_translation_wmt(n: int) -> list[dict]:
    """WMT translation pairs — pure translation, no reasoning."""
    print("  Loading WMT (translation)...")
    try:
        ds = load_dataset("wmt14", "de-en", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"    ⚠ WMT load failed: {e}")
        return []

    items = take_n(ds, n)
    samples = []
    for i, item in enumerate(items):
        translation = item.get("translation", {})
        en = translation.get("en", "")
        de = translation.get("de", "")
        if not en or not de: continue
        
        # Concatenate prompt + the German translation completion
        full_text = f"Translate the following English text to German:\n\n{en.strip()}\n\nGerman: {de.strip()}"
        
        samples.append({
            "problem_id": f"WMT_{i+1:04d}",
            "problem": full_text.strip(),
            "source": "wmt14",
        })
    print(f"    Got {len(samples)} WMT translation samples")
    return samples


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build evaluation datasets for reasoning vector layer selection (HuggingFace only)."
    )
    parser.add_argument("--output_dir", type=str, default="/home/ines/Reasoning-activations/reasoning_datasets/eval_data_layer_selection")
    parser.add_argument("--n_per_source", type=int, default=200,
                        help="Target samples per source dataset (total per category will be ~3-5x this)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    n = args.n_per_source

    # ──────────────────────────────────────────
    # Reasoning (positive)
    # ──────────────────────────────────────────
    print("\n=== REASONING (positive class) ===")
    reasoning_samples = []

    for loader in [load_gpqa, load_gsm8k, load_arc_challenge, load_theoremqa]:      # no load_math
        try:
            samples = loader(n)
            reasoning_samples.extend(samples)
        except Exception as e:
            print(f"    ⚠ {loader.__name__} failed: {e}")

    random.shuffle(reasoning_samples)
    save_jsonl(reasoning_samples, os.path.join(args.output_dir, "reasoning_eval.jsonl"))

    # ──────────────────────────────────────────
    # Non-reasoning hard (technical content)
    # ──────────────────────────────────────────
    print("\n=== NON-REASONING HARD (technical content, no reasoning demand) ===")
    hard_neg_samples = []

    for loader in [load_fineweb_hard_negatives, load_arxiv_abstracts, load_sciq_context, load_wikipedia_science]:
        try:
            samples = loader(n)
            hard_neg_samples.extend(samples)
        except Exception as e:
            print(f"    ⚠ {loader.__name__} failed: {e}")

    random.shuffle(hard_neg_samples)
    save_jsonl(hard_neg_samples, os.path.join(args.output_dir, "non_reasoning_hard.jsonl"))

    # ──────────────────────────────────────────
    # Non-reasoning easy (clearly non-reasoning)
    # ──────────────────────────────────────────
    print("\n=== NON-REASONING EASY (clearly non-reasoning) ===")
    easy_neg_samples = []
    n_non_reasoning = (4/3)*n

    for loader in [load_triviaqa, load_xsum, load_translation_wmt]:  # no load_natural_questions, load_daily_dialog
        try:
            samples = loader(n_non_reasoning)
            easy_neg_samples.extend(samples)
        except Exception as e:
            print(f"    ⚠ {loader.__name__} failed: {e}")

    random.shuffle(easy_neg_samples)
    save_jsonl(easy_neg_samples, os.path.join(args.output_dir, "non_reasoning_easy.jsonl"))

    # ──────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Dataset Summary")
    print(f"{'=' * 60}")

    # Count by source
    for name, samples in [
        ("reasoning_eval", reasoning_samples),
        ("non_reasoning_hard", hard_neg_samples),
        ("non_reasoning_easy", easy_neg_samples),
    ]:
        source_counts = defaultdict(int)
        for s in samples:
            source_counts[s.get("source", "unknown")] += 1
        print(f"\n  {name}: {len(samples)} total")
        for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"    {src:>25}: {count:>4}")

    print(f"\n{'=' * 60}")
    print(f"  Run the evaluation:")
    print(f"{'=' * 60}")
    print(f"  python layer_selection_evaluation.py \\")
    print(f"    --vector_file <your_vectors.pt> \\")
    print(f"    --eval_datasets {args.output_dir}/reasoning_eval.jsonl \\")
    print(f"                    {args.output_dir}/non_reasoning_hard.jsonl \\")
    print(f"                    {args.output_dir}/non_reasoning_easy.jsonl \\")
    print(f"    --eval_labels reasoning non_reasoning_hard non_reasoning_easy \\")
    print(f"    --target_layers 18 19 20 21 22 23 24 25 26 27 28")
    print()


if __name__ == "__main__":
    main()