"""
Embedding Cartographer — Steal the Map, Skip the Brain
========================================================
Jason / Independent Research 2026

Downloads ONLY the embedding table from any HuggingFace model,
no matter how large. A 70B model has a ~2GB embedding table
sitting in a single 4GB shard — we grab that shard, extract
the table, and throw away everything else.

Then runs Procrustes alignment across as many models as you want
to produce high-resolution consensus coordinates.

Usage:
    # Two-model alignment (like before, but with big models)
    python embedding_cartographer.py

    # Custom model list
    python embedding_cartographer.py --models meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B google/gemma-3-1b-pt

    # Go big — steal from 70B
    python embedding_cartographer.py --models meta-llama/Llama-3.1-70B Qwen/Qwen2.5-72B

Requirements:
    pip install huggingface-hub safetensors torch numpy scipy scikit-learn
"""

import torch
import numpy as np
import json
import gc
import os
import time
import sys
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-70B",
    "Qwen/Qwen2.5-72B",
]

# Smaller alternatives (faster download, lower resolution):
# DEFAULT_MODELS = [
#     "meta-llama/Llama-3.1-8B",
#     "Qwen/Qwen2.5-7B",
#     "google/gemma-3-1b-pt",
# ]

MAX_TOKENS = 8000
OUTPUT_PATH = "./consensus_coordinates.json"
PMI_PRIORITY_PATH = "./code_crystals.json"

EMB_KEYS = ['embed_tokens', 'wte', 'word_embed', 'token_embed',
            'embedding', 'embeddings', 'word_embeddings',
            'shared.weight', 'tok_embeddings']


# ═══════════════════════════════════════════════════════════════
#  SMART EMBEDDING LOADER — downloads only what's needed
# ═══════════════════════════════════════════════════════════════

def find_embedding_shard(model_name):
    """
    For sharded models: read the index file to find which shard
    contains the embedding table. Returns (shard_filename, tensor_key).
    For single-file models: returns (filename, None).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(model_name)

    # Check for sharded model index
    index_file = None
    for candidate in ['model.safetensors.index.json',
                      'model.safetensors.index',
                      'pytorch_model.bin.index.json']:
        if candidate in files:
            index_file = candidate
            break

    if index_file and index_file.endswith('.json'):
        # Sharded model — read the weight map
        idx_path = hf_hub_download(model_name, index_file)
        with open(idx_path) as f:
            index = json.load(f)
        weight_map = index.get('weight_map', {})

        # Find embedding tensor in weight map
        for tensor_name, shard_file in weight_map.items():
            if any(k in tensor_name.lower() for k in EMB_KEYS):
                return shard_file, tensor_name

        # Fallback: embedding is usually in the first shard
        first_shard = sorted(set(weight_map.values()))[0]
        return first_shard, None

    # Single-file model
    st_files = sorted([f for f in files if f.endswith('.safetensors')])
    if st_files:
        return st_files[0], None

    return None, None


def load_vocab_and_embeddings(model_name):
    """
    Smart loader: finds which shard has the embedding,
    downloads ONLY that shard + tokenizer. Works for any model size.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print(f"\n  Loading {model_name}...")
    t0 = time.time()

    # ── Tokenizer ──
    tok_path = None
    for tok_file in ['tokenizer.json', 'tokenizer.model']:
        try:
            tok_path = hf_hub_download(model_name, tok_file)
            break
        except Exception:
            continue

    if tok_path is None:
        print(f"    ERROR: no tokenizer found")
        return None, None, None

    vocab = {}
    if tok_path.endswith('.json'):
        with open(tok_path) as f:
            tok_data = json.load(f)
        vocab = tok_data.get('model', {}).get('vocab', {})
        if isinstance(vocab, list):
            vocab = {s: i for i, s in enumerate(vocab)}
        if not vocab:
            added = tok_data.get('added_tokens', [])
            vocab = {t['content']: t['id'] for t in added if 'content' in t}
    elif tok_path.endswith('.model'):
        # SentencePiece model — try to load vocab from tokenizer.json fallback
        try:
            tj_path = hf_hub_download(model_name, 'tokenizer.json')
            with open(tj_path) as f:
                tok_data = json.load(f)
            vocab = tok_data.get('model', {}).get('vocab', {})
            if isinstance(vocab, list):
                vocab = {s: i for i, s in enumerate(vocab)}
            if not vocab:
                added = tok_data.get('added_tokens', [])
                vocab = {t['content']: t['id'] for t in added if 'content' in t}
        except Exception:
            # Last resort: try to use sentencepiece directly
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(tok_path)
                vocab = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
            except Exception as e:
                print(f"    ERROR loading SentencePiece vocab: {e}")
                return None, None, None

    print(f"    Vocab: {len(vocab):,} tokens")

    # ── Find embedding shard ──
    print(f"    Locating embedding shard...")
    shard_file, known_key = find_embedding_shard(model_name)

    if shard_file is None:
        print(f"    ERROR: no safetensors files found")
        return None, None, None

    print(f"    Downloading: {shard_file}")
    try:
        local = hf_hub_download(model_name, shard_file)
    except Exception as e:
        print(f"    ERROR downloading {shard_file}: {e}")
        return None, None, None

    # ── Extract embedding ──
    emb = None
    emb_key = None
    with safe_open(local, framework='pt', device='cpu') as f:
        # If we know the key from the index, try it first
        if known_key and known_key in f.keys():
            tensor = f.get_tensor(known_key)
            if tensor.dim() == 2 and tensor.shape[0] > 1000:
                emb = tensor.float()
                emb_key = known_key

        # Otherwise scan the shard
        if emb is None:
            for key in f.keys():
                if any(k in key.lower() for k in EMB_KEYS):
                    tensor = f.get_tensor(key)
                    if tensor.dim() == 2 and tensor.shape[0] > 1000:
                        emb = tensor.float()
                        emb_key = key
                        break

    if emb is None:
        print(f"    ERROR: could not find embedding matrix in {shard_file}")
        return None, None, None

    dt = time.time() - t0
    size_mb = emb.numel() * 4 / 1024**2
    print(f"    Embedding: {emb_key} → {emb.shape} ({size_mb:.0f} MB)")
    print(f"    Done in {dt:.1f}s")

    return vocab, emb, model_name


# ═══════════════════════════════════════════════════════════════
#  SHARED VOCABULARY
# ═══════════════════════════════════════════════════════════════

def find_shared_vocab(vocabs):
    """Find tokens shared across ALL models."""
    if not vocabs:
        return {}

    shared_tokens = set(vocabs[0].keys())
    for v in vocabs[1:]:
        shared_tokens &= set(v.keys())

    # Filter special tokens
    shared_tokens = {t for t in shared_tokens
                     if not (t.startswith('<') and t.endswith('>'))}

    # Build mapping: token → tuple of IDs (one per model)
    shared = {}
    for tok in shared_tokens:
        ids = tuple(v[tok] for v in vocabs)
        shared[tok] = ids

    return shared


def select_tokens(shared, max_tokens, pmi_path=None):
    """Priority token selection: PMI first, then by importance."""
    selected = []
    pmi_hits = 0

    # Priority 1: PMI tokens
    if pmi_path and os.path.exists(pmi_path):
        with open(pmi_path) as f:
            pmi_data = json.load(f)
        if isinstance(pmi_data, dict):
            pmi_tokens = pmi_data.get('compounds', pmi_data.get('tokens', []))
            if not pmi_tokens:
                pmi_tokens = sorted(
                    [k for k in pmi_data.keys() if len(k) >= 2],
                    key=lambda k: pmi_data[k] if isinstance(pmi_data[k], (int, float)) else 0,
                    reverse=True)
        elif isinstance(pmi_data, list):
            pmi_tokens = pmi_data
        else:
            pmi_tokens = []

        pmi_set = set()
        for tok in pmi_tokens:
            for variant in [tok, tok.strip(),
                            '\u0120' + tok.strip(), '\u2581' + tok.strip()]:
                if variant in shared and variant not in pmi_set:
                    selected.append(variant)
                    pmi_set.add(variant)
                    break
        pmi_hits = len(selected)
        print(f"  PMI priority tokens matched: {pmi_hits} / {len(pmi_tokens)}")

    # Priority 2: importance ranking
    selected_set = set(selected)
    remaining = [t for t in shared.keys() if t not in selected_set]

    CODE_KEYWORDS = {
        'def', 'return', 'import', 'class', 'self', 'print',
        'if', 'else', 'while', 'for', 'in', 'not', 'and', 'or', 'is',
        'true', 'false', 'none', 'try', 'except', 'with', 'as', 'from',
        'int', 'str', 'float', 'list', 'dict', 'set', 'function', 'var',
        'const', 'let', 'async', 'null', 'void', 'new', 'type', 'string',
        'break', 'continue', 'elif', 'lambda', 'yield', 'raise', 'pass',
        'del', 'global', 'assert', 'finally', 'switch', 'case', 'enum',
        'struct', 'static', 'public', 'private', 'protected', 'abstract',
        'interface', 'extends', 'implements', 'override', 'virtual',
        'throw', 'catch', 'this', 'super', 'instanceof', 'typeof',
        'export', 'default', 'require', 'module', 'package',
    }

    def token_importance(tok):
        clean = tok.replace('\u0120', '').replace('\u2581', '').strip()
        if not clean:
            return (99, 0, tok)
        # Skip non-ASCII tokens entirely
        if not all(ord(c) < 128 for c in clean):
            return (90, 0, tok)
        # Skip camelCase/PascalCase code identifiers (UIAlertController etc)
        if (any(c.isupper() for c in clean[1:]) and
                any(c.islower() for c in clean)):
            return (80, 0, tok)
        # Skip tokens longer than 12 chars — these are compound garbage
        if len(clean) > 12:
            return (70, 0, tok)
        # Common English words: 3-10 chars, all lowercase alpha = gold
        if clean.isalpha() and clean.islower() and 3 <= len(clean) <= 10:
            return (0, -len(clean), tok)
        # Capitalized words (sentence starters): "The", "When", etc
        if clean.isalpha() and clean[0].isupper() and clean[1:].islower() and 2 <= len(clean) <= 10:
            return (1, -len(clean), tok)
        # Code keywords
        if clean.lower() in CODE_KEYWORDS:
            return (2, -len(clean), tok)
        # Short subwords (2-3 chars, alpha)
        if clean.isalpha() and len(clean) >= 2:
            return (3, -len(clean), tok)
        # Single letters
        if clean.isalpha() and len(clean) == 1:
            return (4, 0, tok)
        # Digits
        if clean.isdigit():
            return (5, 0, tok)
        # Punctuation and code syntax
        if all(not c.isalnum() for c in clean) and len(clean) <= 4:
            return (6, -len(clean), tok)
        # Short mixed alphanumeric
        if any(c.isalnum() for c in clean) and len(clean) <= 8:
            return (7, -len(clean), tok)
        # Everything else (long mixed, unicode, etc)
        return (50, 0, tok)

    remaining.sort(key=token_importance)

    budget = max_tokens - len(selected)
    if budget > 0:
        selected.extend(remaining[:budget])

    return selected[:max_tokens]


# ═══════════════════════════════════════════════════════════════
#  MULTI-MODEL PROCRUSTES
# ═══════════════════════════════════════════════════════════════

def pairwise_cosine_distances(emb):
    """Compute pairwise cosine distance matrix."""
    return squareform(pdist(emb, metric='cosine'))


def multi_model_alignment(embeddings, labels, model_names):
    """
    Run Procrustes alignment across all model pairs.
    Returns consensus distances averaged across all pairs.
    """
    n = len(labels)
    n_models = len(embeddings)
    triu = np.triu_indices(n, k=1)
    n_pairs = len(triu[0])

    print(f"\n{'=' * 72}")
    print(f"  MULTI-MODEL SEMANTIC PROCRUSTES")
    print(f"{'=' * 72}")
    print(f"\n  Models: {n_models}")
    for i, name in enumerate(model_names):
        print(f"    [{i+1}] {name} — embedding dim {embeddings[i].shape[1]}")
    print(f"  Shared tokens analyzed: {n}")

    # Compute all pairwise distance matrices
    print(f"\n  Computing distance matrices...")
    dist_matrices = []
    for i, emb in enumerate(embeddings):
        d = pairwise_cosine_distances(emb)
        dist_matrices.append(d)
        print(f"    {model_names[i].split('/')[-1]}: done")

    # Pairwise alignment scores
    print(f"\n{'─' * 72}")
    print(f"  PAIRWISE ALIGNMENT")
    print(f"{'─' * 72}")

    pair_rhos = []
    all_normalized = []

    for i in range(n_models):
        flat_i = dist_matrices[i][triu]
        norm_i = flat_i / (flat_i.max() + 1e-8)
        all_normalized.append(norm_i)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            rho, _ = spearmanr(all_normalized[i], all_normalized[j])
            r, _ = pearsonr(all_normalized[i], all_normalized[j])
            name_i = model_names[i].split('/')[-1]
            name_j = model_names[j].split('/')[-1]
            print(f"\n  {name_i} ↔ {name_j}")
            print(f"    Spearman ρ = {rho:.4f}  Pearson r = {r:.4f}")
            pair_rhos.append(rho)

    mean_rho = np.mean(pair_rhos)
    print(f"\n  Mean pairwise ρ = {mean_rho:.4f}")

    # Nearest neighbor agreement (all pairs)
    print(f"\n{'─' * 72}")
    print(f"  NEAREST NEIGHBOR CONSENSUS")
    print(f"{'─' * 72}")

    for k in [1, 5, 10, 20]:
        agreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                overlap = 0
                for t in range(n):
                    nn_i = set(np.argsort(dist_matrices[i][t])[1:k+1])
                    nn_j = set(np.argsort(dist_matrices[j][t])[1:k+1])
                    overlap += len(nn_i & nn_j) / k
                agreements.append(overlap / n * 100)
        mean_agree = np.mean(agreements)
        baseline = k / n * 100
        ratio = mean_agree / baseline if baseline > 0 else 0
        print(f"    Top-{k:<2d} overlap: {mean_agree:5.1f}%  "
              f"(random={baseline:.1f}%, {ratio:.0f}x)")

    # Consensus distance: average across all models
    print(f"\n{'─' * 72}")
    print(f"  CONSENSUS EXTRACTION")
    print(f"{'─' * 72}")

    consensus_dist = np.mean(all_normalized, axis=0)

    # Score each token by agreement across models
    # Low variance = all models agree on this token's relationships
    variance = np.var(all_normalized, axis=0)
    mean_var = variance.mean()

    # Per-token consensus score: fraction of pairs where this token's
    # distances agree across models (low disagreement)
    token_scores = np.zeros(n)
    threshold = np.percentile(variance, 25)  # top 25% agreement
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if variance[pair_idx] < threshold:
                token_scores[i] += 1
                token_scores[j] += 1
            pair_idx += 1
    token_scores /= (n - 1)

    print(f"  Mean cross-model variance: {mean_var:.6f}")
    print(f"  Tokens with high consensus (>0.3): "
          f"{(token_scores > 0.3).sum()}")

    return dist_matrices, all_normalized, token_scores, pair_rhos


# ═══════════════════════════════════════════════════════════════
#  CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def category_analysis(labels, dist_matrices, model_names):
    """Break down alignment by token category."""
    n = len(labels)

    categories = {}
    for i, tok in enumerate(labels):
        s = tok.replace('\u0120', '').replace('\u2581', '').strip()
        if not s:
            cat = 'whitespace'
        elif s.isdigit():
            cat = 'digit'
        elif len(s) == 1 and s.isalpha():
            cat = 'single_letter'
        elif len(s) == 1 and not s.isalnum():
            cat = 'punctuation'
        elif s.isalpha() and len(s) >= 4:
            cat = 'word'
        elif s.isalpha():
            cat = 'short_subword'
        else:
            cat = 'mixed'
        categories.setdefault(cat, []).append(i)

    print(f"\n{'─' * 72}")
    print(f"  CATEGORY BREAKDOWN")
    print(f"{'─' * 72}")
    print(f"\n  {'Category':<20s} {'n':>5s} {'mean ρ':>8s}")
    print(f"  {'─'*20} {'─'*5} {'─'*8}")

    for cat in sorted(categories.keys()):
        idx = np.array(categories[cat])
        if len(idx) < 4:
            continue

        rhos = []
        for i in range(len(dist_matrices)):
            for j in range(i + 1, len(dist_matrices)):
                sub_i = dist_matrices[i][np.ix_(idx, idx)]
                sub_j = dist_matrices[j][np.ix_(idx, idx)]
                ti = np.triu_indices(len(idx), k=1)
                if len(ti[0]) < 3:
                    continue
                r, _ = spearmanr(sub_i[ti], sub_j[ti])
                rhos.append(r)

        if rhos:
            mean_rho = np.mean(rhos)
            print(f"  {cat:<20s} {len(idx):>5d} {mean_rho:>8.4f}")


# ═══════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════

def export_consensus(labels, token_scores, all_normalized, model_names,
                     pair_rhos, output_path):
    """Export consensus coordinates."""
    n = len(labels)

    compounds = []
    for i in np.argsort(token_scores)[::-1]:
        tok = labels[i]
        clean = tok.replace('\u0120', ' ').replace('\u2581', ' ')
        if len(clean.strip()) < 2:
            continue
        if token_scores[i] < 0.1:
            continue
        compounds.append({'token': clean, 'score': float(token_scores[i])})

    overall_rho = float(np.mean(pair_rhos))

    output = {
        'compounds': [c['token'] for c in compounds],
        'metadata': {
            'source': 'embedding_cartographer',
            'models': model_names,
            'n_models': len(model_names),
            'n_analyzed': n,
            'overall_rho': overall_rho,
            'pairwise_rhos': [float(r) for r in pair_rhos],
            'n_compounds': len(compounds),
        },
        'scores': {c['token']: c['score'] for c in compounds},
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 72}")
    print(f"  EXPORTED: {output_path}")
    print(f"{'=' * 72}")
    print(f"  Models: {len(model_names)}")
    print(f"  Overall ρ: {overall_rho:.4f}")
    print(f"  Compounds: {len(compounds)}")
    print(f"  Top 30 by consensus score:")
    for c in compounds[:30]:
        print(f"    {c['score']:.3f}  {repr(c['token'])}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Embedding Cartographer — steal geometry from any model')
    parser.add_argument('--models', nargs='+', default=None,
                        help='HuggingFace model names (2+)')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS,
                        help=f'Max tokens to analyze (default {MAX_TOKENS})')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help=f'Output path (default {OUTPUT_PATH})')
    parser.add_argument('--pmi', type=str, default=PMI_PRIORITY_PATH,
                        help='PMI file for priority token selection')
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS

    if len(models) < 2:
        print("Need at least 2 models for alignment")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("  Embedding Cartographer")
    print("  Steal the map. Skip the brain.")
    print("=" * 72)
    print(f"\n  Models to align: {len(models)}")
    for m in models:
        print(f"    • {m}")

    # ── Load all models ──
    vocabs = []
    embeddings_raw = []
    names = []

    for model_name in models:
        vocab, emb, name = load_vocab_and_embeddings(model_name)
        if vocab is None:
            print(f"\n  SKIPPING {model_name} (failed to load)")
            continue
        vocabs.append(vocab)
        embeddings_raw.append(emb)
        names.append(name)

    if len(vocabs) < 2:
        print("\n  FATAL: Need at least 2 successfully loaded models")
        sys.exit(1)

    # ── Find shared vocabulary ──
    shared = find_shared_vocab(vocabs)
    print(f"\n  Shared vocabulary across {len(vocabs)} models: {len(shared):,} tokens")

    # ── Select tokens ──
    selected = select_tokens(shared, args.max_tokens, args.pmi)
    print(f"  Selected {len(selected)} tokens for analysis")

    # ── Extract aligned embeddings ──
    sub_embeddings = []
    for i, emb in enumerate(embeddings_raw):
        ids = np.array([shared[t][i] for t in selected])
        sub = emb[ids].numpy()
        sub_embeddings.append(sub)
        del emb
    del embeddings_raw
    gc.collect()

    # ── Run alignment ──
    dist_matrices, all_normalized, token_scores, pair_rhos = \
        multi_model_alignment(sub_embeddings, selected, names)

    # ── Category analysis ──
    category_analysis(selected, dist_matrices, names)

    # ── Verdict ──
    mean_rho = np.mean(pair_rhos)
    print(f"\n{'=' * 72}")
    print(f"  VERDICT")
    print(f"{'=' * 72}")
    print(f"\n  Mean Spearman ρ = {mean_rho:.4f}")
    if mean_rho > 0.5:
        print(f"  ρ > 0.5 = STRONG topological agreement.")
        print(f"  The distances are real. They are properties of language.")
    elif mean_rho > 0.3:
        print(f"  ρ 0.3-0.5 = MODERATE agreement.")
    else:
        print(f"  ρ < 0.3 = WEAK agreement.")

    # ── Export ──
    export_consensus(selected, token_scores, all_normalized, names,
                     pair_rhos, args.output)

    print(f"\n  To use in Token128:")
    print(f"    PMI_VOCAB_PATH = \"{args.output}\"")
    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
