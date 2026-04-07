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

GPU ACCELERATED: distance matrices now run on CUDA/XPU instead of CPU.
50-100x faster on large token sets.

Usage:
    # Original 13-model consensus (from mega_consensus_map1.json)
    python embedding_cartographer.py

    # New 8-model expansion run
    python embedding_cartographer.py --models \
        XiaomiMiMo/MiMo-V2-Flash-Base \
        Qwen/Qwen3-Coder-Next-Base \
        MiniMaxAI/MiniMax-M2.1 \
        moonshotai/Kimi-K2.5 \
        arcee-ai/Trinity-Large-Base \
        google/gemma-4-31B \
        baidu/Qianfan-VL-70B

    # Merge new results into existing consensus map
    python embedding_cartographer.py --merge existing_map.json new_map.json

    # Custom model list
    python embedding_cartographer.py --models meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B

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
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

# Original 13-model set (used to build mega_consensus_map1.json)
ORIGINAL_MODELS = [
    "meta-llama/Llama-3.1-70B",
    "Qwen/Qwen2.5-72B",
    "zai-org/GLM-5",
    "google/gemma-2-27b",
    "mistral-community/Mixtral-8x22B-v0.1",
    "deepseek-ai/DeepSeek-V3",
    "01-ai/Yi-1.5-34B",
    "openai/gpt-oss-120b",
    "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
    "sarvamai/sarvam-30b",
    "tiiuae/falcon-40b",
    "bigscience/bloom",
    "CohereForAI/c4ai-command-r-plus",
]

# New expansion models — 2026 frontier
NEW_MODELS = [
    "XiaomiMiMo/MiMo-V2-Flash-Base",       # 309B MoE, 15B active, Qwen3 tokenizer
    "Qwen/Qwen3-Coder-Next-Base",           # 80B MoE, 3B active, 262k vocab
    "MiniMaxAI/MiniMax-M2.1",               # MiniMax MoE
    "moonshotai/Kimi-K2.5",                 # 1T MoE, Kimi tokenizer
    "arcee-ai/Trinity-Large-Base",          # Arcee 400B
    # "baidu/ERNIE-4.5-VL-424B-A47B-Base-Paddle",  # Paddle-based — skipped, non-standard
    "google/gemma-4-31B",                   # Gemma 4, standard safetensors
    "baidu/Qianfan-VL-70B",                 # Baidu 70B VL — may need trust_remote_code
]

DEFAULT_MODELS = NEW_MODELS

MAX_TOKENS = 8000
OUTPUT_PATH = "./consensus_coordinates.json"
PMI_PRIORITY_PATH = "./code_crystals.json"

# Embedding tensor key patterns — extended for new model families
EMB_KEYS = [
    'embed_tokens', 'wte', 'word_embed', 'token_embed',
    'embedding', 'embeddings', 'word_embeddings',
    'shared.weight', 'tok_embeddings',
    # Gemma 4 / new Google models
    'embedder.weight', 'embed.weight',
    # MiniMax / Kimi variants
    'token_embedding', 'input_embedding',
    # Baidu Qianfan
    'word_embedding', 'text_embed',
]


# ═══════════════════════════════════════════════════════════════
#  GPU DEVICE
# ═══════════════════════════════════════════════════════════════

def get_compute_device():
    """Return best available device for matrix ops."""
    try:
        if torch.xpu.is_available():
            print("  Compute device: XPU")
            return torch.device('xpu')
    except Exception:
        pass
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Compute device: CUDA ({name}, {mem:.0f}GB)")
        return torch.device('cuda')
    print("  Compute device: CPU (no GPU found)")
    return torch.device('cpu')


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

    try:
        files = list(list_repo_files(model_name))
    except Exception as e:
        print(f"    ERROR listing files: {e}")
        return None, None

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
            tname_lower = tensor_name.lower()
            if any(k in tname_lower for k in EMB_KEYS):
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
        # Some tokenizers store vocab at top level
        if not vocab and 'vocab' in tok_data:
            v = tok_data['vocab']
            if isinstance(v, dict):
                vocab = v
            elif isinstance(v, list):
                vocab = {s: i for i, s in enumerate(v)}
    elif tok_path.endswith('.model'):
        # SentencePiece model — try tokenizer.json first
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
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(tok_path)
                vocab = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
            except Exception as e:
                print(f"    ERROR loading SentencePiece vocab: {e}")
                return None, None, None

    if not vocab:
        print(f"    ERROR: empty vocab extracted from tokenizer")
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
        keys_available = list(f.keys())

        # If we know the key from the index, try it first
        if known_key and known_key in keys_available:
            tensor = f.get_tensor(known_key)
            if tensor.dim() == 2 and tensor.shape[0] > 1000:
                emb = tensor.float()
                emb_key = known_key

        # Otherwise scan the shard
        if emb is None:
            for key in keys_available:
                if any(k in key.lower() for k in EMB_KEYS):
                    tensor = f.get_tensor(key)
                    if tensor.dim() == 2 and tensor.shape[0] > 1000:
                        emb = tensor.float()
                        emb_key = key
                        break

        # Last resort: find largest 2D tensor — it's almost certainly embeddings
        if emb is None:
            print(f"    Scanning all tensors for embedding (keys tried: {len(keys_available)})")
            best_size = 0
            for key in keys_available:
                try:
                    shape = f.get_slice(key).get_shape()
                    if len(shape) == 2 and shape[0] > 1000 and shape[0] * shape[1] > best_size:
                        best_size = shape[0] * shape[1]
                        emb_key = key
                except Exception:
                    continue
            if emb_key:
                emb = f.get_tensor(emb_key).float()

    if emb is None:
        print(f"    ERROR: could not find embedding matrix in {shard_file}")
        print(f"    Available keys: {keys_available[:20]}")
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
        if not all(ord(c) < 128 for c in clean):
            return (90, 0, tok)
        if (any(c.isupper() for c in clean[1:]) and
                any(c.islower() for c in clean)):
            return (80, 0, tok)
        if len(clean) > 12:
            return (70, 0, tok)
        if clean.isalpha() and clean.islower() and 3 <= len(clean) <= 10:
            return (0, -len(clean), tok)
        if clean.isalpha() and clean[0].isupper() and clean[1:].islower() and 2 <= len(clean) <= 10:
            return (1, -len(clean), tok)
        if clean.lower() in CODE_KEYWORDS:
            return (2, -len(clean), tok)
        if clean.isalpha() and len(clean) >= 2:
            return (3, -len(clean), tok)
        if clean.isalpha() and len(clean) == 1:
            return (4, 0, tok)
        if clean.isdigit():
            return (5, 0, tok)
        if all(not c.isalnum() for c in clean) and len(clean) <= 4:
            return (6, -len(clean), tok)
        if any(c.isalnum() for c in clean) and len(clean) <= 8:
            return (7, -len(clean), tok)
        return (50, 0, tok)

    remaining.sort(key=token_importance)

    budget = max_tokens - len(selected)
    if budget > 0:
        selected.extend(remaining[:budget])

    return selected[:max_tokens]


# ═══════════════════════════════════════════════════════════════
#  GPU-ACCELERATED DISTANCE MATRIX
# ═══════════════════════════════════════════════════════════════

def pairwise_cosine_distances_gpu(emb_np, device, chunk_size=2000):
    """
    Compute pairwise cosine distance matrix on GPU.
    Falls back to CPU scipy if GPU fails or isn't available.

    50-100x faster than scipy on GPU for n=6000+ tokens.
    Uses chunked computation to avoid OOM on smaller GPUs.
    """
    if device.type == 'cpu':
        # CPU fallback — use scipy
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(emb_np, metric='cosine'))

    try:
        n = emb_np.shape[0]
        emb_t = torch.tensor(emb_np, dtype=torch.float32, device=device)

        # L2 normalize
        norms = emb_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
        emb_norm = emb_t / norms

        # Chunked cosine similarity → distance
        # Avoids materializing full n×n on GPU at once
        dist = torch.zeros(n, n, dtype=torch.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = emb_norm[start:end]                   # [chunk, dim]
            sims = torch.mm(chunk, emb_norm.T)            # [chunk, n]
            dist[start:end] = (1.0 - sims).cpu()

        # Symmetrize and zero diagonal (numeric noise)
        dist = (dist + dist.T) / 2
        dist.fill_diagonal_(0.0)

        return dist.numpy()

    except Exception as e:
        print(f"    GPU distance failed ({e}), falling back to CPU")
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(emb_np, metric='cosine'))


# ═══════════════════════════════════════════════════════════════
#  MULTI-MODEL PROCRUSTES
# ═══════════════════════════════════════════════════════════════

def multi_model_alignment(embeddings, labels, model_names, device):
    """
    Run Procrustes alignment across all model pairs.
    Returns consensus distances averaged across all pairs.
    """
    n = len(labels)
    n_models = len(embeddings)
    triu = np.triu_indices(n, k=1)

    print(f"\n{'=' * 72}")
    print(f"  MULTI-MODEL SEMANTIC PROCRUSTES")
    print(f"{'=' * 72}")
    print(f"\n  Models: {n_models}")
    for i, name in enumerate(model_names):
        print(f"    [{i+1}] {name} — embedding dim {embeddings[i].shape[1]}")
    print(f"  Shared tokens analyzed: {n}")
    print(f"  Distance computation: {device}")

    # Compute all pairwise distance matrices (GPU accelerated)
    print(f"\n  Computing distance matrices...")
    dist_matrices = []
    for i, emb in enumerate(embeddings):
        t0 = time.time()
        d = pairwise_cosine_distances_gpu(emb, device)
        dt = time.time() - t0
        dist_matrices.append(d)
        print(f"    {model_names[i].split('/')[-1]}: done ({dt:.1f}s)")

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

    # Nearest neighbor agreement
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

    # Consensus distance
    print(f"\n{'─' * 72}")
    print(f"  CONSENSUS EXTRACTION")
    print(f"{'─' * 72}")

    consensus_dist = np.mean(all_normalized, axis=0)

    variance = np.var(all_normalized, axis=0)
    mean_var = variance.mean()

    token_scores = np.zeros(n)
    threshold = np.percentile(variance, 25)
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
#  MERGE: combine two consensus maps
# ═══════════════════════════════════════════════════════════════

def merge_consensus_maps(path_a, path_b, output_path):
    """
    Merge two consensus map JSONs into one.
    Averages scores for tokens that appear in both.
    Adds tokens that appear in only one.
    Updates metadata to reflect combined model list.
    """
    print(f"\n  Merging consensus maps:")
    print(f"    A: {path_a}")
    print(f"    B: {path_b}")

    with open(path_a) as f:
        map_a = json.load(f)
    with open(path_b) as f:
        map_b = json.load(f)

    scores_a = map_a.get('scores', {})
    scores_b = map_b.get('scores', {})

    # Merge scores — average where both have it
    merged_scores = {}
    all_tokens = set(scores_a.keys()) | set(scores_b.keys())
    for tok in all_tokens:
        if tok in scores_a and tok in scores_b:
            merged_scores[tok] = (scores_a[tok] + scores_b[tok]) / 2
        elif tok in scores_a:
            merged_scores[tok] = scores_a[tok] * 0.8  # discount single-source
        else:
            merged_scores[tok] = scores_b[tok] * 0.8

    # Sort by score
    compounds_sorted = sorted(merged_scores.keys(),
                               key=lambda t: merged_scores[t], reverse=True)

    meta_a = map_a.get('metadata', {})
    meta_b = map_b.get('metadata', {})
    models_a = meta_a.get('models', [])
    models_b = meta_b.get('models', [])
    all_models = models_a + [m for m in models_b if m not in models_a]

    rhos_a = meta_a.get('pairwise_rhos', [])
    rhos_b = meta_b.get('pairwise_rhos', [])
    combined_rho = float(np.mean(rhos_a + rhos_b)) if (rhos_a or rhos_b) else 0.0

    merged = {
        'compounds': compounds_sorted,
        'metadata': {
            'source': 'embedding_cartographer_merged',
            'models': all_models,
            'n_models': len(all_models),
            'n_analyzed': len(merged_scores),
            'overall_rho': combined_rho,
            'pairwise_rhos': rhos_a + rhos_b,
            'n_compounds': len(compounds_sorted),
            'merged_from': [path_a, path_b],
        },
        'scores': merged_scores,
    }

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n  Merged map → {output_path}")
    print(f"  Models: {len(all_models)}")
    print(f"  Tokens: {len(merged_scores):,}")
    print(f"  Combined ρ: {combined_rho:.4f}")
    print(f"  Top 20 tokens:")
    for tok in compounds_sorted[:20]:
        print(f"    {merged_scores[tok]:.3f}  {repr(tok)}")


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
    parser.add_argument('--merge', nargs=2, metavar=('MAP_A', 'MAP_B'),
                        help='Merge two existing consensus JSON files')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Force CPU even if GPU available')
    args = parser.parse_args()

    # ── Merge mode ──
    if args.merge:
        merge_consensus_maps(args.merge[0], args.merge[1], args.output)
        return

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

    # ── Compute device ──
    device = torch.device('cpu') if args.no_gpu else get_compute_device()

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

    if len(shared) < 100:
        print(f"  WARNING: only {len(shared)} shared tokens — models may use different tokenizers")
        print(f"  Proceeding anyway with reduced token set")

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
        multi_model_alignment(sub_embeddings, selected, names, device)

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
        print(f"  Low shared vocab or very different tokenizers likely.")

    # ── Export ──
    export_consensus(selected, token_scores, all_normalized, names,
                     pair_rhos, args.output)

    print(f"\n  To merge with existing map:")
    print(f"    python embedding_cartographer.py --merge mega_consensus_map1.json {args.output} --output mega_consensus_map2.json")
    print(f"\n  To use in Token128:")
    print(f"    consensus_path = \"{args.output}\"")
    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
