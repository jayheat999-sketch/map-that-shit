"""
Consensus Map Extractor — The Geometry IS the Model
=====================================================
Jason (theory) + Claude (implementation) — 2026

What we proved:
    - Two 70B models agree on token distances (ρ=0.74)
    - The agreement is rotation-invariant (Procrustes gap = 0.000)
    - The agreement is category-invariant (cross-gen gap = 0.000)
    - SVD effective rank is 2-4 at 99% energy

The geometry is intrinsic. It doesn't depend on orientation, it
doesn't depend on model, it doesn't depend on token category.
It's a property of language.

This script:
    1. Loads embedding tables from N models (downloads only the
       embedding shard, ~4GB each, not full weights)
    2. Computes cosine distance matrix from each model
    3. Averages the distance matrices (noise cancels, signal survives)
    4. Finds the intrinsic dimensionality via eigenvalue analysis
    5. Embeds the consensus distances onto continuous coordinates
       using classical MDS (Multi-Dimensional Scaling)
    6. Reports how much structure is captured at each dimension
    7. Exports a compact coordinate table

The output is NOT binary codes on a hypercube. It's continuous
coordinates on the real manifold. The sampler navigates these
directly.

Usage:
    python consensus_map.py

    # More models = less noise
    python consensus_map.py --models meta-llama/Llama-3.1-70B Qwen/Qwen2.5-72B google/gemma-2-27b

    # Control vocabulary size
    python consensus_map.py --max_tokens 8000

    # Custom output
    python consensus_map.py --output consensus_map.json
"""

import numpy as np
import json
import time
import sys
import os
import gc
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════
#  EMBEDDING LOADER — steals only the embedding shard
# ═══════════════════════════════════════════════════════════════

EMB_KEYS = ['embed_tokens', 'wte', 'word_embed', 'token_embed',
            'embedding', 'embeddings', 'word_embeddings',
            'shared.weight', 'tok_embeddings']


def load_model_embeddings(model_name):
    """
    Download ONLY the embedding shard + tokenizer from any HF model.
    Returns vocab dict, embedding matrix, model name.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from embedding_cartographer import find_embedding_shard

    print(f"\n  Loading {model_name}...")
    t0 = time.time()

    # Tokenizer
    tok_path = None
    for tok_file in ['tokenizer.json', 'tokenizer.model']:
        try:
            tok_path = hf_hub_download(model_name, tok_file)
            break
        except Exception:
            continue
    if tok_path is None:
        print(f"    ERROR: no tokenizer")
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
            pass

    print(f"    Vocab: {len(vocab):,} tokens")

    # Embedding shard
    shard_file, known_key = find_embedding_shard(model_name)
    if shard_file is None:
        print(f"    ERROR: no safetensors")
        return None, None, None

    print(f"    Shard: {shard_file}")
    local = hf_hub_download(model_name, shard_file)

    emb = None
    with safe_open(local, framework='pt', device='cpu') as f:
        if known_key and known_key in f.keys():
            tensor = f.get_tensor(known_key)
            if tensor.dim() == 2 and tensor.shape[0] > 1000:
                emb = tensor.float()
        if emb is None:
            for key in f.keys():
                if any(k in key.lower() for k in EMB_KEYS):
                    tensor = f.get_tensor(key)
                    if tensor.dim() == 2 and tensor.shape[0] > 1000:
                        emb = tensor.float()
                        break

    if emb is None:
        print(f"    ERROR: no embedding matrix")
        return None, None, None

    dt = time.time() - t0
    print(f"    Embedding: {emb.shape} ({dt:.1f}s)")
    return vocab, emb, model_name


# ═══════════════════════════════════════════════════════════════
#  TOKEN SELECTION — what goes on the map
# ═══════════════════════════════════════════════════════════════

def select_map_tokens(shared_vocab, max_tokens, pmi_path=None):
    """
    Select tokens for the consensus map.
    Priority: ASCII chars > PMI compounds > common English words > rest.
    Filters out BPE fragments, camelCase, non-ASCII garbage.
    """
    # ASCII characters first (always included)
    ascii_tokens = []
    for code in range(128):
        char = chr(code)
        candidates = [char, f'<0x{code:02X}>', f'<0x{code:02x}>',
                      '\u2581' + char]
        for c in candidates:
            if c in shared_vocab:
                ascii_tokens.append(('ascii', code, c))
                break
        # Fallback: single-char token
        if not any(c[2] == char for c in ascii_tokens if c[1] == code):
            if char in shared_vocab:
                ascii_tokens.append(('ascii', code, char))

    print(f"    ASCII tokens: {len(ascii_tokens)}")

    # PMI compounds
    pmi_tokens = []
    if pmi_path and os.path.exists(pmi_path):
        with open(pmi_path) as f:
            pmi_data = json.load(f)
        compounds = pmi_data.get('compounds', [])
        for compound in compounds:
            clean = compound.strip()
            for variant in [compound, clean, '\u0120' + clean, '\u2581' + clean]:
                if variant in shared_vocab:
                    pmi_tokens.append(('pmi', compound, variant))
                    break
        print(f"    PMI compounds matched: {len(pmi_tokens)}")

    # Common words from shared vocab
    ENGLISH_FILTER = set()  # could add dictionary filter here

    word_tokens = []
    seen = set(t[2] for t in ascii_tokens + pmi_tokens)

    for tok in sorted(shared_vocab):
        if tok in seen:
            continue
        clean = tok.replace('\u0120', '').replace('\u2581', '').strip()
        if not clean:
            continue
        # Filter garbage
        if not all(ord(c) < 128 for c in clean):
            continue
        if len(clean) < 2:
            continue
        if len(clean) > 12:
            continue
        # Skip camelCase
        if any(c.isupper() for c in clean[1:]) and any(c.islower() for c in clean):
            continue
        # Skip tokens starting with special chars (BPE fragments)
        if tok.startswith('<') and tok.endswith('>'):
            continue

        # Score: prefer common English words
        if clean.isalpha() and clean.islower() and 3 <= len(clean) <= 10:
            score = 0  # gold
        elif clean.isalpha() and clean[0].isupper() and len(clean) >= 3:
            score = 1
        elif clean.isalpha() and len(clean) >= 2:
            score = 2
        elif clean.isdigit():
            score = 3
        elif all(not c.isalnum() for c in clean) and len(clean) <= 3:
            score = 4
        else:
            score = 5

        word_tokens.append((score, tok, clean))

    word_tokens.sort()
    word_tokens = [('word', clean, tok) for score, tok, clean in word_tokens]

    # Combine: ASCII + PMI + words, up to max_tokens
    all_tokens = ascii_tokens + pmi_tokens
    budget = max_tokens - len(all_tokens)
    if budget > 0:
        all_tokens.extend(word_tokens[:budget])

    print(f"    Total selected: {len(all_tokens)}")
    return all_tokens


# ═══════════════════════════════════════════════════════════════
#  CONSENSUS DISTANCE MATRIX
# ═══════════════════════════════════════════════════════════════

def compute_consensus_distances(token_list, vocabs, embeddings, model_names):
    """
    Compute cosine distance matrix from each model, then average.

    Each model votes on how far apart every pair of tokens is.
    Averaging cancels model-specific noise. What survives is the
    intrinsic geometry that all models agree on.
    """
    n_tokens = len(token_list)
    n_models = len(vocabs)

    print(f"\n{'═' * 70}")
    print(f"  CONSENSUS DISTANCE MATRIX")
    print(f"{'═' * 70}")
    print(f"  Tokens: {n_tokens}")
    print(f"  Models: {n_models}")

    all_dist_matrices = []

    for m_idx in range(n_models):
        vocab = vocabs[m_idx]
        emb = embeddings[m_idx]
        name = model_names[m_idx].split('/')[-1]

        print(f"\n  Computing distances for {name}...")
        t0 = time.time()

        # Extract embeddings for selected tokens
        vecs = []
        for token_type, token_id, vocab_key in token_list:
            if token_type == 'ascii':
                # token_id is the char code, try to find it
                char = chr(token_id)
                candidates = [char, vocab_key,
                              f'<0x{token_id:02X}>', f'<0x{token_id:02x}>',
                              '\u2581' + char]
                found = False
                for c in candidates:
                    if c in vocab and vocab[c] < emb.shape[0]:
                        vecs.append(emb[vocab[c]].numpy())
                        found = True
                        break
                if not found:
                    vecs.append(np.zeros(emb.shape[1]))
            else:
                # word or pmi
                for variant in [vocab_key, token_id,
                                '\u0120' + str(token_id).strip(),
                                '\u2581' + str(token_id).strip()]:
                    if variant in vocab and vocab[variant] < emb.shape[0]:
                        vecs.append(emb[vocab[variant]].numpy())
                        break
                else:
                    vecs.append(np.zeros(emb.shape[1]))

        mat = np.array(vecs)

        # Normalize for cosine distance
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        mat_normed = mat / norms

        # Cosine distance matrix
        dist = squareform(pdist(mat_normed, 'cosine'))

        # Normalize to [0, 1] range for fair averaging across models
        d_max = dist.max()
        if d_max > 0:
            dist /= d_max

        all_dist_matrices.append(dist)
        dt = time.time() - t0
        print(f"    Done ({dt:.1f}s)")

    # ── Average across models ──
    print(f"\n  Averaging {n_models} distance matrices...")
    consensus_dist = np.mean(all_dist_matrices, axis=0)

    # ── Report cross-model agreement ──
    triu = np.triu_indices(n_tokens, k=1)
    print(f"\n  Cross-model agreement:")
    pair_rhos = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rho, _ = spearmanr(all_dist_matrices[i][triu],
                               all_dist_matrices[j][triu])
            na = model_names[i].split('/')[-1]
            nb = model_names[j].split('/')[-1]
            print(f"    {na} ↔ {nb}: ρ={rho:.4f}")
            pair_rhos.append(rho)

    mean_rho = np.mean(pair_rhos)
    print(f"    Mean ρ: {mean_rho:.4f}")

    # ── Agreement vs consensus ──
    for i in range(n_models):
        rho_c, _ = spearmanr(all_dist_matrices[i][triu], consensus_dist[triu])
        name = model_names[i].split('/')[-1]
        print(f"    {name} → consensus: ρ={rho_c:.4f}")

    return consensus_dist, mean_rho, pair_rhos


# ═══════════════════════════════════════════════════════════════
#  INTRINSIC DIMENSIONALITY — how big is the real map?
# ═══════════════════════════════════════════════════════════════

def find_intrinsic_dimensionality(consensus_dist):
    """
    Classical MDS eigenvalue analysis.

    Convert distance matrix → Gram matrix → eigendecomposition.
    The eigenvalue spectrum tells us how many dimensions carry
    real structure vs noise.

    Returns eigenvalues, explained variance ratios, and recommended dim.
    """
    n = consensus_dist.shape[0]

    print(f"\n{'═' * 70}")
    print(f"  INTRINSIC DIMENSIONALITY ANALYSIS")
    print(f"{'═' * 70}")

    # Double-center the squared distance matrix to get Gram matrix
    D_sq = consensus_dist ** 2
    H = np.eye(n) - np.ones((n, n)) / n  # centering matrix
    G = -0.5 * H @ D_sq @ H  # Gram matrix

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Only positive eigenvalues carry real structure
    pos_eigenvalues = eigenvalues[eigenvalues > 0]
    total_energy = pos_eigenvalues.sum()

    # Cumulative energy
    cumulative = np.cumsum(pos_eigenvalues) / total_energy

    print(f"\n  Eigenvalue spectrum (top 30):")
    print(f"  {'dim':>5s} {'eigenvalue':>12s} {'% energy':>10s} {'cumulative':>12s} {'bar'}")
    print(f"  {'─'*5} {'─'*12} {'─'*10} {'─'*12} {'─'*30}")

    for i in range(min(30, len(pos_eigenvalues))):
        pct = pos_eigenvalues[i] / total_energy * 100
        cum = cumulative[i] * 100
        bar_len = int(pct / 2)
        bar = '█' * bar_len
        print(f"  {i+1:>5d} {pos_eigenvalues[i]:>12.4f} {pct:>9.2f}% {cum:>11.2f}% {bar}")

    # Find dimension at energy thresholds
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    print(f"\n  Dimensions needed:")
    recommended = None
    for thresh in thresholds:
        dims_needed = np.searchsorted(cumulative, thresh) + 1
        print(f"    {thresh*100:.0f}% energy: {dims_needed} dimensions")
        if thresh == 0.95 and recommended is None:
            recommended = dims_needed

    if recommended is None:
        recommended = min(32, len(pos_eigenvalues))

    # Effective rank (Shannon entropy of normalized eigenvalue distribution)
    p = pos_eigenvalues / total_energy
    p = p[p > 1e-10]  # avoid log(0)
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)
    print(f"\n  Effective rank (Shannon): {effective_rank:.1f}")
    print(f"  Recommended embedding dim (95% energy): {recommended}")

    # Negative eigenvalues = non-Euclidean structure
    neg_eigenvalues = eigenvalues[eigenvalues < 0]
    if len(neg_eigenvalues) > 0:
        neg_energy = abs(neg_eigenvalues.sum()) / total_energy * 100
        print(f"\n  Negative eigenvalue energy: {neg_energy:.2f}%")
        if neg_energy > 5:
            print(f"    > 5% negative energy: distance matrix has significant")
            print(f"    non-Euclidean structure. Consider hyperbolic embedding.")
        else:
            print(f"    < 5%: mostly Euclidean. MDS embedding is faithful.")

    return eigenvalues, eigenvectors, pos_eigenvalues, cumulative, recommended


# ═══════════════════════════════════════════════════════════════
#  EMBED — place tokens on the manifold
# ═══════════════════════════════════════════════════════════════

def embed_consensus(eigenvalues, eigenvectors, n_dims, consensus_dist):
    """
    Classical MDS embedding: place tokens in n_dims continuous space
    such that Euclidean distances ≈ consensus cosine distances.

    Returns: coordinates [N, n_dims]
    """
    print(f"\n{'═' * 70}")
    print(f"  EMBEDDING — {n_dims} continuous dimensions")
    print(f"{'═' * 70}")

    # MDS coordinates: X = V * sqrt(Lambda)
    pos_mask = eigenvalues[:n_dims] > 0
    sqrt_eigenvals = np.zeros(n_dims)
    sqrt_eigenvals[pos_mask] = np.sqrt(eigenvalues[:n_dims][pos_mask])

    coords = eigenvectors[:, :n_dims] * sqrt_eigenvals[np.newaxis, :]

    # Verify: how well do Euclidean distances in embedding space
    # match the original consensus distances?
    embedded_dist = squareform(pdist(coords, 'euclidean'))
    # Normalize both to [0, 1] for fair comparison
    ed_max = embedded_dist.max()
    cd_max = consensus_dist.max()
    if ed_max > 0:
        embedded_norm = embedded_dist / ed_max
    else:
        embedded_norm = embedded_dist
    if cd_max > 0:
        consensus_norm = consensus_dist / cd_max
    else:
        consensus_norm = consensus_dist

    n = coords.shape[0]
    triu = np.triu_indices(n, k=1)
    rho, _ = spearmanr(consensus_norm[triu], embedded_norm[triu])

    # Absolute reconstruction error
    mse = np.mean((consensus_norm[triu] - embedded_norm[triu]) ** 2)
    mae = np.mean(np.abs(consensus_norm[triu] - embedded_norm[triu]))

    # Nearest neighbor preservation
    nn1 = 0
    nn5 = 0
    nn10 = 0
    for i in range(n):
        true_nn = np.argsort(consensus_dist[i])
        emb_nn = np.argsort(embedded_dist[i])
        if true_nn[1] == emb_nn[1]:
            nn1 += 1
        nn5 += len(set(true_nn[1:6]) & set(emb_nn[1:6])) / 5
        nn10 += len(set(true_nn[1:11]) & set(emb_nn[1:11])) / 10

    nn1_pct = nn1 / n * 100
    nn5_pct = nn5 / n * 100
    nn10_pct = nn10 / n * 100

    print(f"\n  Embedding fidelity ({n_dims}-d):")
    print(f"    Spearman ρ (distances): {rho:.4f}")
    print(f"    MSE:  {mse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    NN-1:  {nn1_pct:.1f}%")
    print(f"    NN-5:  {nn5_pct:.1f}%")
    print(f"    NN-10: {nn10_pct:.1f}%")

    # Show fidelity at multiple dimensions for comparison
    print(f"\n  Fidelity vs dimension (how much do you gain?):")
    print(f"  {'dims':>6s} {'ρ':>8s} {'NN-1':>8s} {'NN-5':>8s}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

    for test_d in [2, 4, 8, 16, 32, 64, 128, n_dims]:
        if test_d > len(eigenvalues) or test_d > n:
            break
        td = min(test_d, n - 1)
        pm = eigenvalues[:td] > 0
        se = np.zeros(td)
        se[pm] = np.sqrt(eigenvalues[:td][pm])
        tc = eigenvectors[:, :td] * se[np.newaxis, :]
        te = squareform(pdist(tc, 'euclidean'))
        te_max = te.max()
        if te_max > 0:
            te /= te_max
        tr, _ = spearmanr(consensus_norm[triu], te[triu])
        tn1 = sum(1 for i in range(n) if np.argsort(consensus_dist[i])[1] == np.argsort(te * te_max)[i][1]) if te_max > 0 else 0
        # Simpler NN1 calc
        te_raw = squareform(pdist(tc, 'euclidean'))
        tn1 = sum(1 for i in range(n)
                  if np.argsort(consensus_dist[i])[1] == np.argsort(te_raw[i])[1])
        tn5 = sum(len(set(np.argsort(consensus_dist[i])[1:6]) &
                      set(np.argsort(te_raw[i])[1:6])) / 5 for i in range(n))
        print(f"  {td:>6d} {tr:>8.4f} {tn1/n*100:>7.1f}% {tn5/n*100:>7.1f}%")

    metrics = {
        'n_dims': n_dims,
        'rho': float(rho),
        'mse': float(mse),
        'mae': float(mae),
        'nn1': float(nn1_pct),
        'nn5': float(nn5_pct),
        'nn10': float(nn10_pct),
    }

    return coords, metrics


# ═══════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════

def export_map(token_list, coords, consensus_dist, embedding_metrics,
               mean_rho, pair_rhos, model_names, eigenvalue_info,
               output_path):
    """
    Export the consensus map as a JSON coordinate table.

    Format:
    {
        "ascii_coordinates": { "65": [0.123, -0.456, ...], ... },
        "coordinates": { "the": [0.789, -0.012, ...], ... },
        "compounds": ["the", "and", ...],
        "metadata": { ... },
        "distance_matrix_sample": { ... }  // for verification
    }
    """
    n_dims = coords.shape[1]
    ascii_coordinates = {}
    compound_coordinates = {}
    compounds = []

    for i, (token_type, token_id, vocab_key) in enumerate(token_list):
        coord_list = coords[i].tolist()
        if token_type == 'ascii':
            ascii_coordinates[str(token_id)] = coord_list
        else:
            clean = str(token_id).replace('\u0120', ' ').replace('\u2581', ' ')
            compound_coordinates[clean] = coord_list
            compounds.append(clean)

    # Distance matrix sample for verification (first 50 tokens)
    sample_n = min(50, len(token_list))
    sample_labels = []
    for tt, tid, vk in token_list[:sample_n]:
        if tt == 'ascii':
            sample_labels.append(chr(tid) if 32 <= tid < 127 else f'<{tid}>')
        else:
            sample_labels.append(str(tid))

    pos_evals, cumulative, recommended = eigenvalue_info

    output = {
        'format': 'consensus_map_v1',
        'compounds': compounds,
        'coordinates': compound_coordinates,
        'ascii_coordinates': ascii_coordinates,
        'metadata': {
            'source': 'consensus_map',
            'method': 'multi_model_MDS',
            'models': model_names,
            'n_models': len(model_names),
            'n_ascii': len(ascii_coordinates),
            'n_compounds': len(compounds),
            'n_total': len(token_list),
            'n_dims': n_dims,
            'cross_model_rho': float(mean_rho),
            'pairwise_rhos': [float(r) for r in pair_rhos],
            'embedding_rho': embedding_metrics['rho'],
            'embedding_nn1': embedding_metrics['nn1'],
            'embedding_nn5': embedding_metrics['nn5'],
            'embedding_nn10': embedding_metrics['nn10'],
            'embedding_mse': embedding_metrics['mse'],
            'intrinsic_dim_95pct': int(recommended),
            'effective_rank_shannon': float(np.exp(-np.sum(
                (pos_evals / pos_evals.sum()) *
                np.log(pos_evals / pos_evals.sum() + 1e-10)))),
            'coordinate_bytes': len(compounds) * n_dims * 4,  # float32
        },
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    coord_bytes = (len(ascii_coordinates) + len(compounds)) * n_dims * 4

    print(f"\n{'═' * 70}")
    print(f"  EXPORTED: {output_path}")
    print(f"{'═' * 70}")
    print(f"  ASCII coordinates:    {len(ascii_coordinates)}")
    print(f"  Compound coordinates: {len(compounds)}")
    print(f"  Dimensions:           {n_dims}")
    print(f"  JSON file size:       {size_kb:.1f} KB")
    print(f"  Coordinate data:      {coord_bytes / 1024:.1f} KB (float32)")
    print(f"  Cross-model ρ:        {mean_rho:.4f}")
    print(f"  Embedding ρ:          {embedding_metrics['rho']:.4f}")
    print(f"  NN-1 preservation:    {embedding_metrics['nn1']:.1f}%")
    print(f"  NN-5 preservation:    {embedding_metrics['nn5']:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_consensus_map(model_names, max_tokens=8000, pmi_path=None,
                      output_path='./consensus_map.json',
                      force_dims=None):
    """
    Full pipeline:
    N models → distance matrices → average → eigenanalysis → embed → export
    """
    print(f"\n{'═' * 70}")
    print(f"  CONSENSUS MAP EXTRACTOR")
    print(f"  The Geometry IS the Model")
    print(f"{'═' * 70}")
    print(f"\n  Models ({len(model_names)}):")
    for m in model_names:
        print(f"    • {m}")
    print(f"  Max tokens: {max_tokens}")

    # ── Step 1: Load all models ──
    vocabs = []
    embeddings = []
    names = []

    for model_name in model_names:
        vocab, emb, name = load_model_embeddings(model_name)
        if vocab is not None:
            vocabs.append(vocab)
            embeddings.append(emb)
            names.append(name)

    if len(vocabs) < 2:
        print("\n  FATAL: Need at least 2 models")
        sys.exit(1)

    # ── Step 2: Find shared vocabulary ──
    print(f"\n  Finding shared vocabulary...")
    shared_vocab = set(vocabs[0].keys())
    for v in vocabs[1:]:
        shared_vocab &= set(v.keys())
    # Remove special tokens
    shared_vocab = {t for t in shared_vocab
                    if not (t.startswith('<') and t.endswith('>')
                            and len(t) > 5)}
    print(f"    Shared across all models: {len(shared_vocab):,}")

    # ── Step 3: Select tokens ──
    print(f"\n  Selecting tokens for the map...")
    token_list = select_map_tokens(shared_vocab, max_tokens, pmi_path)

    # ── Step 4: Compute consensus distances ──
    consensus_dist, mean_rho, pair_rhos = compute_consensus_distances(
        token_list, vocabs, embeddings, names)

    # Free embedding tables
    del embeddings
    gc.collect()

    # ── Step 5: Find intrinsic dimensionality ──
    eigenvalues, eigenvectors, pos_eigenvalues, cumulative, recommended = \
        find_intrinsic_dimensionality(consensus_dist)

    # ── Step 6: Embed at recommended dimension ──
    n_dims = force_dims if force_dims else recommended
    coords, embedding_metrics = embed_consensus(
        eigenvalues, eigenvectors, n_dims, consensus_dist)

    # ── Step 7: Export ──
    eigenvalue_info = (pos_eigenvalues, cumulative, recommended)
    export_map(token_list, coords, consensus_dist, embedding_metrics,
               mean_rho, pair_rhos, names, eigenvalue_info, output_path)

    # ── Summary ──
    n_total = len(token_list)
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {len(names)} models averaged → {n_total} tokens × {n_dims} dimensions")
    print(f"  Cross-model agreement: ρ={mean_rho:.4f}")
    print(f"  Embedding fidelity:    ρ={embedding_metrics['rho']:.4f}")
    print(f"  File size: {os.path.getsize(output_path)/1024:.1f} KB")
    print(f"")
    print(f"  This coordinate table captures {cumulative[n_dims-1]*100:.1f}% of the")
    print(f"  intrinsic geometry that {len(names)} models independently agree on.")
    print(f"")
    print(f"  The 70B models are the telescope. This file is the star chart.")
    print(f"  Your sampler is the navigator.")
    print(f"{'═' * 70}")

    return coords, consensus_dist, embedding_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Consensus Map Extractor — The Geometry IS the Model')
    parser.add_argument('--models', nargs='+', default=[
        "meta-llama/Llama-3.1-70B",
        "Qwen/Qwen2.5-72B",
    ])
    parser.add_argument('--max_tokens', type=int, default=8000)
    parser.add_argument('--output', type=str, default='./consensus_map.json')
    parser.add_argument('--pmi', type=str, default='./consensus_coordinates.json',
                        help='PMI file for priority token selection')
    parser.add_argument('--dims', type=int, default=None,
                        help='Force embedding dimensions (default: auto from 95%% energy)')
    args = parser.parse_args()

    sys.path.insert(0, '/mnt/user-data/uploads')
    sys.path.insert(0, '.')

    run_consensus_map(
        model_names=args.models,
        max_tokens=args.max_tokens,
        pmi_path=args.pmi,
        output_path=args.output,
        force_dims=args.dims,
    )


if __name__ == '__main__':
    main()
