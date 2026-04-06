"""
Mega Consensus Map — Maximum Diversity Embedding Extraction
============================================================
Jason (theory) + Claude (implementation) — 2026

Extracts embeddings from as many diverse models as possible,
computes consensus distance matrix, exports the map.

Only downloads the embedding shard (~4GB) + tokenizer per model.
No GPU needed. CPU + RAM only.

Usage:
    python mega_consensus.py
"""

import numpy as np
import json
import time
import sys
import os
import gc
import traceback
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

EMB_KEYS = ['embed_tokens', 'wte', 'word_embed', 'token_embed',
            'embedding', 'embeddings', 'word_embeddings',
            'shared.weight', 'tok_embeddings', 'word_embeddings_layernorm',
            'transformer.wte']


def find_embedding_shard(model_name):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(model_name)
    for candidate in ['model.safetensors.index.json',
                      'model.safetensors.index',
                      'pytorch_model.bin.index.json']:
        if candidate in files:
            if candidate.endswith('.json'):
                idx_path = hf_hub_download(model_name, candidate)
                with open(idx_path) as f:
                    index = json.load(f)
                weight_map = index.get('weight_map', {})
                for tensor_name, shard_file in weight_map.items():
                    if any(k in tensor_name.lower() for k in EMB_KEYS):
                        return shard_file, tensor_name
                first_shard = sorted(set(weight_map.values()))[0]
                return first_shard, None
    st_files = sorted([f for f in files if f.endswith('.safetensors')])
    if st_files:
        return st_files[0], None
    return None, None


def load_model(model_name):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

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
        print(f"    SKIP: no tokenizer")
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

    if not vocab:
        print(f"    SKIP: empty vocab")
        return None, None, None

    print(f"    Vocab: {len(vocab):,}")

    # Embedding
    shard_file, known_key = find_embedding_shard(model_name)
    if shard_file is None:
        print(f"    SKIP: no safetensors")
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
            for key in sorted(f.keys()):
                if any(k in key.lower() for k in EMB_KEYS):
                    tensor = f.get_tensor(key)
                    if tensor.dim() == 2 and tensor.shape[0] > 1000:
                        emb = tensor.float()
                        print(f"    Found embedding key: {key}")
                        break

    if emb is None:
        print(f"    SKIP: no embedding matrix found")
        return None, None, None

    dt = time.time() - t0
    print(f"    Embedding: {emb.shape} ({dt:.1f}s)")
    return vocab, emb, model_name


def main():
    print(f"{'═' * 70}")
    print(f"  MEGA CONSENSUS MAP — Maximum Diversity")
    print(f"{'═' * 70}")

    # All models we want to try, ordered by priority
    # Ungated first, then gated (may fail if license not accepted)
    ALL_MODELS = [
        # Already proven
        "meta-llama/Llama-3.1-70B",          # USA, Meta, dense
        "Qwen/Qwen2.5-72B",                  # China, Alibaba, dense
        "zai-org/GLM-5",                      # China, Zhipu, MoE, Huawei chips
        "google/gemma-2-27b",                 # USA, Google, dense

        # France
        "mistral-community/Mixtral-8x22B-v0.1",  # France, Mistral, MoE

        # China - more
        "deepseek-ai/DeepSeek-V3",           # China, DeepSeek, MoE+MLA
        "01-ai/Yi-1.5-34B",                  # China, 01.AI, dense

        # USA - diverse orgs
        "openai/gpt-oss-120b",               # USA, OpenAI, MoE+MXFP4
        "EleutherAI/gpt-neox-20b",           # USA, EleutherAI, dense
        "allenai/OLMo-2-0325-32B",           # USA, AI2, dense, fully open
        "databricks/dbrx-base",              # USA, Databricks, fine-grained MoE

        # Korea
        "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",  # Korea, LG, dense

        # India
        "sarvamai/sarvam-30b",               # India, Sarvam AI, MoE

        # UAE
        "tiiuae/falcon-40b",                 # UAE, TII, dense

        # Israel
        "ai21labs/AI21-Jamba-1.5-Large",     # Israel, AI21, hybrid SSM+MoE

        # International
        "bigscience/bloom",                   # International consortium, dense

        # Canada
        "CohereForAI/c4ai-command-r-plus",   # Canada, Cohere, dense

        # USA - more
        "ibm-granite/granite-20b-code-instruct-8k",  # USA, IBM, dense
    ]

    # Load all models, skip failures
    vocabs = []
    names = []
    # Store distance matrices incrementally to save RAM
    # (don't keep all embedding tables in memory at once)

    # First pass: load all vocabs to find shared tokens
    print(f"\n  Phase 1: Loading vocabularies...")
    all_vocabs = []
    valid_models = []

    for model_name in ALL_MODELS:
        try:
            vocab, emb, name = load_model(model_name)
            if vocab is not None and emb is not None:
                all_vocabs.append(vocab)
                valid_models.append(model_name)
                # Don't keep embedding in memory yet
                del emb
                gc.collect()
            else:
                print(f"    → SKIPPED")
        except Exception as e:
            print(f"    → ERROR: {e}")
            continue

    print(f"\n  Successfully loaded: {len(valid_models)} models")
    for m in valid_models:
        print(f"    ✓ {m}")

    if len(valid_models) < 2:
        print("  FATAL: Need at least 2 models")
        sys.exit(1)

    # Find shared vocabulary
    print(f"\n  Phase 2: Finding shared vocabulary...")
    shared = set(all_vocabs[0].keys())
    for v in all_vocabs[1:]:
        shared &= set(v.keys())
    shared = {t for t in shared
              if not (t.startswith('<') and t.endswith('>') and len(t) > 5)}
    print(f"    Shared across {len(valid_models)} models: {len(shared):,}")

    # Token selection
    print(f"\n  Phase 3: Selecting tokens...")

    # ASCII
    ascii_tokens = []
    for code in range(32, 127):
        char = chr(code)
        for c in [char, f'<0x{code:02X}>', '\u2581' + char]:
            if c in shared:
                ascii_tokens.append(('ascii', code, c))
                break

    # PMI compounds if available
    pmi_tokens = []
    pmi_path = './consensus_coordinates.json'
    if os.path.exists(pmi_path):
        with open(pmi_path) as f:
            pmi_data = json.load(f)
        for compound in pmi_data.get('compounds', []):
            clean = compound.strip()
            for variant in [compound, clean, '\u0120' + clean, '\u2581' + clean]:
                if variant in shared:
                    pmi_tokens.append(('pmi', compound, variant))
                    break

    # Fill with words
    seen = set(t[2] for t in ascii_tokens + pmi_tokens)
    word_tokens = []
    for tok in sorted(shared):
        if tok in seen:
            continue
        clean = tok.replace('\u0120', '').replace('\u2581', '').strip()
        if not clean or len(clean) < 3 or len(clean) > 12:
            continue
        if not all(ord(c) < 128 for c in clean):
            continue
        if not clean.isalpha():
            continue
        if tok.startswith('<') and tok.endswith('>'):
            continue
        if any(c.isupper() for c in clean[1:]) and any(c.islower() for c in clean):
            continue
        word_tokens.append(('word', clean, tok))

    MAX_TOKENS = 8000
    token_list = ascii_tokens + pmi_tokens
    budget = MAX_TOKENS - len(token_list)
    if budget > 0:
        token_list.extend(word_tokens[:budget])

    n_tokens = len(token_list)
    print(f"    ASCII: {len(ascii_tokens)}")
    print(f"    PMI: {len(pmi_tokens)}")
    print(f"    Words: {min(budget, len(word_tokens))}")
    print(f"    Total: {n_tokens}")

    # Phase 4: Compute distance matrices one model at a time
    # This way we only hold one embedding table + one distance matrix at a time
    print(f"\n  Phase 4: Computing distance matrices (one at a time)...")

    all_dist = []
    successful_names = []

    for model_name in valid_models:
        try:
            print(f"\n  Processing {model_name}...")
            t0 = time.time()

            vocab, emb, name = load_model(model_name)
            if vocab is None or emb is None:
                print(f"    → SKIPPED on reload")
                continue

            # Extract vectors for selected tokens
            vecs = []
            for ttype, tid, vkey in token_list:
                if ttype == 'ascii':
                    char = chr(tid)
                    found = False
                    for c in [char, vkey, f'<0x{tid:02X}>', '\u2581' + char]:
                        if c in vocab and vocab[c] < emb.shape[0]:
                            vecs.append(emb[vocab[c]].numpy())
                            found = True
                            break
                    if not found:
                        vecs.append(np.zeros(emb.shape[1]))
                else:
                    found = False
                    for variant in [vkey, str(tid),
                                    '\u0120' + str(tid).strip(),
                                    '\u2581' + str(tid).strip()]:
                        if variant in vocab and vocab[variant] < emb.shape[0]:
                            vecs.append(emb[vocab[variant]].numpy())
                            found = True
                            break
                    if not found:
                        vecs.append(np.zeros(emb.shape[1]))

            # Free embedding immediately
            del emb, vocab
            gc.collect()

            mat = np.array(vecs)
            del vecs
            gc.collect()

            # Normalize
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            mat /= norms

            # Distance matrix
            dist = squareform(pdist(mat, 'cosine'))
            del mat
            gc.collect()

            # Normalize to [0,1]
            d_max = dist.max()
            if d_max > 0:
                dist /= d_max

            all_dist.append(dist)
            successful_names.append(model_name)

            dt = time.time() - t0
            print(f"    Done ({dt:.1f}s) — {len(all_dist)} models complete")

        except Exception as e:
            print(f"    → ERROR: {e}")
            traceback.print_exc()
            gc.collect()
            continue

    n_models = len(all_dist)
    print(f"\n{'═' * 70}")
    print(f"  RESULTS: {n_models} models successfully processed")
    print(f"{'═' * 70}")
    for m in successful_names:
        print(f"    ✓ {m.split('/')[-1]}")

    if n_models < 2:
        print("  FATAL: Need at least 2")
        sys.exit(1)

    # Average
    print(f"\n  Averaging {n_models} distance matrices...")
    consensus = np.mean(all_dist, axis=0)

    # Cross-model agreement
    triu = np.triu_indices(n_tokens, k=1)
    print(f"\n  Cross-model agreement (pairwise ρ):")
    pair_rhos = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rho, _ = spearmanr(all_dist[i][triu], all_dist[j][triu])
            na = successful_names[i].split('/')[-1]
            nb = successful_names[j].split('/')[-1]
            print(f"    {na:>25s} ↔ {nb:<25s}: ρ={rho:.4f}")
            pair_rhos.append(rho)
    mean_rho = np.mean(pair_rhos)
    print(f"\n    Mean ρ: {mean_rho:.4f}")

    # Free individual matrices
    del all_dist
    gc.collect()

    # Eigenvalue analysis
    print(f"\n  Eigenvalue analysis...")
    D_sq = consensus ** 2
    n = consensus.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ D_sq @ H
    del D_sq, H
    gc.collect()

    eigenvalues, eigenvectors = np.linalg.eigh(G)
    del G
    gc.collect()

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos_evals = eigenvalues[eigenvalues > 0]
    total_energy = pos_evals.sum()
    cumulative = np.cumsum(pos_evals) / total_energy
    neg_energy = abs(eigenvalues[eigenvalues < 0].sum()) / total_energy * 100

    for thresh in [0.80, 0.90, 0.95, 0.99]:
        dims = int(np.searchsorted(cumulative, thresh) + 1)
        print(f"    {thresh*100:.0f}% energy: {dims} dims")

    eff_rank = float(np.exp(-np.sum((pos_evals/total_energy) *
                                     np.log(pos_evals/total_energy + 1e-10))))
    print(f"    Effective rank: {eff_rank:.1f}")
    print(f"    Negative eigenvalue energy: {neg_energy:.2f}%")

    # Embed at 95% energy, capped at 2048
    recommended = int(np.searchsorted(cumulative, 0.95) + 1)
    n_dims = min(recommended, 2048)
    print(f"\n  Embedding at {n_dims} dims (95% = {recommended})...")

    sqrt_evals = np.zeros(n_dims)
    pos_mask = eigenvalues[:n_dims] > 0
    sqrt_evals[pos_mask] = np.sqrt(eigenvalues[:n_dims][pos_mask])
    coords = eigenvectors[:, :n_dims] * sqrt_evals[np.newaxis, :]

    # Fidelity check
    emb_dist = squareform(pdist(coords, 'euclidean'))
    ed_max = emb_dist.max()
    if ed_max > 0:
        emb_norm = emb_dist / ed_max
    cd_max = consensus.max()
    if cd_max > 0:
        cons_norm = consensus / cd_max

    rho_emb, _ = spearmanr(cons_norm[triu], emb_norm[triu])
    nn1 = sum(1 for i in range(n_tokens)
              if np.argsort(consensus[i])[1] == np.argsort(emb_dist[i])[1])
    nn5 = sum(len(set(np.argsort(consensus[i])[1:6]) &
                  set(np.argsort(emb_dist[i])[1:6])) / 5
              for i in range(n_tokens))

    print(f"    ρ: {rho_emb:.4f}")
    print(f"    NN-1: {nn1/n_tokens*100:.1f}%")
    print(f"    NN-5: {nn5/n_tokens*100:.1f}%")

    # Export
    print(f"\n  Exporting...")
    ascii_coordinates = {}
    compound_coordinates = {}
    compounds = []

    for i, (ttype, tid, vkey) in enumerate(token_list):
        coord_list = coords[i].tolist()
        if ttype == 'ascii':
            ascii_coordinates[str(tid)] = coord_list
        else:
            clean = str(tid).replace('\u0120', ' ').replace('\u2581', ' ')
            compound_coordinates[clean] = coord_list
            compounds.append(clean)

    output = {
        'format': 'consensus_map_v1',
        'compounds': compounds,
        'coordinates': compound_coordinates,
        'ascii_coordinates': ascii_coordinates,
        'metadata': {
            'source': 'mega_consensus',
            'method': 'multi_model_MDS',
            'models': successful_names,
            'n_models': n_models,
            'n_ascii': len(ascii_coordinates),
            'n_compounds': len(compounds),
            'n_total': n_tokens,
            'n_dims': n_dims,
            'cross_model_rho': float(mean_rho),
            'pairwise_rhos': [float(r) for r in pair_rhos],
            'embedding_rho': float(rho_emb),
            'embedding_nn1': float(nn1 / n_tokens * 100),
            'embedding_nn5': float(nn5 / n_tokens * 100),
            'negative_eigenvalue_energy': float(neg_energy),
            'effective_rank': float(eff_rank),
        },
    }

    output_path = './mega_consensus_map.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False,
                  default=lambda x: int(x) if hasattr(x, 'item') else x)

    size_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"\n{'═' * 70}")
    print(f"  DONE: {output_path}")
    print(f"  Size: {size_mb:.0f} MB")
    print(f"  Models: {n_models}")
    print(f"  Tokens: {n_tokens}")
    print(f"  Dims: {n_dims}")
    print(f"  Cross-model ρ: {mean_rho:.4f}")
    print(f"  Embedding ρ: {rho_emb:.4f}")
    print(f"  NN-1: {nn1/n_tokens*100:.1f}%  NN-5: {nn5/n_tokens*100:.1f}%")
    print(f"  Negative eigenvalue energy: {neg_energy:.2f}%")
    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()
