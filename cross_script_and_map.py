"""
Cross-Script Geometry Test + Fixed Consensus Map
==================================================
Jason (theory) + Claude (implementation) — 2026

Part 1: Test whether Korean and Latin characters have the same
         cross-model geometric agreement as ASCII (ρ=0.79).
         Fast — just loads embeddings and computes correlations.

Part 2: Rerun consensus map with the expensive fidelity-vs-dimension
         loop removed so it actually finishes.

Usage:
    # Test cross-script geometry
    python cross_script_and_map.py test

    # Run the fixed consensus map
    python cross_script_and_map.py map --models meta-llama/Llama-3.1-70B Qwen/Qwen2.5-72B zai-org/GLM-5 google/gemma-2-27b

    # Both
    python cross_script_and_map.py both
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
#  SHARED: Embedding loader
# ═══════════════════════════════════════════════════════════════

EMB_KEYS = ['embed_tokens', 'wte', 'word_embed', 'token_embed',
            'embedding', 'embeddings', 'word_embeddings',
            'shared.weight', 'tok_embeddings']


def load_model(model_name):
    """Load vocab + embedding table from any HF model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from embedding_cartographer import find_embedding_shard

    print(f"\n  Loading {model_name}...")
    t0 = time.time()

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

    print(f"    Vocab: {len(vocab):,}")

    shard_file, known_key = find_embedding_shard(model_name)
    if shard_file is None:
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
        return None, None, None

    dt = time.time() - t0
    print(f"    Embedding: {emb.shape} ({dt:.1f}s)")
    return vocab, emb, model_name


# ═══════════════════════════════════════════════════════════════
#  PART 1: Cross-Script Geometry Test
# ═══════════════════════════════════════════════════════════════

# Korean Jamo (consonants and vowels)
JAMO_CHARS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # consonants
                  "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")  # vowels

# Common Korean syllables
KOREAN_SYLLABLES = list("가나다라마바사아자차카타파하"
                        "고노도로모보소오조초코토포호"
                        "구누두루무부수우주추쿠투푸후"
                        "기니디리미비시이지치키티피히"
                        "게네데레메베세에제체케테페헤")

# Latin extended (accented characters common in European languages)
LATIN_EXTENDED = list("àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ"
                      "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"
                      "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
                      "ăîșțĂÎȘȚ"
                      "čďěňřšťůžČĎĚŇŘŠŤŮŽ")

# CJK common characters
CJK_COMMON = list("的一是不了人我在有他这为之大来以个中上们到说时地也子就道"
                  "要出会可她那你对生能而行方过家十用发天如然但信前所二起与"
                  "同工已下手小让开正新因其从被好看学进种将还分此心前经长")


def find_char_tokens(vocab, char_list, label):
    """Find which characters from char_list exist in the vocabulary."""
    found = {}
    for char in char_list:
        # Try direct match and common BPE variants
        for candidate in [char, '\u2581' + char, '\u0120' + char]:
            if candidate in vocab:
                found[char] = vocab[candidate]
                break
    print(f"    {label}: {len(found)}/{len(char_list)} found")
    return found


def compute_cross_model_rho(emb_a, emb_b, shared_chars, char_map_a, char_map_b):
    """Compute Spearman ρ between two models on shared characters."""
    if len(shared_chars) < 5:
        return 0.0, 0

    mat_a = np.array([emb_a[char_map_a[c]].numpy() for c in shared_chars])
    mat_b = np.array([emb_b[char_map_b[c]].numpy() for c in shared_chars])

    # Normalize
    mat_a = mat_a / (np.linalg.norm(mat_a, axis=1, keepdims=True) + 1e-8)
    mat_b = mat_b / (np.linalg.norm(mat_b, axis=1, keepdims=True) + 1e-8)

    dist_a = squareform(pdist(mat_a, 'cosine'))
    dist_b = squareform(pdist(mat_b, 'cosine'))

    triu = np.triu_indices(len(shared_chars), k=1)
    rho, p = spearmanr(dist_a[triu], dist_b[triu])
    return rho, len(shared_chars)


def run_cross_script_test(model_names):
    """Test geometry agreement across writing systems."""
    print(f"\n{'═' * 70}")
    print(f"  CROSS-SCRIPT GEOMETRY TEST")
    print(f"  Does character geometry extend beyond ASCII?")
    print(f"{'═' * 70}")

    # Load models
    models = []
    for name in model_names:
        vocab, emb, mname = load_model(name)
        if vocab is not None:
            models.append((vocab, emb, mname))

    if len(models) < 2:
        print("  FATAL: Need at least 2 models")
        return

    # Find characters in each model's vocabulary
    char_sets = {
        'ASCII printable': [chr(c) for c in range(32, 127)],
        'Korean Jamo': JAMO_CHARS,
        'Korean Syllables': KOREAN_SYLLABLES,
        'Latin Extended': LATIN_EXTENDED,
        'CJK Common': CJK_COMMON,
    }

    print(f"\n  Finding characters in vocabularies...")
    model_char_maps = []
    for vocab, emb, mname in models:
        short = mname.split('/')[-1]
        print(f"\n  {short}:")
        maps = {}
        for set_name, chars in char_sets.items():
            maps[set_name] = find_char_tokens(vocab, chars, set_name)
        model_char_maps.append(maps)

    # Pairwise comparison for each character set
    print(f"\n{'═' * 70}")
    print(f"  CROSS-MODEL AGREEMENT BY WRITING SYSTEM")
    print(f"{'═' * 70}")

    for set_name, chars in char_sets.items():
        print(f"\n  {set_name}:")

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                vocab_a, emb_a, name_a = models[i]
                vocab_b, emb_b, name_b = models[j]
                map_a = model_char_maps[i][set_name]
                map_b = model_char_maps[j][set_name]

                # Shared characters
                shared = sorted(set(map_a.keys()) & set(map_b.keys()))
                if len(shared) < 5:
                    short_a = name_a.split('/')[-1]
                    short_b = name_b.split('/')[-1]
                    print(f"    {short_a} ↔ {short_b}: only {len(shared)} shared, skipping")
                    continue

                rho, n = compute_cross_model_rho(
                    emb_a, emb_b, shared, map_a, map_b)

                short_a = name_a.split('/')[-1]
                short_b = name_b.split('/')[-1]
                print(f"    {short_a} ↔ {short_b}: ρ={rho:.4f} (n={n})")

    # All-pairs summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n  If Korean/Latin/CJK ρ values are comparable to ASCII ρ,")
    print(f"  the geometry is universal across writing systems.")
    print(f"  If they're higher, those scripts have STRONGER geometric")
    print(f"  structure (Korean Jamo would be expected to — it's phonemic).")
    print(f"  If they're near zero, the geometry is English-specific.")

    # Free memory
    del models
    gc.collect()


# ═══════════════════════════════════════════════════════════════
#  PART 2: Fixed Consensus Map (no expensive fidelity loop)
# ═══════════════════════════════════════════════════════════════

def run_fixed_consensus_map(model_names, max_tokens=8000, pmi_path=None,
                            output_path='./consensus_map_4model.json'):
    """Consensus map without the loop that hangs."""

    print(f"\n{'═' * 70}")
    print(f"  CONSENSUS MAP (fixed — no fidelity loop)")
    print(f"{'═' * 70}")

    # Load models
    vocabs = []
    embeddings = []
    names = []
    for name in model_names:
        vocab, emb, mname = load_model(name)
        if vocab is not None:
            vocabs.append(vocab)
            embeddings.append(emb)
            names.append(mname)

    if len(vocabs) < 2:
        print("  FATAL: Need at least 2 models")
        return

    # Shared vocabulary
    print(f"\n  Finding shared vocabulary...")
    shared_vocab = set(vocabs[0].keys())
    for v in vocabs[1:]:
        shared_vocab &= set(v.keys())
    shared_vocab = {t for t in shared_vocab
                    if not (t.startswith('<') and t.endswith('>') and len(t) > 5)}
    print(f"    Shared: {len(shared_vocab):,}")

    # Token selection (inline, no import needed)
    print(f"\n  Selecting tokens...")

    # ASCII
    ascii_tokens = []
    for code in range(32, 127):
        char = chr(code)
        for c in [char, f'<0x{code:02X}>', '\u2581' + char]:
            if c in shared_vocab:
                ascii_tokens.append(('ascii', code, c))
                break

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

    # Fill remaining with words
    seen = set(t[2] for t in ascii_tokens + pmi_tokens)
    word_tokens = []
    for tok in sorted(shared_vocab):
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

    token_list = ascii_tokens + pmi_tokens
    budget = max_tokens - len(token_list)
    if budget > 0:
        token_list.extend(word_tokens[:budget])

    print(f"    ASCII: {len(ascii_tokens)}")
    print(f"    PMI: {len(pmi_tokens)}")
    print(f"    Words: {min(budget, len(word_tokens))}")
    print(f"    Total: {len(token_list)}")

    # Distance matrices
    n_tokens = len(token_list)
    n_models = len(vocabs)
    all_dist = []

    for m_idx in range(n_models):
        vocab = vocabs[m_idx]
        emb = embeddings[m_idx]
        short = names[m_idx].split('/')[-1]
        print(f"\n  Distances for {short}...")
        t0 = time.time()

        vecs = []
        for token_type, token_id, vocab_key in token_list:
            if token_type == 'ascii':
                char = chr(token_id)
                found = False
                for c in [char, vocab_key, f'<0x{token_id:02X}>', '\u2581' + char]:
                    if c in vocab and vocab[c] < emb.shape[0]:
                        vecs.append(emb[vocab[c]].numpy())
                        found = True
                        break
                if not found:
                    vecs.append(np.zeros(emb.shape[1]))
            else:
                found = False
                for variant in [vocab_key, str(token_id),
                                '\u0120' + str(token_id).strip(),
                                '\u2581' + str(token_id).strip()]:
                    if variant in vocab and vocab[variant] < emb.shape[0]:
                        vecs.append(emb[vocab[variant]].numpy())
                        found = True
                        break
                if not found:
                    vecs.append(np.zeros(emb.shape[1]))

        mat = np.array(vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        mat_normed = mat / norms

        dist = squareform(pdist(mat_normed, 'cosine'))
        d_max = dist.max()
        if d_max > 0:
            dist /= d_max
        all_dist.append(dist)
        print(f"    Done ({time.time()-t0:.1f}s)")

    # Free embeddings
    del embeddings
    gc.collect()

    # Average
    print(f"\n  Averaging {n_models} distance matrices...")
    consensus = np.mean(all_dist, axis=0)

    # Cross-model agreement
    triu = np.triu_indices(n_tokens, k=1)
    print(f"\n  Cross-model agreement:")
    pair_rhos = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rho, _ = spearmanr(all_dist[i][triu], all_dist[j][triu])
            na = names[i].split('/')[-1]
            nb = names[j].split('/')[-1]
            print(f"    {na} ↔ {nb}: ρ={rho:.4f}")
            pair_rhos.append(rho)
    mean_rho = np.mean(pair_rhos)
    print(f"    Mean: ρ={mean_rho:.4f}")

    # Eigenvalue analysis
    print(f"\n  Eigenvalue analysis...")
    D_sq = consensus ** 2
    n = consensus.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ D_sq @ H

    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos_eigenvalues = eigenvalues[eigenvalues > 0]
    total_energy = pos_eigenvalues.sum()
    cumulative = np.cumsum(pos_eigenvalues) / total_energy

    neg_energy = abs(eigenvalues[eigenvalues < 0].sum()) / total_energy * 100

    # Find dimensions at thresholds
    for thresh in [0.80, 0.90, 0.95, 0.99]:
        dims = np.searchsorted(cumulative, thresh) + 1
        print(f"    {thresh*100:.0f}% energy: {dims} dims")

    eff_rank = np.exp(-np.sum((pos_eigenvalues/total_energy) *
                               np.log(pos_eigenvalues/total_energy + 1e-10)))
    print(f"    Effective rank: {eff_rank:.1f}")
    print(f"    Negative eigenvalue energy: {neg_energy:.2f}%")

    # Embed at 95% energy
    recommended = np.searchsorted(cumulative, 0.95) + 1
    # Cap at something reasonable to avoid memory issues
    n_dims = min(recommended, 512)
    print(f"\n  Embedding at {n_dims} dims (capped from {recommended})...")

    sqrt_evals = np.zeros(n_dims)
    pos_mask = eigenvalues[:n_dims] > 0
    sqrt_evals[pos_mask] = np.sqrt(eigenvalues[:n_dims][pos_mask])
    coords = eigenvectors[:, :n_dims] * sqrt_evals[np.newaxis, :]

    # Quick fidelity check (no loop, just the final dimension)
    emb_dist = squareform(pdist(coords, 'euclidean'))
    ed_max = emb_dist.max()
    if ed_max > 0:
        emb_norm = emb_dist / ed_max
    else:
        emb_norm = emb_dist
    cd_max = consensus.max()
    if cd_max > 0:
        cons_norm = consensus / cd_max
    else:
        cons_norm = consensus

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
            'source': 'consensus_map_4model',
            'method': 'multi_model_MDS',
            'models': names,
            'n_models': len(names),
            'n_ascii': len(ascii_coordinates),
            'n_compounds': len(compounds),
            'n_total': len(token_list),
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

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n{'═' * 70}")
    print(f"  EXPORTED: {output_path}")
    print(f"  Size: {size_kb:.0f} KB")
    print(f"  Tokens: {len(token_list)} ({len(ascii_coordinates)} ASCII + {len(compounds)} compounds)")
    print(f"  Dims: {n_dims}")
    print(f"  Cross-model ρ: {mean_rho:.4f}")
    print(f"  Embedding ρ: {rho_emb:.4f}")
    print(f"  NN-1: {nn1/n_tokens*100:.1f}%  NN-5: {nn5/n_tokens*100:.1f}%")
    print(f"{'═' * 70}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['test', 'map', 'both'],
                        help='test=cross-script geometry, map=consensus map, both=both')
    parser.add_argument('--models', nargs='+', default=[
        "meta-llama/Llama-3.1-70B",
        "Qwen/Qwen2.5-72B",
        "zai-org/GLM-5",
        "google/gemma-2-27b",
    ])
    parser.add_argument('--max_tokens', type=int, default=8000)
    parser.add_argument('--pmi', type=str, default='./consensus_coordinates.json')
    parser.add_argument('--output', type=str, default='./consensus_map_4model.json')
    args = parser.parse_args()

    sys.path.insert(0, '.')

    if args.mode in ('test', 'both'):
        run_cross_script_test(args.models)

    if args.mode in ('map', 'both'):
        run_fixed_consensus_map(
            args.models, args.max_tokens, args.pmi, args.output)


if __name__ == '__main__':
    main()
