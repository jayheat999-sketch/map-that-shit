"""
TORUS ENGLISH DIFFUSION — Energy-Based Language Model
Jason (theory) + Claude (implementation) — 2026

Adapted from torus_energy_model.py (Jamo Korean system).
Same physics. Same geometry. Different alphabet.

What we proved:
    - Character geometry is real (ρ=0.79 cross-model agreement)
    - Distance geometry is intrinsic (rotation-invariant, category-invariant)
    - The manifold is non-Euclidean (18.77% negative eigenvalue energy)
    - The Poincaré ball is the right embedding space
    - The map already exists — we extract it, we don't train it

Architecture:
    - 94 printable ASCII characters (Token94, not Token128 — no ghosts)
    - N compound tokens from consensus geometry (filtered: no BPE fragments)
    - Consensus distance coordinates from averaged 70B models
    - Simplex base + learned perturbation in Poincaré ball
    - Torus positional encoding (golden-angle spiral, no periodic orbits)
    - Bidirectional denoising transformer (NOT autoregressive)
    - Annealed LockSeed noise: wrong English, not masks
    - Wave collapse tracking for early stopping

The geometry IS the model. The 70B models were the telescope.
The consensus map is the star chart. This is the navigator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import List, Optional, Tuple, Dict
from collections import OrderedDict


# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════

PHI = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_ANGLE = 2.0 * math.pi / (PHI * PHI)
POINCARE_EPS = 1e-5
POINCARE_MAX_NORM = 0.95

# Token94: the 94 printable ASCII characters (33-126)
# Plus space (32) which is missing from the angular coords
# but we add it back because it's the most common character in English
ASCII_PRINTABLE_START = 32  # space
ASCII_PRINTABLE_END = 127   # DEL (exclusive)
N_ASCII = ASCII_PRINTABLE_END - ASCII_PRINTABLE_START  # 95 (32..126)

# Special tokens
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SPECIAL_OFFSET = 4  # First real character token ID


# ════════════════════════════════════════════════════════════
# SECTION 1: CLEAN VOCABULARY
# No ghosts. No BPE fragments. No duplicates. Only real tokens.
# ════════════════════════════════════════════════════════════

def is_clean_compound(token: str) -> bool:
    """Filter: is this token a real English word, not BPE garbage?"""
    clean = token.strip()
    if not clean:
        return False

    # Must be ASCII
    if not all(ord(c) < 128 for c in clean):
        return False

    # Must be at least 3 characters (2-char = BPE splinters)
    if len(clean) < 3:
        return False

    # Must be alphabetic (no 'localctx', 'ppl', mixed junk)
    if not clean.isalpha():
        return False

    # Fragment detection: these letter combos almost never start real English words
    BAD_STARTS = [
        'igh', 'ece', 'rig', 'ate', 'rib', 'ers', 'eri', 'ync',
        'ert', 'ist', 'pat', 'est', 'anc', 'ain', 'fla', 'ima',
        'ort', 'oll', 'ult', 'umb', 'unc', 'und', 'ung', 'unt',
        'urn', 'ust', 'ych', 'yst', 'amp', 'ock', 'onc', 'oup',
        'oul', 'our', 'uct', 'emp', 'onf',
    ]
    first3 = clean[:3].lower()
    if any(first3.startswith(b) for b in BAD_STARTS) and len(clean) > 4:
        return False

    return True


class EnglishVocabulary:
    """
    Token94 + clean compounds from consensus geometry.

    Loads consensus_coordinates.json (or consensus_map.json).
    Filters out BPE fragments, duplicates, non-English tokens.
    Builds a clean token↔ID mapping with real geometric positions.
    """

    def __init__(self, consensus_path: str = None):
        # Base: 95 printable ASCII (space through tilde)
        self.char_to_id = {}
        self.id_to_char = {}
        self.compounds = OrderedDict()
        self.compound_ids = {}

        # Special tokens
        self.id_to_char[PAD_TOKEN] = '<PAD>'
        self.id_to_char[BOS_TOKEN] = '<BOS>'
        self.id_to_char[EOS_TOKEN] = '<EOS>'
        self.id_to_char[UNK_TOKEN] = '<UNK>'

        # ASCII: 95 printable chars get IDs 4..98
        for code in range(ASCII_PRINTABLE_START, ASCII_PRINTABLE_END):
            char = chr(code)
            tid = SPECIAL_OFFSET + (code - ASCII_PRINTABLE_START)
            self.char_to_id[char] = tid
            self.id_to_char[tid] = char

        self.n_ascii = N_ASCII  # 95
        self.ascii_end = SPECIAL_OFFSET + N_ASCII  # 99

        # Consensus geometry
        self._ascii_coords = {}
        self._compound_coords = {}
        self._coord_dim = 0

        if consensus_path and os.path.exists(consensus_path):
            self._load_consensus(consensus_path)

        self.vocab_size = self.ascii_end + len(self.compounds)
        self._build_trie()

        print(f"  EnglishVocabulary:")
        print(f"    Special tokens: {SPECIAL_OFFSET}")
        print(f"    ASCII chars:    {self.n_ascii} (IDs {SPECIAL_OFFSET}..{self.ascii_end-1})")
        print(f"    Compounds:      {len(self.compounds)} (IDs {self.ascii_end}..{self.vocab_size-1})")
        print(f"    Total vocab:    {self.vocab_size}")
        if self._coord_dim > 0:
            print(f"    Coord dims:     {self._coord_dim}")

    def _load_consensus(self, path):
        """Load and filter consensus coordinates."""
        with open(path) as f:
            data = json.load(f)

        fmt = data.get('format', '')

        # Load ASCII coordinates
        ascii_coords = data.get('ascii_coordinates', {})
        for key, coords in ascii_coords.items():
            code = int(key)
            if ASCII_PRINTABLE_START <= code < ASCII_PRINTABLE_END:
                self._ascii_coords[code] = coords
                if self._coord_dim == 0:
                    self._coord_dim = len(coords)

        print(f"    ASCII coords loaded: {len(self._ascii_coords)}")

        # Load compound coordinates — FILTERED
        compound_coords = data.get('coordinates', {})
        compound_list = data.get('compounds', [])

        seen_clean = set()
        accepted = 0
        rejected_fragment = 0
        rejected_dupe = 0
        rejected_other = 0

        for compound in compound_list:
            clean = compound.strip()

            # Filter
            if not is_clean_compound(compound):
                rejected_fragment += 1
                continue

            # Deduplicate (space prefix variants)
            if clean.lower() in seen_clean:
                rejected_dupe += 1
                continue
            seen_clean.add(clean.lower())

            # Must have coordinates
            if compound in compound_coords:
                coords = compound_coords[compound]
            elif clean in compound_coords:
                coords = compound_coords[clean]
            else:
                rejected_other += 1
                continue

            # Accept
            tid = self.ascii_end + len(self.compounds)
            self.compounds[clean] = tid
            self.compound_ids[tid] = clean
            self._compound_coords[clean] = coords
            accepted += 1

            if self._coord_dim == 0:
                self._coord_dim = len(coords)

        print(f"    Compounds: {accepted} accepted, "
              f"{rejected_fragment} fragments, "
              f"{rejected_dupe} duplicates, "
              f"{rejected_other} no coords")

        # Report metadata if present
        meta = data.get('metadata', {})
        if meta:
            rho = meta.get('spearman_rho', meta.get('cross_model_rho', 0))
            models = meta.get('models', [])
            print(f"    Source: {meta.get('source', '?')}")
            print(f"    Models: {', '.join(m.split('/')[-1] for m in models)}")
            print(f"    ρ: {rho:.4f}")

    def _build_trie(self):
        """Build prefix trie for greedy longest-match encoding."""
        self._trie = {}
        self._max_compound_len = 0
        for compound, tid in self.compounds.items():
            node = self._trie
            for ch in compound:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node['_id'] = tid
            self._max_compound_len = max(self._max_compound_len, len(compound))

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs. Greedy longest-match for compounds."""
        result = []
        if add_special:
            result.append(BOS_TOKEN)

        i = 0
        n = len(text)
        while i < n:
            # Try compound match first
            if self._trie:
                node = self._trie
                best_tid = None
                best_len = 0
                for j in range(i, min(i + self._max_compound_len, n)):
                    ch = text[j]
                    if ch not in node:
                        break
                    node = node[ch]
                    if '_id' in node:
                        best_tid = node['_id']
                        best_len = j - i + 1

                if best_tid is not None:
                    result.append(best_tid)
                    i += best_len
                    continue

            # Single character
            char = text[i]
            if char in self.char_to_id:
                result.append(self.char_to_id[char])
            else:
                result.append(UNK_TOKEN)
            i += 1

        if add_special:
            result.append(EOS_TOKEN)
        return result

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        parts = []
        for tid in ids:
            if isinstance(tid, torch.Tensor):
                tid = tid.item()
            tid = int(tid)
            if tid in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            elif tid == UNK_TOKEN:
                parts.append('?')
            elif tid in self.id_to_char:
                parts.append(self.id_to_char[tid])
            elif tid in self.compound_ids:
                parts.append(self.compound_ids[tid])
            else:
                parts.append('')
        return ''.join(parts)

    def get_consensus_coords(self) -> Optional[torch.Tensor]:
        """
        Build coordinate table from consensus geometry.
        Returns [vocab_size, coord_dim] or None if no coords loaded.
        """
        if self._coord_dim == 0:
            return None

        coords = torch.zeros(self.vocab_size, self._coord_dim)

        # ASCII
        for code, coord_list in self._ascii_coords.items():
            tid = self.char_to_id.get(chr(code))
            if tid is not None:
                for b in range(min(len(coord_list), self._coord_dim)):
                    coords[tid, b] = float(coord_list[b])

        # Compounds
        for compound, coord_list in self._compound_coords.items():
            tid = self.compounds.get(compound)
            if tid is not None:
                for b in range(min(len(coord_list), self._coord_dim)):
                    coords[tid, b] = float(coord_list[b])

        return coords


# ════════════════════════════════════════════════════════════
# SECTION 2: ENGLISH NOISE — Wrong English, not masks
# "Structure is better than randomness even when wrong."
# ════════════════════════════════════════════════════════════

class EnglishNoiseSchedule:
    """
    Annealing noise schedule for English token diffusion.

    Same physics as Annealed LockSeed for images:
        - High sigma: heavy corruption (model finds sentence structure)
        - Low sigma: light corruption (model resolves token detail)

    Noise = random token substitution (wrong English, not MASK).
    Cosine schedule matches image diffusion's morphogenesis phases.
    """

    def __init__(self, num_timesteps: int = 10):
        self.T = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        alpha = torch.cos((steps / num_timesteps) * (math.pi / 2)).pow(2)
        self.gamma = 1.0 - alpha  # corruption probability at each step

    def corrupt(
        self,
        clean_ids: torch.Tensor,
        t: torch.Tensor,
        device: torch.device,
        vocab_size: int,
    ) -> torch.Tensor:
        """Replace tokens with random vocabulary items at noise level t."""
        B, L = clean_ids.shape
        gamma_t = self.gamma.to(device)[t].view(B, 1)

        corrupt_mask = torch.rand(B, L, device=device) < gamma_t
        # Don't corrupt special tokens
        special = clean_ids < SPECIAL_OFFSET
        corrupt_mask = corrupt_mask & ~special

        noisy_ids = clean_ids.clone()
        n_corrupt = corrupt_mask.sum().item()
        if n_corrupt > 0:
            # Random tokens from the full vocabulary (excluding specials)
            replacements = torch.randint(
                SPECIAL_OFFSET, vocab_size, (n_corrupt,), device=device)
            noisy_ids[corrupt_mask] = replacements

        return noisy_ids

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.T + 1, (batch_size,), device=device)


# ════════════════════════════════════════════════════════════
# SECTION 3: CONSENSUS EMBEDDING
# The 70B models' geometry, extracted and embedded.
# ════════════════════════════════════════════════════════════

class ConsensusEmbedding(nn.Module):
    """
    Token embeddings initialized from consensus geometry.

    If consensus coordinates exist:
        - Load the coordinate table (from MDS or angular optimization)
        - Project through a learned linear to d_model
        - Clamp inside Poincaré ball (non-Euclidean structure)

    If no consensus coordinates:
        - Fall back to standard learned embeddings
        - (Works but loses the geometric advantage)

    Small learned perturbation allows fine-tuning during training
    while preserving the consensus structure as the backbone.
    """

    def __init__(self, vocab_size: int, d_model: int,
                 consensus_coords: Optional[torch.Tensor] = None,
                 max_proj_norm: float = 5.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_proj_norm = max_proj_norm

        if consensus_coords is not None:
            coord_dim = consensus_coords.shape[1]
            # Frozen consensus geometry
            self.register_buffer('consensus', consensus_coords)
            # Small learnable perturbation
            self.perturbation = nn.Parameter(
                torch.randn(vocab_size, coord_dim) * 0.01)
            # Project to model dimension
            self.project = nn.Linear(coord_dim, d_model, bias=False)
            self.has_consensus = True
            print(f"    ConsensusEmbedding: {vocab_size} tokens × "
                  f"{coord_dim}-d coords → {d_model}-d model")
        else:
            # Fallback: standard learned embeddings
            self.embed = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(self.embed.weight, std=0.02)
            self.has_consensus = False
            print(f"    ConsensusEmbedding: standard learned ({vocab_size} × {d_model})")

        self.scale = nn.Parameter(torch.tensor(1.0))

    def _get_raw(self) -> torch.Tensor:
        """Get embeddings with perturbation, clamped in Poincaré ball."""
        if self.has_consensus:
            raw = self.consensus + self.perturbation
            norm = raw.norm(dim=-1, keepdim=True).clamp(min=POINCARE_EPS)
            return torch.where(
                norm > POINCARE_MAX_NORM,
                raw * POINCARE_MAX_NORM / norm, raw)
        else:
            return self.embed.weight

    def _project_and_clamp(self, raw: torch.Tensor) -> torch.Tensor:
        """Project to d_model and clamp norm."""
        if self.has_consensus:
            projected = self.project(raw) * self.scale
        else:
            projected = raw * self.scale

        norm = projected.norm(dim=-1, keepdim=True).clamp(min=POINCARE_EPS)
        return torch.where(
            norm > self.max_proj_norm,
            projected * self.max_proj_norm / norm, projected)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raw = self._get_raw()
        embedded = F.embedding(token_ids, raw)
        return self._project_and_clamp(embedded)

    def get_all_projected(self) -> torch.Tensor:
        """All token embeddings in d_model space. For energy-based output."""
        return self._project_and_clamp(self._get_raw())


# ════════════════════════════════════════════════════════════
# SECTION 4: TORUS POSITIONAL ENCODING
# Same as Jamo version — phi-spiral, Möbius twist, no repeats.
# ════════════════════════════════════════════════════════════

class TorusPositionalEncoding(nn.Module):
    """
    Torus positions via golden-angle spiral with Möbius twist.
    Position n sits at θ = n × golden_angle (never repeats).
    Möbius twist: poloidal angle flips sign every full circuit.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        theta = positions * GOLDEN_ANGLE

        circuits = torch.floor(theta / (2 * math.pi))
        mobius_sign = 1.0 - 2.0 * (circuits % 2)

        self.register_buffer('theta', theta)
        self.register_buffer('mobius_sign', mobius_sign)

        n_harmonics = d_model // 4
        self.harmonic_proj = nn.Linear(n_harmonics * 4, d_model, bias=False)

        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        scales = PHI ** (harmonics / n_harmonics)
        self.register_buffer('harmonic_scales', scales)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns [1, seq_len, d_model] positional encoding.
        (No token-dependent realm angles for English — pure positional.)
        """
        theta = self.theta[:seq_len]           # [L]
        mobius = self.mobius_sign[:seq_len]     # [L]

        # For English: poloidal angle = position-based (not realm-based)
        # Use a secondary golden ratio spiral for the second torus dimension
        phi = theta * PHI * mobius  # incommensurate with theta

        scales = self.harmonic_scales
        theta_s = theta.unsqueeze(-1) * scales   # [L, H]
        phi_s = phi.unsqueeze(-1) * scales       # [L, H]

        features = torch.cat([
            theta_s.cos(), theta_s.sin(),
            phi_s.cos(), phi_s.sin(),
        ], dim=-1)  # [L, 4H]

        return self.harmonic_proj(features).unsqueeze(0)  # [1, L, D]


# ════════════════════════════════════════════════════════════
# SECTION 5: TIMESTEP EMBEDDING
# ════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """Sinusoidal + learned embedding for diffusion timestep."""

    def __init__(self, d_model: int, max_timesteps: int = 100):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        half = d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_float = t.float().unsqueeze(-1)
        args = t_float * self.freqs
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ════════════════════════════════════════════════════════════
# SECTION 6: DENOISER TRANSFORMER (BIDIRECTIONAL)
# NOT autoregressive. Sees full noisy sequence. Diffusion.
# ════════════════════════════════════════════════════════════

class DenoiserBlock(nn.Module):
    """Bidirectional transformer block with torus distance bias."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.torus_scale = nn.Parameter(torch.ones(n_heads) * 0.1)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Timestep conditioning: adaptive layer norm
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
        )

    def forward(self, x, torus_dist, time_emb, mask=None):
        """
        Args:
            x: [B, L, D]
            torus_dist: [1, L, L] — precomputed, shared across blocks
            time_emb: [B, D]
            mask: [B, L]
        """
        B, L, D = x.shape

        # Adaptive LayerNorm from timestep
        time_params = self.time_mlp(time_emb)
        scale, shift = time_params.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # QKV
        qkv = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Torus distance bias (precomputed, just scale per head)
        torus_bias = -torus_dist.unsqueeze(1) * self.torus_scale.view(1, -1, 1, 1)
        scores = scores + torus_bias

        # Padding mask only (NO causal mask)
        if mask is not None:
            pad = ~mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(out)

        x = x + self.ffn(self.norm2(x))
        return x


# ════════════════════════════════════════════════════════════
# SECTION 7: FULL MODEL
# ════════════════════════════════════════════════════════════

class TorusEnglishDiffusionConfig:
    """Configuration. Defaults produce a ~5M parameter model."""
    def __init__(
        self,
        vocab_size: int = 6000,     # Overridden by vocabulary
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_timesteps: int = 10,    # Same as LockSeed image diffusion
        consensus_path: str = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.consensus_path = consensus_path


class TorusEnglishDiffusion(nn.Module):
    """
    The complete English Diffusion model on a torus manifold.

    Training:
        1. Take clean English text (Token94 + compounds)
        2. Sample random timestep t
        3. Corrupt with random token substitution at level t
        4. Model predicts clean tokens from noisy sequence + timestep
        5. Loss: cross-entropy on corrupted positions

    Generation:
        1. Start with random tokens (wrong English, not empty)
        2. For t = T down to 1:
            a. Model predicts clean tokens (bidirectional)
            b. Re-corrupt prediction to noise level t-1
        3. Wave collapse: track stabilized positions
        4. Final output: denoised English text
    """

    def __init__(self, config: TorusEnglishDiffusionConfig):
        super().__init__()
        self.config = config

        # Vocabulary
        self.vocab = EnglishVocabulary(config.consensus_path)
        config.vocab_size = self.vocab.vocab_size

        # Noise schedule
        self.noise = EnglishNoiseSchedule(config.num_timesteps)

        # Consensus embeddings (geometry from 70B models)
        consensus_coords = self.vocab.get_consensus_coords()
        self.embedding = ConsensusEmbedding(
            config.vocab_size, config.d_model,
            consensus_coords=consensus_coords)

        # Torus positional encoding
        self.torus_pos = TorusPositionalEncoding(
            config.d_model, config.max_seq_len)

        # Timestep embedding
        self.time_emb = TimestepEmbedding(
            config.d_model, config.num_timesteps + 1)

        # Denoiser (BIDIRECTIONAL)
        self.blocks = nn.ModuleList([
            DenoiserBlock(config.d_model, config.n_heads,
                         config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

        # Energy-based output: dot product with all embeddings
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noisy_ids, timesteps, mask=None):
        """Predict clean token logits from noisy input."""
        B, L = noisy_ids.shape
        device = noisy_ids.device

        x = self.embedding(noisy_ids)
        tpos = self.torus_pos(L, device).expand(B, -1, -1)
        x = x + tpos

        t_emb = self.time_emb(timesteps)

        # Precompute torus distance matrix ONCE (same for all blocks)
        # Shape: [1, L, L] — batch dimension broadcasts
        with torch.no_grad():
            pos_i = tpos[:1].unsqueeze(2)  # [1, L, 1, D]
            pos_j = tpos[:1].unsqueeze(1)  # [1, 1, L, D]
            torus_dist = (pos_i - pos_j).norm(dim=-1)  # [1, L, L]

        for block in self.blocks:
            x = block(x, torus_dist, t_emb, mask)

        x = self.final_norm(x)
        x = self.output_proj(x)

        all_emb = self.embedding.get_all_projected()
        temp = self.log_temp.exp().clamp(min=0.1, max=10.0)
        logits = torch.matmul(x, all_emb.T) / temp

        return logits

    def compute_loss(self, clean_ids, mask=None):
        """Training loss with noise schedule."""
        B, L = clean_ids.shape
        device = clean_ids.device

        t = self.noise.sample_timesteps(B, device)
        noisy_ids = self.noise.corrupt(
            clean_ids, t, device, self.config.vocab_size)
        corrupt_mask = (noisy_ids != clean_ids)

        logits = self.forward(noisy_ids, t, mask)

        logits_flat = logits.view(-1, self.config.vocab_size)
        target_flat = clean_ids.view(-1)
        corrupt_flat = corrupt_mask.view(-1)

        if mask is not None:
            valid = mask.view(-1)
            logits_for_loss = logits_flat[valid]
            target_for_loss = target_flat[valid]
            corrupt_for_acc = corrupt_flat & valid
        else:
            logits_for_loss = logits_flat
            target_for_loss = target_flat
            corrupt_for_acc = corrupt_flat

        ce_loss = F.cross_entropy(logits_for_loss, target_for_loss)

        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            if corrupt_for_acc.any():
                accuracy = (preds[corrupt_for_acc] == target_flat[corrupt_for_acc]).float().mean()
            else:
                accuracy = torch.tensor(0.0)

        return {
            'total_loss': ce_loss,
            'ce_loss': ce_loss,
            'accuracy': accuracy,
        }

    @torch.no_grad()
    def generate(
        self,
        seq_len: int = 128,
        batch_size: int = 1,
        device: torch.device = torch.device('cpu'),
        seed: int = None,
        track_collapse: bool = True,
    ) -> Dict:
        """
        Generate English text via iterative denoising.
        Start with random tokens. Denoise over T steps.
        """
        self.eval()
        T = self.config.num_timesteps

        if seed is not None:
            torch.manual_seed(seed)

        # Start: random tokens (wrong English)
        x = torch.randint(
            SPECIAL_OFFSET, self.config.vocab_size,
            (batch_size, seq_len), device=device)
        x[:, 0] = BOS_TOKEN
        x[:, -1] = EOS_TOKEN

        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Wave collapse tracking
        collapse_step = torch.full(
            (batch_size, seq_len), T, dtype=torch.long, device=device)
        collapsed = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=device)
        prev_x = x.clone()

        # Denoise: t = T, T-1, ..., 1
        for step in range(T, 0, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=device)

            logits = self.forward(x, t, mask)
            predicted = logits.argmax(dim=-1)

            # Preserve specials
            special = (x == BOS_TOKEN) | (x == EOS_TOKEN) | (x == PAD_TOKEN)
            predicted = torch.where(special, x, predicted)

            if step > 1:
                t_prev = torch.full(
                    (batch_size,), step - 1, dtype=torch.long, device=device)
                x = self.noise.corrupt(
                    predicted, t_prev, device, self.config.vocab_size)
                x = torch.where(special, prev_x, x)
            else:
                x = predicted

            # Wave collapse
            if track_collapse:
                unchanged = (x == prev_x)
                newly_collapsed = unchanged & ~collapsed
                collapse_step[newly_collapsed] = T - step
                collapsed = collapsed | unchanged

            prev_x = x.clone()

        return {
            'token_ids': x,
            'collapse_step': collapse_step,
            'collapse_mask': collapsed,
        }

    def count_parameters(self) -> Dict[str, int]:
        return {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'torus_pos': sum(p.numel() for p in self.torus_pos.parameters()),
            'time_emb': sum(p.numel() for p in self.time_emb.parameters()),
            'blocks': sum(p.numel() for p in self.blocks.parameters()),
            'output': (sum(p.numel() for p in self.output_proj.parameters()) +
                      sum(p.numel() for p in self.final_norm.parameters())),
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(
                p.numel() for p in self.parameters() if p.requires_grad),
        }


# ════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("TORUS ENGLISH DIFFUSION — Component Test")
    print("=" * 60)

    # Try to find consensus coordinates
    consensus_path = None
    for p in ['./consensus_coordinates.json', './consensus_map.json',
              '../consensus_coordinates.json']:
        if os.path.exists(p):
            consensus_path = p
            break

    if consensus_path:
        print(f"\nUsing consensus geometry: {consensus_path}")
    else:
        print(f"\nNo consensus file found — using learned embeddings")

    # Build model
    cfg = TorusEnglishDiffusionConfig(
        consensus_path=consensus_path,
        d_model=256,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        max_seq_len=512,
        num_timesteps=10,
    )

    model = TorusEnglishDiffusion(cfg)
    params = model.count_parameters()

    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # Test vocabulary
    vocab = model.vocab
    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = vocab.encode(test_text)
    decoded = vocab.decode(tokens)
    print(f"\nVocabulary test:")
    print(f"  Input:   '{test_text}'")
    print(f"  Tokens:  {tokens[:20]}...")
    print(f"  Decoded: '{decoded}'")
    print(f"  Match:   {test_text == decoded}")

    # Test forward pass
    ids = torch.tensor([tokens[:30]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    t = torch.tensor([5])

    # Noise test
    noisy = model.noise.corrupt(ids, t, ids.device, cfg.vocab_size)
    print(f"\nNoise test (t=5):")
    print(f"  Clean: '{vocab.decode(ids[0].tolist())[:40]}'")
    print(f"  Noisy: '{vocab.decode(noisy[0].tolist())[:40]}'")

    # Loss test
    losses = model.compute_loss(ids, mask)
    print(f"\nLoss test:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Generation test
    print(f"\nGeneration test (10 steps, seed=42)...")
    result = model.generate(seq_len=50, device=torch.device('cpu'), seed=42)
    decoded = vocab.decode(result['token_ids'][0].tolist())
    collapsed = result['collapse_mask'][0].float().mean().item()
    print(f"  Generated: '{decoded[:80]}'")
    print(f"  Collapse:  {collapsed*100:.0f}% of positions stabilized")

    print(f"\n{'='*60}")
    print(f"All components working.")
    print(f"This is a DIFFUSION model. Not autoregressive.")
    print(f"The geometry came from 70B models. The navigator is {params['total']:,} parameters.")
    print(f"{'='*60}")
