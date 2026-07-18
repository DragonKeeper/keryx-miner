/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.
///
/// Uncensored five-tier lineup, active at `COIN_AGE_VERIFICATION_ACTIVATION_DAA` (the H4
/// hardfork) — below that DAA this binary refuses to mine (`pom_tier_index` = None). Every
/// model is untied so the in-process llama engine hosts walk + inference in one resident copy:
///   --very-light  EXAONE-4.0-1.2B  Q4_K_M (LG)       — 2 GB+
///   --light       Mistral-7B-v0.3  Q6_K   (Mistral)  — 8 GB
///   (default)     GLM-4-9B-0414    Q6_K   (Zhipu)    — 12 GB
///   --high        Qwen3.6-27B      Q4_K_M (Alibaba)  — 24 GB
///   --very-high   Kimi-Linear-48B  Q4_K_M (Moonshot) — 32 GB
///
/// All GGUF weights are pinned on the Keryx IPFS gateway; each
/// model_id = base58-decode(weight CID)[2..34].

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// GGUF quantized — LLaMA architecture (Mistral-7B). llama-served.
    Gguf,
    /// GGUF quantized — EXAONE 4 architecture (H4 tier 0). llama-served.
    GgufExaone4,
    /// GGUF quantized — GLM 4 architecture (H4 tier 2). llama-served.
    GgufGlm4,
    /// GGUF quantized — Qwen3.5 hybrid-SSM architecture (H4 tier 3). llama-served.
    GgufQwen35,
    /// GGUF quantized — Kimi-Linear MoE architecture (H4 tier 4). llama-served.
    GgufKimiLinear,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// 32-byte on-chain identifier embedded in AiRequest payloads.
    pub model_id: [u8; 32],
    pub format: ModelFormat,
    /// Empty for the whole lineup: llama uses the tokenizer embedded in the GGUF.
    pub tokenizer_cid: &'static str,
    /// Single entry: the model.gguf CID.
    pub weight_cids: &'static [&'static str],
    /// Local directory name under `<exe_dir>/models/`.
    pub dir_name: &'static str,
    /// Minimum VRAM (MB) required to actually serve this model: weights +
    /// KV cache + CUDA workspace. Used by the OPoI capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
}

// ── H4 lineup ───────────────────────────────────────────────────
// Active at `crate::pom::COIN_AGE_VERIFICATION_ACTIVATION_DAA` (the H4 hardfork). Every model is
// UNTIED so the in-process llama engine hosts walk + inference in one resident copy;
// `libkeryx-llama.so` is REQUIRED to serve them.
// `tokenizer_cid` is empty: llama uses the tokenizer embedded in the GGUF, no separate file.
// model_id bytes MUST equal the node's `params.rs` H4 constants (CIDv0[2..34] of the pinned GGUF).

pub const EXAONE_4_0_1_2B: ModelSpec = ModelSpec {
    name: "exaone-4.0-1.2b",
    // CIDv0[2..34] of model.gguf — EXAONE-4.0-1.2B-abliterated Q4_K_M (mradermacher, i1)
    model_id: [
        0x30, 0x0a, 0x99, 0xb3, 0xa8, 0x5b, 0x0a, 0xb4,
        0x5d, 0x1d, 0x93, 0x0b, 0xb7, 0xb1, 0xd4, 0xb0,
        0xf3, 0x59, 0x83, 0xd5, 0x21, 0xe7, 0x9f, 0xf2,
        0x11, 0x93, 0xa6, 0x90, 0x8d, 0xc4, 0xb8, 0x10,
    ],
    format: ModelFormat::GgufExaone4,
    tokenizer_cid: "",
    weight_cids: &["QmRaBetZg8SaeWeGrQDBhRMd362mf4Nm3w2YacCSQ8tocb"],
    dir_name: "EXAONE-4.0-1.2B",
    // ~0.9 GB Q4_K_M — smallest tier, runs on 4-6 GB GPUs. Never gated.
    min_vram_mb: 0,
};

pub const MISTRAL_7B_V03: ModelSpec = ModelSpec {
    name: "mistral-7b-v0.3",
    // CIDv0[2..34] of model.gguf — Mistral-7B-Instruct-v0.3-abliterated Q6_K (mradermacher)
    model_id: [
        0x8c, 0x2f, 0xea, 0x60, 0x0f, 0x0e, 0xef, 0xe7,
        0x04, 0x87, 0x41, 0xa5, 0x11, 0x9c, 0xb7, 0xbe,
        0x30, 0x30, 0x37, 0xf5, 0x9f, 0xc0, 0x26, 0xe4,
        0x83, 0x82, 0x65, 0x8f, 0x23, 0x58, 0x1e, 0x0a,
    ],
    // llama architecture, but llama-served like the rest of the H4 lineup (no pinned tokenizer.json).
    format: ModelFormat::Gguf,
    tokenizer_cid: "",
    weight_cids: &["QmXmtATpJerCCcWWF515vAe5FanSvqJrD4L1ogZxDurQ3s"],
    dir_name: "Mistral-7B-v0.3",
    // ~5.9 GB Q6_K weights + ~1.5 GB KV/workspace. The Q6_K quant is deliberate VRAM gating:
    // it does NOT fit a 6 GB card, so 6 GB stays on tier 0 (EXAONE) and 8 GB serves tier 1.
    min_vram_mb: 8_000,
};

pub const GLM_4_9B_0414: ModelSpec = ModelSpec {
    name: "glm-4-9b-0414",
    // CIDv0[2..34] of model.gguf — GLM-4-9B-0414-abliterated Q6_K
    model_id: [
        0xfa, 0x2f, 0x13, 0xbe, 0x08, 0x50, 0xe2, 0x6c,
        0x5c, 0xe8, 0x6c, 0x7a, 0xc7, 0x9d, 0xa8, 0x5e,
        0x30, 0x0c, 0x1d, 0xa8, 0xb3, 0x29, 0x0f, 0x9a,
        0x18, 0xd4, 0x71, 0x05, 0xf1, 0xf2, 0x14, 0x0a,
    ],
    format: ModelFormat::GgufGlm4,
    tokenizer_cid: "",
    weight_cids: &["QmfBGGZumBR4XGFLLPjYozvhRSt3kXjrgsV3jXciCdAeM7"],
    dir_name: "GLM-4-9B-0414",
    // ~8.3 GB Q6_K weights + ~1.5 GB KV/workspace → 12 GB card (3060 12GB / 3080 12GB).
    min_vram_mb: 12_000,
};

pub const QWEN3_6_27B: ModelSpec = ModelSpec {
    name: "qwen3.6-27b",
    // CIDv0[2..34] of model.gguf — Qwen3.6-27B-abliterated-v2 Q4_K_M (mradermacher)
    model_id: [
        0xb8, 0xbd, 0xc0, 0x1f, 0xa4, 0x07, 0xea, 0xb9,
        0x43, 0xe4, 0xfe, 0xfc, 0x80, 0x74, 0x83, 0xb3,
        0x9f, 0x81, 0x42, 0x78, 0x52, 0x56, 0x04, 0x9e,
        0x1f, 0x55, 0x96, 0x98, 0xa5, 0x28, 0x47, 0x46,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["QmamoYQGGAkBaqiWuNmwxeC9AQnt9F7sLyX57VoqbJWeUV"],
    dir_name: "Qwen3.6-27B",
    // ~16.5 GB Q4_K_M weights + ~2.5 GB KV/workspace → 24 GB card (3090/4090/5090).
    min_vram_mb: 24_000,
};

pub const KIMI_LINEAR_48B: ModelSpec = ModelSpec {
    name: "kimi-linear-48b",
    // CIDv0[2..34] of model.gguf — Kimi-Linear-48B-A3B-Instruct-abliterated Q4_K_M (mradermacher, i1)
    model_id: [
        0x3d, 0xc0, 0x93, 0x58, 0xad, 0x75, 0xc6, 0xef,
        0x0c, 0x9c, 0x86, 0xee, 0x4f, 0x47, 0xc4, 0xd6,
        0xac, 0xda, 0x96, 0x1f, 0xec, 0xbd, 0x0e, 0x4f,
        0x9c, 0xf5, 0x5e, 0x8f, 0x0f, 0xdf, 0xfd, 0xdb,
    ],
    format: ModelFormat::GgufKimiLinear,
    tokenizer_cid: "",
    weight_cids: &["QmSVhtoNrL8bWJXZuEXMMWqty8qHScQMRuacuoa9ujsYqp"],
    dir_name: "Kimi-Linear-48B",
    // ~29.7 GB Q4_K_M weights (MoE, 3B active) + KV/workspace → needs a 32 GB card (5090),
    // so the top tier stays 5090-class.
    min_vram_mb: 30_000,
};

/// Whether `model_id` is one of the Proof-of-Model tier models (any era). DAA-independent —
/// used at startup to pick a mineable PoM model before any block DAA is known (the tier *index*
/// is then computed per block via `pom_tier_index`).
pub fn is_pom_model(model_id: &[u8; 32]) -> bool {
    *model_id == EXAONE_4_0_1_2B.model_id
        || *model_id == MISTRAL_7B_V03.model_id
        || *model_id == GLM_4_9B_0414.model_id
        || *model_id == QWEN3_6_27B.model_id
        || *model_id == KIMI_LINEAR_48B.model_id
}

pub fn pom_tier_index(model_id: &[u8; 32], daa: u64) -> Option<u8> {
    // H4 gate: below the flip this binary refuses to mine (None) — it never produces a
    // pre-H4-era block. MUST mirror the node's `POM_TIERS_H4` order, recomputed per block
    // from that block's DAA.
    if daa < crate::pom::COIN_AGE_VERIFICATION_ACTIVATION_DAA {
        return None;
    }
    if *model_id == EXAONE_4_0_1_2B.model_id {
        Some(0)
    } else if *model_id == MISTRAL_7B_V03.model_id {
        Some(1)
    } else if *model_id == GLM_4_9B_0414.model_id {
        Some(2)
    } else if *model_id == QWEN3_6_27B.model_id {
        Some(3)
    } else if *model_id == KIMI_LINEAR_48B.model_id {
        Some(4)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    VeryLight,
    Light,
    Default,
    High,
    VeryHigh,
}

/// The single model a hardware tier mines AND serves — one flag = one model. A PoM GPU is
/// bound to its tier (serving a lower tier would mean unloading the mined model and pausing
/// mining); multi-tier coverage is a network property (different miners per tier), not a
/// per-GPU one.
pub fn spec_for_tier(tier: Tier) -> &'static ModelSpec {
    match tier {
        Tier::VeryLight => &EXAONE_4_0_1_2B,
        Tier::Light => &MISTRAL_7B_V03,
        Tier::Default => &GLM_4_9B_0414,
        Tier::High => &QWEN3_6_27B,
        Tier::VeryHigh => &KIMI_LINEAR_48B,
    }
}

/// Resolves a model name/id.
pub const REGISTRY: &[&ModelSpec] = &[
    &EXAONE_4_0_1_2B,
    &MISTRAL_7B_V03,
    &GLM_4_9B_0414,
    &QWEN3_6_27B,
    &KIMI_LINEAR_48B,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}
