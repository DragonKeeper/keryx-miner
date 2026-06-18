/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.
///
/// Uncensored lineup (4 tiers / 4 model families):
///   --light       Gemma-3-4B-it-abliterated     (Google)  — any GPU (6 GB+)
///   (default)     Dolphin-3.0-Llama-3.1-8B       (Llama)  — RTX 3060 12GB / 3070
///   --high        Qwen3-32B-abliterated (Q4_K_M) (Qwen)   — 24 GB (3090 / 4090 / 5090)
///   --very-high   Llama-3.3-70B-abliterated      (Meta)   — 48 GB single-GPU or --vram-pool
///
/// All GGUF weights + tokenizers are pinned on the Keryx IPFS gateway; each
/// model_id = base58-decode(weight CID)[2..34].

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Full-precision safetensors (one or more shards).
    Safetensors,
    /// GGUF quantized — LLaMA/LLaMA3 architecture.
    Gguf,
    /// GGUF quantized — Qwen3 architecture (Qwen3-32B).
    GgufQwen3,
    /// GGUF quantized — Gemma 3 architecture (Gemma-3-4B, baseline tier).
    GgufGemma3,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// 32-byte on-chain identifier embedded in AiRequest payloads.
    pub model_id: [u8; 32],
    pub format: ModelFormat,
    pub tokenizer_cid: &'static str,
    /// Unused for GGUF (architecture embedded in file).
    pub config_cid: &'static str,
    /// Safetensors: one entry per shard. GGUF: single entry.
    pub weight_cids: &'static [&'static str],
    /// Local directory name under `<exe_dir>/models/`.
    pub dir_name: &'static str,
    /// Minimum VRAM (MB) required to actually serve this model: weights +
    /// KV cache + CUDA workspace. Used by the OPoI capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
}

pub const GEMMA_3_4B: ModelSpec = ModelSpec {
    name: "gemma-3-4b",
    // CIDv0[2..34] of model.gguf — mlabonne/gemma-3-4b-it-abliterated Q4_K_M
    model_id: [
        0xad, 0x50, 0xad, 0x0b, 0xd4, 0x61, 0xd8, 0xab,
        0x44, 0xef, 0xc0, 0x21, 0x49, 0x89, 0xeb, 0x33,
        0x29, 0x16, 0x85, 0xef, 0x4a, 0xde, 0x22, 0xa0,
        0xf4, 0xf2, 0x17, 0xd0, 0x32, 0x66, 0xd8, 0x37,
    ],
    format: ModelFormat::GgufGemma3,
    tokenizer_cid: "QmTh2MsVfAvWp7grN9rvkF9NkMkCW2PhWez2WbNh81KRXD",
    config_cid: "",
    weight_cids: &["Qma1CbFzWTNhy2ReVjDG1GvM5q2Uy4VhqTbnS9c641jUQ6"],
    dir_name: "Gemma-3-4B",
    // Baseline model — never gated. ~4 GB Q4_K_M; GPUs too small must use --cpu-inference.
    min_vram_mb: 0,
};

pub const DOLPHIN_LLAMA3_8B: ModelSpec = ModelSpec {
    name: "dolphin-llama3-8b",
    // CIDv0[2..34] of model.gguf — Dolphin3.0-Llama3.1-8B Q4_K_M
    model_id: [
        0x94, 0x21, 0x06, 0x6a, 0x64, 0x00, 0xc9, 0x8b,
        0xa1, 0x37, 0x11, 0x4f, 0x7f, 0x4b, 0x7d, 0x4a,
        0x2d, 0xdf, 0x13, 0xab, 0x16, 0x3a, 0x5d, 0xe3,
        0x8c, 0x01, 0x84, 0x79, 0x3a, 0xf6, 0x31, 0x3a,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmQSe8rZQcTQ6q1xDGquv6s9wpzFT9u27U4wfGVZqwMJgJ",
    config_cid: "",
    weight_cids: &["QmYJtFpaDnVwAVSbzRo42fsb19nLpt8LHe8WVKoyxd4AkZ"],
    dir_name: "Dolphin-Llama3-8B",
    // ~4.9 GB Q4_K_M weights + ~1.6 GB KV/workspace.
    min_vram_mb: 8_000,
};

pub const QWEN3_32B: ModelSpec = ModelSpec {
    name: "qwen3-32b",
    // CIDv0[2..34] of model.gguf — Qwen3-32B-abliterated Q4_K_M (mradermacher)
    model_id: [
        0x65, 0xc6, 0xeb, 0x6f, 0xe1, 0x8b, 0x9e, 0xfd,
        0x80, 0x60, 0xab, 0x9d, 0x2d, 0x03, 0xbb, 0x9b,
        0x01, 0x05, 0x0a, 0x3b, 0x13, 0x78, 0xcb, 0xac,
        0x00, 0x0c, 0x5c, 0xc0, 0xac, 0xdc, 0x0d, 0x2a,
    ],
    format: ModelFormat::GgufQwen3,
    tokenizer_cid: "QmcuGkJvR343ry3b4jy7u5L9ior3ujas3yGAFMSyZdACb5",
    config_cid: "",
    weight_cids: &["QmVBwp5n3muQJwYNLTHSu3EnzBWviQqfh58FvHvKRfLtam"],
    dir_name: "Qwen3-32B",
    // ~19.5 GB Q4_K_M weights + ~2.5 GB KV/workspace → fits a 24 GB card (3090/4090/5090).
    min_vram_mb: 24_000,
};

pub const LLAMA_3_3_70B: ModelSpec = ModelSpec {
    name: "llama-3.3-70b",
    // CIDv0[2..34] of model.gguf — Llama-3.3-70B-Instruct-abliterated Q4_K_M (bartowski)
    model_id: [
        0x13, 0x29, 0xfb, 0xe2, 0x1b, 0x3f, 0x36, 0xf6,
        0xd0, 0x06, 0x89, 0xfc, 0xaa, 0x74, 0xf7, 0xa2,
        0x22, 0xb8, 0xcc, 0x4c, 0x08, 0xc0, 0x19, 0x1f,
        0xeb, 0x23, 0x97, 0x55, 0xa7, 0x23, 0x42, 0x1e,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmPd7WQvoQupfzpPVnVVc1Zra5SH4jKnGqNrdTHFtdQuvd",
    config_cid: "",
    weight_cids: &["QmPdTayXcEsfUwMCoMKKcLSv7Dwpp2xVBWELwrG2M7Rhzu"],
    dir_name: "Llama-3.3-70B",
    // ~42.5 GB Q4_K_M weights + ~3.5 GB KV/workspace → 48 GB card or --vram-pool (matches the
    // --very-high 46 GB startup gate).
    min_vram_mb: 46_000,
};

pub const REGISTRY: &[&ModelSpec] =
    &[&GEMMA_3_4B, &DOLPHIN_LLAMA3_8B, &QWEN3_32B, &LLAMA_3_3_70B];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}
