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
/// ⚠️ CIDs / model_id below are PLACEHOLDERS. Before release: pin each GGUF to
/// IPFS, then set tokenizer_cid + weight_cids and recompute
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

/// Placeholder model_id — all-zero until the real GGUF is pinned to IPFS.
/// Replace with base58-decode(weight CID)[2..34].
const PLACEHOLDER_MODEL_ID: [u8; 32] = [0u8; 32];

pub const GEMMA_3_4B: ModelSpec = ModelSpec {
    name: "gemma-3-4b",
    // TODO(release): pin Gemma-3-4B-it-abliterated GGUF, set model_id = CIDv0[2..34].
    model_id: PLACEHOLDER_MODEL_ID,
    format: ModelFormat::GgufGemma3,
    tokenizer_cid: "TODO_PIN_GEMMA_3_4B_TOKENIZER_CID",
    config_cid: "",
    weight_cids: &["TODO_PIN_GEMMA_3_4B_WEIGHT_CID"],
    dir_name: "Gemma-3-4B",
    // Baseline model — never gated. ~4 GB Q4_K_M; GPUs too small must use --cpu-inference.
    min_vram_mb: 0,
};

pub const DOLPHIN_LLAMA3_8B: ModelSpec = ModelSpec {
    name: "dolphin-llama3-8b",
    // TODO(release): pin Dolphin-3.0-Llama-3.1-8B GGUF, set model_id = CIDv0[2..34].
    model_id: PLACEHOLDER_MODEL_ID,
    format: ModelFormat::Gguf,
    tokenizer_cid: "TODO_PIN_DOLPHIN_8B_TOKENIZER_CID",
    config_cid: "",
    weight_cids: &["TODO_PIN_DOLPHIN_8B_WEIGHT_CID"],
    dir_name: "Dolphin-Llama3-8B",
    // ~4.9 GB Q4_K_M weights + ~1.6 GB KV/workspace.
    min_vram_mb: 8_000,
};

pub const QWEN3_32B: ModelSpec = ModelSpec {
    name: "qwen3-32b",
    // TODO(release): pin Qwen3-32B-abliterated Q4_K_M GGUF, set model_id = CIDv0[2..34].
    model_id: PLACEHOLDER_MODEL_ID,
    format: ModelFormat::GgufQwen3,
    tokenizer_cid: "TODO_PIN_QWEN3_32B_TOKENIZER_CID",
    config_cid: "",
    weight_cids: &["TODO_PIN_QWEN3_32B_WEIGHT_CID"],
    dir_name: "Qwen3-32B",
    // ~19.5 GB Q4_K_M weights + ~2.5 GB KV/workspace → fits a 24 GB card (3090/4090/5090).
    min_vram_mb: 24_000,
};

pub const LLAMA_3_3_70B: ModelSpec = ModelSpec {
    name: "llama-3.3-70b",
    // TODO(release): pin Llama-3.3-70B-Instruct-abliterated Q4_K_M GGUF, set model_id = CIDv0[2..34].
    model_id: PLACEHOLDER_MODEL_ID,
    format: ModelFormat::Gguf,
    tokenizer_cid: "TODO_PIN_LLAMA_70B_TOKENIZER_CID",
    config_cid: "",
    weight_cids: &["TODO_PIN_LLAMA_70B_WEIGHT_CID"],
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
