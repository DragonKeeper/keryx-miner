//! Inference trace of a sealed response: commitment, draw and opening. Byte-exact mirror of the
//! node's `collateral.rs` trace rules on top of the shared `keryx-verifiable-inference` crate.

use keryx_inference::TraceCommitment;
use keryx_verifiable_inference::chain::ChainCtx;
use keryx_verifiable_inference::draw;
use keryx_verifiable_inference::manifest::{self, Entry};
use borsh::{BorshDeserialize, BorshSerialize};
use keryx_verifiable_inference::merkle::{b3_chunk_cv, b3_path, b3_root, rt_range_proof, Hash, B3_CHUNK};
use keryx_verifiable_inference::opening::{self, ChunkRange, EntryOpening, Opening, Point, PrefixPoint};
use keryx_verifiable_inference::prefix::{self, PrefixCtx};

use crate::pom::WeightIndex;

pub const TRACE_DRAW_DOMAIN: &[u8] = b"KeryxTraceDrawV1";
pub const OPENING_SIG_DOMAIN: &[u8] = b"KeryxTraceOpeningV1";
/// DAA between a response's acceptance and the chain block that seeds its draw.
pub const TRACE_SEED_DELAY_DAA: u64 = 100;
/// Stage committed by this miner: products only (1a).
pub const STAGE_1A: u8 = 1;
/// Points opened per audited response.
pub const POINTS: u32 = draw::POINTS;

/// Serialized manifests of the audited tiers, in tier order; a missing tier is not audited.
const MANIFESTS: [Option<&[u8]>; 5] = [
    Some(include_bytes!("../assets/trace-manifests/tier0.bin")),
    Some(include_bytes!("../assets/trace-manifests/tier1.bin")),
    Some(include_bytes!("../assets/trace-manifests/tier2.bin")),
    Some(include_bytes!("../assets/trace-manifests/tier3.bin")),
    None,
];

/// Prefix contexts of the audited tiers, in tier order. MUST equal the node's `TRACE_PREFIXES`.
const PREFIXES: [Option<PrefixCtx>; 5] = [
    Some(PrefixCtx {
        table: [
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
            254, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 255, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
            172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
        ],
        offset: 6,
        embd_entry: 249,
        norm_entry: 250,
        eps: 9.999999974752427e-07,
        embd_scale: 1.0,
    }),
    Some(PrefixCtx {
        table: [
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
            254, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 255, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
            172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
        ],
        offset: 7,
        embd_entry: 361,
        norm_entry: 362,
        eps: 9.999999747378752e-06,
        embd_scale: 1.0,
    }),
    Some(PrefixCtx {
        table: [
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 107, 249, 250, 251, 252, 253,
            254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
            270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
            286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
            302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
            318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
            334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
            350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365,
            366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381,
            382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
            398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
            414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
            430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445,
            446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
            462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
            478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
        ],
        offset: 7,
        embd_entry: 329,
        norm_entry: 330,
        eps: 9.999999974752427e-07,
        embd_scale: 61.967735290527344,
    }),
    Some(PrefixCtx {
        table: [
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
            238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
            254, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 255, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
            172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
        ],
        offset: 6,
        embd_entry: 497,
        norm_entry: 498,
        eps: 9.999999974752427e-07,
        embd_scale: 1.0,
    }),
    None,
];

/// Stage committed once `trace_chain_activation` is live: products, chain leaves and links.
pub const STAGE_CHAIN: u8 = keryx_verifiable_inference::chain::STAGE_CHAIN;

/// Serialized stage-2 manifests (`KXMANIF2`), in tier order. MUST equal the node's
/// `TRACE_CHAINS_H15` manifests.
const CHAIN_MANIFESTS: [&[u8]; 5] = [
    include_bytes!("../assets/trace-manifests/chain-tier0.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier1.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier2.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier3.bin"),
    include_bytes!("../assets/trace-manifests/chain-tier4.bin"),
];

/// Stage-2 chain context of each tier, with the prefix embedding and norm entries of its
/// manifest. MUST equal the node's `TRACE_CHAINS_H15`.
const CHAIN_CTX: [Option<(ChainCtx, u32, u32)>; 5] = [
    Some((ChainCtx { n_layers: 32, layer_base: 315, embd_entry: 249, eps: 9.999999974752427e-07, embd_scale: 1.0, trim_last: false, n_used: 0 }, 249, 250)),
    Some((ChainCtx { n_layers: 40, layer_base: 523, embd_entry: 361, eps: 9.999999747378752e-06, embd_scale: 1.0, trim_last: true, n_used: 0 }, 361, 362)),
    Some((ChainCtx { n_layers: 48, layer_base: 523, embd_entry: 329, eps: 9.999999974752427e-07, embd_scale: 61.967735290527344, trim_last: false, n_used: 0 }, 329, 330)),
    Some((ChainCtx { n_layers: 64, layer_base: 627, embd_entry: 497, eps: 9.999999974752427e-07, embd_scale: 1.0, trim_last: false, n_used: 0 }, 497, 498)),
    Some((ChainCtx { n_layers: 27, layer_base: 503, embd_entry: 395, eps: 9.999999747378752e-06, embd_scale: 1.0, trim_last: true, n_used: 8 }, 395, 396)),
];

/// Entries of the stage-2 manifest of `tier` and its product count.
pub fn chain_manifest(tier: u8) -> Option<keryx_verifiable_inference::manifest::Manifest> {
    keryx_verifiable_inference::manifest::read_file(CHAIN_MANIFESTS.get(tier as usize)?)
}

/// Stage-2 chain context of `tier`, and its prefix context on the stage-2 manifest.
pub fn chain_ctx(tier: u8) -> Option<(ChainCtx, PrefixCtx)> {
    let (ctx, embd, norm) = (*CHAIN_CTX.get(tier as usize)?)?;
    let prefix = PREFIXES.get(tier as usize)?.clone().map(|p| PrefixCtx { embd_entry: embd, norm_entry: norm, ..p });
    let prefix = match (tier, prefix) {
        (_, Some(p)) => p,
        (4, None) => PrefixCtx { table: PREFIXES[0].clone()?.table, offset: KIMI_PREFIX_OFFSET, embd_entry: embd, norm_entry: norm, eps: ctx.eps, embd_scale: ctx.embd_scale },
        _ => return None,
    };
    Some((ctx, prefix))
}

/// Tokens of the Kimi chat-template head before the request prefix (`<|im_system|>system<|im_middle|>Request id:`).
pub const KIMI_PREFIX_OFFSET: u32 = 6;


/// The prefix context of `tier`, `None` when the tier is not audited.
pub fn prefix_ctx(tier: u8) -> Option<&'static PrefixCtx> {
    PREFIXES.get(tier as usize).and_then(|p| p.as_ref())
}

/// Whether a response accepted at `accepted_daa` is opened under the prefix rule.
pub fn prefix_active(accepted_daa: u64) -> bool {
    accepted_daa >= crate::pom::trace_prefix_activation_daa()
}

/// Whether every traced response accepted at `accepted_daa` is audited.
pub fn audit_all_active(accepted_daa: u64) -> bool {
    accepted_daa >= crate::pom::trace_audit_activation_daa()
}

/// The token ids the request id puts at the prefix positions of a traced prompt.
pub fn prefix_tokens(ctx: &PrefixCtx, request_hash: &[u8; 32]) -> Vec<i32> {
    request_hash.iter().map(|b| ctx.table[*b as usize] as i32).collect()
}

/// Manifest of an audited tier: the canonical product list and its root `M_T`.
pub struct TierManifest {
    pub entries: Vec<(Entry, Vec<u8>)>,
    pub n_products: u32,
    leaves: Vec<Hash>,
    pub m_t: Hash,
}

impl TierManifest {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let (entries, n_products) = manifest::read_file(data)?;
        if n_products == 0 || n_products as usize > entries.len() {
            return None;
        }
        let leaves: Vec<Hash> = entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
        let (m_t, _) = rt_range_proof(&leaves, 0, 1);
        Some(Self { entries, n_products, leaves, m_t })
    }

    /// The embedded manifest of `tier`, `None` when the tier is not audited.
    pub fn for_tier(tier: u8) -> Option<Self> {
        MANIFESTS.get(tier as usize).copied().flatten().and_then(Self::parse)
    }

    /// The manifest the node holds for a response accepted at `accepted_daa`: the embedded one
    /// under the prefix rule, else its entries before the embedding and norm.
    pub fn for_tier_at(tier: u8, accepted_daa: u64) -> Option<Self> {
        let m = Self::for_tier(tier)?;
        if prefix_active(accepted_daa) { Some(m) } else { Some(m.legacy()) }
    }

    /// This manifest without the prefix entries (`M_T` of the node's `TRACE_MANIFESTS_H14`).
    pub fn legacy(&self) -> Self {
        let n = self.entries.iter().position(|(e, _)| e.role == manifest::ROLE_EMBD).unwrap_or(self.entries.len());
        let entries: Vec<(Entry, Vec<u8>)> = self.entries[..n].to_vec();
        let leaves: Vec<Hash> = self.leaves[..n].to_vec();
        let (m_t, _) = rt_range_proof(&leaves, 0, 1);
        Self { entries, n_products: self.n_products, leaves, m_t }
    }

    pub fn raw_for_tier(tier: u8) -> Option<&'static [u8]> {
        MANIFESTS.get(tier as usize).copied().flatten()
    }

    /// Leaves of one position: `X(r)` at `2r`, `Y(r)` at `2r + 1`, then the meta leaf.
    pub fn n_leaves(&self) -> usize {
        2 * self.n_products as usize + 1
    }

    pub fn entry(&self, index: u64) -> &Entry {
        &self.entries[index as usize].0
    }

    pub fn entry_opening(&self, index: u64) -> EntryOpening {
        EntryOpening {
            index: index as u32,
            bytes: self.entries[index as usize].1.clone(),
            path: rt_range_proof(&self.leaves, index, 1).1,
        }
    }
}

/// Trace of one response: the leaves the engine wrote, their position roots and `trace_root`.
#[derive(BorshSerialize, BorshDeserialize)]
pub struct TraceRecord {
    pub n_prompt: u32,
    pub n_pos: u32,
    pub ubatch: Vec<u32>,
    n_leaves: usize,
    leaves: Vec<Hash>,
    pos_roots: Vec<Hash>,
    pub trace_root: Hash,
}

fn meta_leaf(t: u32, n_ubatch: u32) -> Hash {
    let mut meta = [0u8; 8];
    meta[..4].copy_from_slice(&t.to_le_bytes());
    meta[4..].copy_from_slice(&n_ubatch.to_le_bytes());
    *blake3::hash(&meta).as_bytes()
}

impl TraceRecord {
    /// Leaves as written by the engine, position-major (`n_pos × n_leaves`). The fill-ins of
    /// leaves the engine never writes are applied here; any other leaf left unwritten is an error.
    pub fn new(m: &TierManifest, n_prompt: u32, ubatch: Vec<u32>, mut leaves: Vec<Hash>) -> Option<Self> {
        let nl = m.n_leaves();
        let n_pos = ubatch.len();
        if n_pos == 0 || n_prompt == 0 || n_prompt as usize > n_pos || leaves.len() != n_pos * nl || n_pos > u32::MAX as usize {
            return None;
        }
        let empty = *blake3::hash(&[]).as_bytes();
        let np = m.n_products as usize;
        for t in 0..n_pos {
            let pos = &mut leaves[t * nl..(t + 1) * nl];
            for (r, (e, _)) in m.entries.iter().enumerate().take(np) {
                if e.glu_form == manifest::FORM_PAIR_UP {
                    pos[2 * r + 1] = empty;
                }
                if e.role == manifest::ROLE_OUTPUT && t + 1 < n_prompt as usize {
                    pos[2 * r] = empty;
                    pos[2 * r + 1] = empty;
                }
            }
            pos[nl - 1] = meta_leaf(t as u32, ubatch[t]);
        }
        if leaves.iter().any(|h| *h == [0u8; 32]) {
            return None;
        }
        let pos_roots: Vec<Hash> = (0..n_pos).map(|t| b3_root(&leaves[t * nl..(t + 1) * nl])).collect();
        let trace_root = b3_root(&pos_roots);
        Some(Self { n_prompt, n_pos: n_pos as u32, ubatch, n_leaves: nl, leaves, pos_roots, trace_root })
    }

    pub fn commitment(&self) -> TraceCommitment {
        TraceCommitment { trace_root: self.trace_root, n_prompt: self.n_prompt, n_pos: self.n_pos, stage: STAGE_1A }
    }

    pub fn leaves_at(&self, t: u64) -> &[Hash] {
        let t = t as usize;
        &self.leaves[t * self.n_leaves..(t + 1) * self.n_leaves]
    }

    pub fn leaf(&self, t: u64, index: u32) -> Hash {
        self.leaves_at(t)[index as usize]
    }
}

/// Draw seed of a response accepted before chain block `seed_block_hash`.
pub fn trace_seed(seed_block_hash: &[u8; 32], response_txid: &[u8; 32]) -> Hash {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(TRACE_DRAW_DOMAIN);
    hasher.update(seed_block_hash);
    hasher.update(response_txid);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(hasher.finalize().as_bytes());
    seed
}

pub fn audited(seed: &Hash) -> bool {
    draw::audited(seed)
}

/// What the miner keeps of a traced response to open it later.
#[derive(BorshSerialize, BorshDeserialize)]
pub struct ResponseTrace {
    pub record: TraceRecord,
    /// Prompt then generated tokens, as fed to the engine.
    pub tokens: Vec<i32>,
}

impl ResponseTrace {
    /// A small synthetic trace for store tests.
    #[cfg(test)]
    pub fn test_sample() -> Self {
        let record = TraceRecord {
            n_prompt: 2,
            n_pos: 3,
            ubatch: vec![2, 1],
            n_leaves: 3,
            leaves: (0..9u8).map(|i| [i; 32]).collect(),
            pos_roots: (0..3u8).map(|i| [0x10 + i; 32]).collect(),
            trace_root: [0xaa; 32],
        };
        Self { record, tokens: vec![5, 6, 7] }
    }
}

/// Opening window after the seed block, in DAA. MUST equal the node's `service_window_daa_at`.
pub fn opening_window_daa(tier: u8, max_tokens: u32) -> u64 {
    const BASE_DAA: u64 = 3_000;
    const MAX_TOKENS_CAP: u32 = 4_096;
    let per_token_daa: u64 = match tier {
        0..=2 => 2,
        3 => 3,
        _ => 4,
    };
    BASE_DAA + max_tokens.min(MAX_TOKENS_CAP) as u64 * per_token_daa
}

/// One drawn point: the manifest entry, the gate of its pair, the position and the output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointSpec {
    pub rank: u64,
    pub g_rank: u64,
    pub t: u64,
    pub row: u64,
}

impl PointSpec {
    pub fn leaf_x(&self) -> u32 {
        2 * self.g_rank as u32
    }

    pub fn leaf_y(&self) -> u32 {
        2 * self.g_rank as u32 + 1
    }
}

/// Point `i` of the draw of `seed`, as the node will verify it.
pub fn draw_point(m: &TierManifest, n_prompt: u32, n_pos: u32, seed: &Hash, i: u32) -> Option<PointSpec> {
    if n_pos == 0 || n_prompt == 0 || n_prompt > n_pos {
        return None;
    }
    let d = draw::point(seed, i);
    let rank = d[1] % m.n_products as u64;
    let e0 = m.entry(rank);
    let n_traced = n_pos as u64;
    let t = if e0.role == manifest::ROLE_OUTPUT {
        let first_pos = n_prompt as u64 - 1;
        first_pos + d[0] % (n_traced - first_pos)
    } else {
        d[0] % n_traced
    };
    let g_rank = if e0.glu_form == manifest::FORM_PAIR_UP { e0.partner as u64 } else { rank };
    let g = m.entry(g_rank);
    if g.n_exp != 1 {
        return None;
    }
    let m_out = if g.glu_form == manifest::FORM_SPLIT { g.m / 2 } else { g.m };
    if m_out == 0 {
        return None;
    }
    Some(PointSpec { rank, g_rank, t, row: d[2] % m_out })
}

/// Chunks `first..=last` of the resident model with their range proof against `R_T`.
pub fn range_proof(index: &WeightIndex, first: u64, last: u64) -> Option<ChunkRange> {
    if first > last || last >= index.n_chunks {
        return None;
    }
    let mut chunks = Vec::with_capacity(((last - first + 1) * 32) as usize);
    for c in first..=last {
        chunks.extend_from_slice(&index.read_chunk_bytes(c));
    }
    let path_first = index.merkle_path(first);
    let path_last = if last == first { path_first.clone() } else { index.merkle_path(last) };
    let (mut lo, mut hi, mut n) = (first, last, index.n_chunks);
    let mut proof = Vec::new();
    let mut level = 0;
    while n > 1 {
        if lo % 2 == 1 {
            proof.push(*path_first.get(level)?);
        }
        if hi % 2 == 0 && hi + 1 < n {
            proof.push(*path_last.get(level)?);
        }
        lo /= 2;
        hi /= 2;
        n = n.div_ceil(2);
        level += 1;
    }
    Some(ChunkRange { first_chunk: first, chunks, proof })
}

/// The opening of `p` from the captured input bytes `x` and output vector `y` of its gate.
pub fn build_point(m: &TierManifest, rec: &TraceRecord, p: &PointSpec, x: Vec<u8>, y: &[u8], index: &WeightIndex) -> Option<Point> {
    let g = m.entry(p.g_rank);
    let m_out = if g.glu_form == manifest::FORM_SPLIT { g.m / 2 } else { g.m };
    if y.len() as u64 != m_out * 4 || p.t >= rec.n_pos as u64 {
        return None;
    }
    let row_range = |e: &Entry, row: u64| -> Option<ChunkRange> {
        let (first, last, _) = e.row_chunks(0, row)?;
        range_proof(index, first, last)
    };
    let mut rows = vec![row_range(g, p.row)?];
    let partner = match g.glu_form {
        manifest::FORM_NONE => None,
        manifest::FORM_SPLIT => {
            rows.push(row_range(g, p.row + m_out)?);
            None
        }
        manifest::FORM_PAIR_GATE => {
            let up_rank = g.partner as u64;
            rows.push(row_range(m.entry(up_rank), p.row)?);
            Some(m.entry_opening(if p.rank == p.g_rank { up_rank } else { p.g_rank }))
        }
        _ => return None,
    };
    let bias = if g.bias != manifest::NONE {
        let b = m.entry(g.bias as u64);
        let first = b.chunk_base + p.row * 4 / 32;
        Some((m.entry_opening(g.bias as u64), range_proof(index, first, first)?))
    } else {
        None
    };
    let pos_leaves = rec.leaves_at(p.t);
    let nl = rec.n_leaves;
    let c = (p.row * 4 / B3_CHUNK as u64) as usize;
    let n_yc = y.len().div_ceil(B3_CHUNK);
    let y_chunk_path = if n_yc == 1 {
        Vec::new()
    } else {
        let cvs: Vec<Hash> = (0..n_yc).map(|j| b3_chunk_cv(&y[j * B3_CHUNK..((j + 1) * B3_CHUNK).min(y.len())], j as u64)).collect();
        b3_path(&cvs, c)
    };
    Some(Point {
        n_ubatch: rec.ubatch[p.t as usize],
        pos_path: b3_path(&rec.pos_roots, p.t as usize),
        meta_path: b3_path(pos_leaves, nl - 1),
        entry: m.entry_opening(p.rank),
        partner,
        bias,
        rows,
        x,
        x_path: b3_path(pos_leaves, p.leaf_x() as usize),
        y_total_len: y.len() as u32,
        y_chunk_index: c as u32,
        y_chunk: y[c * B3_CHUNK..((c + 1) * B3_CHUNK).min(y.len())].to_vec(),
        y_chunk_path,
        y_path: b3_path(pos_leaves, p.leaf_y() as usize),
    })
}

/// The prefix position the draw of `seed` opens.
pub fn draw_prefix(ctx: &PrefixCtx, seed: &Hash) -> u64 {
    prefix::position(seed, ctx.offset)
}

/// The prefix point at position `t` from the captured input bytes `x` of the first product.
pub fn build_prefix_point(m: &TierManifest, rec: &TraceRecord, ctx: &PrefixCtx, request_hash: &[u8; 32], t: u64, x: Vec<u8>, index: &WeightIndex) -> Option<PrefixPoint> {
    if t >= rec.n_pos as u64 {
        return None;
    }
    let tok = prefix::expected_token(ctx, request_hash, t)?;
    let embd = m.entry(ctx.embd_entry as u64);
    let norm = m.entry(ctx.norm_entry as u64);
    let (first, last, _) = embd.row_chunks(0, tok as u64)?;
    let norm_last = norm.chunk_base + (4 * norm.k - 1) / 32;
    let pos_leaves = rec.leaves_at(t);
    Some(PrefixPoint {
        n_ubatch: rec.ubatch[t as usize],
        pos_path: b3_path(&rec.pos_roots, t as usize),
        meta_path: b3_path(pos_leaves, rec.n_leaves - 1),
        entry: m.entry_opening(0),
        x,
        x_path: b3_path(pos_leaves, 0),
        embd: (m.entry_opening(ctx.embd_entry as u64), range_proof(index, first, last)?),
        norm: (m.entry_opening(ctx.norm_entry as u64), range_proof(index, norm.chunk_base, norm_last)?),
    })
}

/// Encoded opening body, `None` above the consensus size cap.
pub fn opening_body(response_txid: [u8; 32], points: Vec<Point>, prefix: Option<PrefixPoint>) -> Option<Vec<u8>> {
    let body = Opening { response_txid, points, prefix, links: Vec::new(), experts: Vec::new(), resid: None }.encode();
    (body.len() <= opening::MAX_BODY_LEN).then_some(body)
}

/// The message an opening is signed over.
pub fn opening_signature_message(body: &[u8]) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(OPENING_SIG_DOMAIN);
    hasher.update(body);
    let mut msg = [0u8; 32];
    msg.copy_from_slice(hasher.finalize().as_bytes());
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_verifiable_inference::merkle::{rt_range_proof, rt_verify_range};
    use keryx_verifiable_inference::verify::{verify_point, ResponseCtx, TierCtx};

    /// `M_T`, entries and products pinned by the node (`TRACE_MANIFESTS_PREFIX`).
    const PINNED: [(&str, u64, u64); 4] = [
        ("9b802ae5f4cafe073bd97b72222a77f44eb597491ab07235180c5ef8783f62bc", 251, 249),
        ("5d678ffbd6c4164cc8d4006168adc276541fdf3f181ea8a89cac70231e7d7d1a", 363, 241),
        ("c09c80b1815b93dba8b430f2648e50be87137de945f96ca82e4cfa2084b2a808", 331, 329),
        ("98dc87dc3cf1297232e89e58bfae501c1172ba05626807a086ca38cdaffb4c47", 499, 497),
    ];

    /// `M_T` and entries pinned by the node before the prefix rule (`TRACE_MANIFESTS_H14`).
    const PINNED_LEGACY: [(&str, u64); 4] = [
        ("4c6212d16c7fc573b372e57e60176190b63ed3637a7bdfa93fbf6a7ca2bfd492", 249),
        ("c586bd7c3ef5aeadd669a48e28beb135a9df43890bf4b30a26072cdebba1b045", 361),
        ("1550e2f3a709f1653663111196edeb8088a0c50873687e70ef77bbbcdaa949c8", 329),
        ("5ff3b42ff04356106ae7a56512321fae0f37bde171495b774bf944e4d0235c33", 497),
    ];

    /// `M_T`, entries and products of the stage-2 manifests pinned by the node (`TRACE_CHAINS_H15`).
    const PINNED_CHAIN: [(&str, u64, u64); 5] = [
        ("9f1ea28681bfa10a6afe4d68cbe27ebea8656f76369af1bfa2adbf71e10374d0", 347, 249),
        ("0431fa87e1a48b48151370c3fc7604de9b72461c9ac82881beeebb11febbfdeb", 563, 241),
        ("4463e82828cc458ec86753a2c792d8f0da50c0d13d9758b9a49ad030dc2bef0f", 571, 329),
        ("bc417110ea8e18db4fe6fe656bb6c7a47375585c86f9047ab7e40c28e11d0800", 691, 497),
        ("c846b9cf4d7fd0119653d517c11cec3ac096013ed07268839db6949f0de7db93", 530, 375),
    ];

    #[test]
    fn chain_manifests_match_the_node_pins() {
        for (tier, (m_t, n_entries, n_products)) in PINNED_CHAIN.iter().enumerate() {
            let (entries, np) = chain_manifest(tier as u8).expect("chain manifest");
            let leaves: Vec<_> = entries.iter().map(|(_, b)| keryx_verifiable_inference::manifest::Entry::leaf(b)).collect();
            assert_eq!(hex::encode(rt_range_proof(&leaves, 0, 1).0), *m_t, "tier {tier}");
            assert_eq!((entries.len() as u64, np as u64), (*n_entries, *n_products), "tier {tier}");
            let (ctx, prefix) = chain_ctx(tier as u8).expect("chain ctx");
            assert_eq!(entries[prefix.embd_entry as usize].0.role, keryx_verifiable_inference::manifest::ROLE_EMBD, "tier {tier}");
            assert_eq!(entries[prefix.norm_entry as usize].0.role, keryx_verifiable_inference::manifest::ROLE_NORM, "tier {tier}");
            assert_eq!(entries[ctx.layer_base as usize].0.role, keryx_verifiable_inference::manifest::ROLE_LAYER, "tier {tier}");
        }
    }

    #[test]
    fn embedded_manifests_match_the_node_pins() {
        for (tier, (m_t, n_entries, n_products)) in PINNED.iter().enumerate() {
            let m = TierManifest::for_tier(tier as u8).expect("manifest");
            assert_eq!(hex::encode(m.m_t), *m_t, "tier {tier}");
            assert_eq!(m.entries.len() as u64, *n_entries, "tier {tier}");
            assert_eq!(m.n_products as u64, *n_products, "tier {tier}");
            for i in 0..m.entries.len() as u64 {
                let eo = m.entry_opening(i);
                assert!(manifest::verify_entry(&m.m_t, *n_entries, i, &eo.bytes, &eo.path));
            }
            let ctx = prefix_ctx(tier as u8).expect("prefix context");
            let (embd, norm) = (m.entry(ctx.embd_entry as u64), m.entry(ctx.norm_entry as u64));
            assert_eq!((embd.role, norm.role), (manifest::ROLE_EMBD, manifest::ROLE_NORM), "tier {tier}");
            assert_eq!((embd.k, norm.k, norm.wtype), (m.entry(0).k, m.entry(0).k, 0), "tier {tier}");
            assert_eq!(m.entry(0).role, manifest::ROLE_PRODUCT, "tier {tier}");
            assert_eq!(ctx.table.iter().collect::<std::collections::BTreeSet<_>>().len(), 256, "tier {tier}");

            let (m_t, n_entries) = PINNED_LEGACY[tier];
            let legacy = m.legacy();
            assert_eq!(hex::encode(legacy.m_t), m_t, "tier {tier}");
            assert_eq!(legacy.entries.len() as u64, n_entries, "tier {tier}");
            assert_eq!(legacy.n_products, m.n_products, "tier {tier}");
            assert!(legacy.entries.iter().all(|(e, _)| e.role != manifest::ROLE_EMBD && e.role != manifest::ROLE_NORM));
        }
        assert!(TierManifest::for_tier(4).is_none());
        assert!(TierManifest::for_tier(9).is_none());
        assert!(prefix_ctx(4).is_none());
    }

    #[test]
    fn prefix_tokens_follow_the_tier_table() {
        let ctx = prefix_ctx(0).unwrap();
        let mut rh = [0u8; 32];
        rh[0] = 0x41;
        rh[31] = 0xff;
        let toks = prefix_tokens(ctx, &rh);
        assert_eq!(toks.len(), prefix::PREFIX_LEN as usize);
        assert_eq!(toks[0], ctx.table[0x41] as i32);
        assert_eq!(toks[31], ctx.table[0xff] as i32);
        assert_eq!(toks[1], ctx.table[0] as i32);
        for t in 0..64u8 {
            let seed = [t; 32];
            let p = draw_prefix(ctx, &seed);
            assert!((ctx.offset as u64..ctx.offset as u64 + 32).contains(&p));
            assert_eq!(prefix::expected_token(ctx, &rh, p), Some(toks[(p - ctx.offset as u64) as usize] as u32));
        }
    }

    #[test]
    fn range_proofs_match_the_reference_tree() {
        for n in [64u64, 65, 127, 1000, 4097, 70_001] {
            let data: Vec<u8> = (0..n * 32).map(|i| (i * 31 + 7) as u8).collect();
            let index = crate::pom::index_from_ram(data.clone());
            let leaves: Vec<Hash> = data.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
            let (root, _) = rt_range_proof(&leaves, 0, 1);
            assert_eq!(root, index.r_t, "n {n}");
            for first in [0, 1, n / 2, n.saturating_sub(2), n - 1] {
                for count in 1..=(n - first).min(5) {
                    let r = range_proof(&index, first, first + count - 1).unwrap();
                    let (_, expected) = rt_range_proof(&leaves, first, count);
                    assert_eq!(r.proof, expected, "n {n} first {first} count {count}");
                    let got: Vec<Hash> = r.chunks.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
                    assert!(rt_verify_range(&index.r_t, n, first, &got, &r.proof));
                }
            }
            assert!(range_proof(&index, n - 1, n).is_none());
        }
    }

    #[test]
    fn record_fills_in_the_canonical_leaves() {
        let m = TierManifest::for_tier(0).unwrap();
        let nl = m.n_leaves();
        let n_pos = 5;
        let mut leaves = vec![[1u8; 32]; n_pos * nl];
        // The engine never writes Y(up) of a pair, nor the head before the last prompt token.
        let empty = *blake3::hash(&[]).as_bytes();
        let rec = TraceRecord::new(&m, 3, vec![3, 3, 3, 1, 1], leaves.clone()).unwrap();
        assert_eq!(rec.n_pos, 5);
        let up = m.entries.iter().position(|(e, _)| e.glu_form == manifest::FORM_PAIR_UP).unwrap();
        let head = m.entries.iter().position(|(e, _)| e.role == manifest::ROLE_OUTPUT).unwrap();
        assert_eq!(rec.leaf(0, 2 * up as u32 + 1), empty);
        assert_eq!(rec.leaf(0, 2 * head as u32), empty);
        assert_eq!(rec.leaf(2, 2 * head as u32), [1u8; 32]);
        assert_eq!(rec.leaf(4, nl as u32 - 1), meta_leaf(4, 1));
        let c = rec.commitment();
        assert_eq!((c.n_prompt, c.n_pos, c.stage), (3, 5, STAGE_1A));
        // An unwritten product leaf is an incomplete trace.
        leaves[nl + 2] = [0u8; 32];
        assert!(TraceRecord::new(&m, 3, vec![3, 3, 3, 1, 1], leaves.clone()).is_none());
        assert!(TraceRecord::new(&m, 0, vec![1; 5], leaves.clone()).is_none());
        assert!(TraceRecord::new(&m, 6, vec![1; 5], leaves).is_none());
    }

    #[test]
    fn draw_matches_the_verifier() {
        let m = TierManifest::for_tier(1).unwrap();
        let seed = trace_seed(&[7u8; 32], &[9u8; 32]);
        let mut seen_out = false;
        for i in 0..64u32 {
            let s = blake3::hash(&i.to_le_bytes());
            let p = draw_point(&m, 10, 40, s.as_bytes(), 0).unwrap();
            let e = m.entry(p.rank);
            if e.role == manifest::ROLE_OUTPUT {
                assert!(p.t >= 9 && p.t < 40);
                seen_out = true;
            } else {
                assert!(p.t < 40);
            }
            let g = m.entry(p.g_rank);
            let m_out = if g.glu_form == manifest::FORM_SPLIT { g.m / 2 } else { g.m };
            assert!(p.row < m_out);
            if e.glu_form == manifest::FORM_PAIR_UP {
                assert_eq!(p.g_rank, e.partner as u64);
            } else {
                assert_eq!(p.g_rank, p.rank);
            }
        }
        assert!(seen_out, "the head is drawn within 64 seeds");
        assert_eq!(audited(&seed), u16::from_le_bytes([seed[0], seed[1]]) < draw::AUDIT_THRESHOLD);
        assert!(draw_point(&m, 0, 40, &seed, 0).is_none());
        assert!(draw_point(&m, 41, 40, &seed, 0).is_none());
    }

    /// Bench dump of a traced response (`KXQ8C001`) and one capture, as written by `q8canon.cu`.
    fn read_dump(path: &std::path::Path) -> (u32, Vec<u32>, Vec<Hash>) {
        let b = std::fs::read(path).unwrap();
        assert_eq!(&b[..8], b"KXQ8C001");
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (n_pos, n_prompt) = (u(8) as usize, u(12));
        let ubatch = (0..n_pos).map(|i| u(24 + 4 * i)).collect();
        let o = 24 + 4 * n_pos;
        let leaves = b[o..].chunks_exact(32).map(|c| c.try_into().unwrap()).collect();
        (n_prompt, ubatch, leaves)
    }

    fn read_capture(path: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
        let cap = std::fs::read(path).unwrap();
        let xl = u32::from_le_bytes(cap[..4].try_into().unwrap()) as usize;
        let yl = u32::from_le_bytes(cap[4 + xl..8 + xl].try_into().unwrap()) as usize;
        (cap[4..4 + xl].to_vec(), cap[8 + xl..8 + xl + yl].to_vec())
    }

    /// Real openings from the bench traces, verified as the node would. Needs the bench
    /// directory (`KERYX_VI_BENCH`) and the tier GGUFs with their `pom-tree.bin`
    /// (`KERYX_VI_MODELS`); skipped otherwise.
    #[test]
    fn bench_openings_verify() {
        let (Ok(bench), Ok(models)) = (std::env::var("KERYX_VI_BENCH"), std::env::var("KERYX_VI_MODELS")) else {
            eprintln!("skipped: set KERYX_VI_BENCH and KERYX_VI_MODELS");
            return;
        };
        let bench = std::path::Path::new(&bench).join("spec/q8c");
        let cases: [(u8, &str, &str, &str, &str); 3] = [
            (1, "GLM-4-9B-0414", "glm-run1.bin", "glm-seeds.txt", "glmcap"),
            (2, "Gemma-4-12B-abliterated", "Gemma-4-12B-abliterated-run.bin", "Gemma-4-12B-abliterated-seeds.txt", "Gemma-4-12B-abliterated-cap"),
            (3, "Qwen3.8-27B", "Qwen3.8-27B-run.bin", "Qwen3.8-27B-seeds.txt", "Qwen3.8-27B-cap"),
        ];
        for (tier, model, run, seeds, cap_prefix) in cases {
            let gguf = std::path::Path::new(&models).join(model).join("model.gguf");
            if !gguf.exists() || !bench.join(run).exists() {
                eprintln!("skipped tier {tier}: files missing");
                continue;
            }
            let spec = crate::models::spec_for_gguf(gguf.to_str().unwrap()).expect("registry model");
            let index = WeightIndex::build_from_gguf(gguf.to_str().unwrap(), spec.model_id).unwrap();
            let m = TierManifest::for_tier(tier).unwrap();
            let (n_prompt, ubatch, leaves) = read_dump(&bench.join(run));
            let rec = TraceRecord::new(&m, n_prompt, ubatch, leaves).expect("complete trace");
            let tier_ctx = TierCtx { r_t: index.r_t, n_chunks: index.n_chunks, m_t: m.m_t, n_entries: m.entries.len() as u64, n_products: m.n_products as u64 };
            let resp = ResponseCtx { trace_root: rec.trace_root, n_prompt: rec.n_prompt, n_pos: rec.n_pos };
            let mut n_ok = 0;
            for line in std::fs::read_to_string(bench.join(seeds)).unwrap().lines() {
                let counter: u64 = line.split_whitespace().next().unwrap().parse().unwrap();
                let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
                let p = draw_point(&m, rec.n_prompt, rec.n_pos, &seed, 0).unwrap();
                let (x, y) = read_capture(&bench.join(format!("{cap_prefix}{counter}.bin")));
                assert_eq!(*blake3::hash(&x).as_bytes(), rec.leaf(p.t, p.leaf_x()), "{model} seed {counter}: X leaf");
                assert_eq!(*blake3::hash(&y).as_bytes(), rec.leaf(p.t, p.leaf_y()), "{model} seed {counter}: Y leaf");
                let point = build_point(&m, &rec, &p, x, &y, &index).unwrap();
                let body = opening_body([3u8; 32], vec![point], None).unwrap();
                let decoded = Opening::decode(&body).unwrap();
                let report = verify_point(&tier_ctx, &resp, &seed, 0, &decoded.points[0]).unwrap_or_else(|e| panic!("{model} seed {counter}: {e:?}"));
                assert_eq!((report.t, report.rank, report.row), (p.t, p.rank, p.row));
                n_ok += 1;
            }
            eprintln!("{model}: {n_ok} openings verified");
            assert!(n_ok > 0);
        }
    }

    /// Traced generation, replay and opening on the resident model, verified as the node would.
    #[test]
    #[ignore = "requires one CUDA GPU, libkeryx-llama, and KERYX_TEST_MODEL_GPU0 (a tier GGUF)"]
    fn traced_generation_replays_and_verifies() {
        use crate::llama_engine;
        let gguf = std::env::var("KERYX_TEST_MODEL_GPU0").expect("set KERYX_TEST_MODEL_GPU0");
        let spec = crate::models::spec_for_gguf(&gguf).expect("registry model");
        let tier = crate::models::pom_tier_index(&spec.model_id, u64::MAX).expect("tiered model");
        let m = TierManifest::for_tier(tier).expect("audited tier");
        let index = WeightIndex::build_from_gguf(&gguf, spec.model_id).unwrap();

        llama_engine::ensure_loaded(&gguf, 0).unwrap();
        let nl = llama_engine::trace_configure(&gguf, 0, TierManifest::raw_for_tier(tier).unwrap()).unwrap();
        assert_eq!(nl, m.n_leaves());
        let prompt = "Explain in two sentences why the sky is blue.";
        let t0 = std::time::Instant::now();
        let g = llama_engine::generate_traced(&gguf, 0, prompt, None, 24, 0, 0).unwrap();
        eprintln!("generated {} tokens ({} prompt) in {:.2?}: {:?}", g.tokens.len(), g.n_prompt, t0.elapsed(), g.text);
        assert_eq!(g.n_pos as usize, g.tokens.len());
        assert_eq!(g.leaves.len(), g.tokens.len() * nl);
        let rec = TraceRecord::new(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone()).expect("complete trace");
        let c = rec.commitment();
        assert_eq!((c.n_prompt, c.n_pos, c.stage), (g.n_prompt, g.n_pos, STAGE_1A));

        let tier_ctx = TierCtx { r_t: index.r_t, n_chunks: index.n_chunks, m_t: m.m_t, n_entries: m.entries.len() as u64, n_products: m.n_products as u64 };
        let resp = ResponseCtx { trace_root: rec.trace_root, n_prompt: rec.n_prompt, n_pos: rec.n_pos };
        let mut n_ok = 0;
        for counter in 0..8u64 {
            let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
            for i in 0..POINTS {
                let p = draw_point(&m, rec.n_prompt, rec.n_pos, &seed, i).unwrap();
                let t0 = std::time::Instant::now();
                let (x, y) = llama_engine::replay(&gguf, 0, &g.tokens, g.n_prompt, p.t as u32, p.leaf_x(), p.leaf_y()).unwrap();
                let replay = t0.elapsed();
                assert_eq!(*blake3::hash(&x).as_bytes(), rec.leaf(p.t, p.leaf_x()), "seed {counter} point {i}: X leaf");
                assert_eq!(*blake3::hash(&y).as_bytes(), rec.leaf(p.t, p.leaf_y()), "seed {counter} point {i}: Y leaf");
                let point = build_point(&m, &rec, &p, x, &y, &index).unwrap();
                let body = opening_body([3u8; 32], vec![point], None).unwrap();
                let decoded = Opening::decode(&body).unwrap();
                let report = verify_point(&tier_ctx, &resp, &seed, i, &decoded.points[0]).unwrap_or_else(|e| panic!("seed {counter} point {i}: {e:?}"));
                assert_eq!((report.t, report.rank, report.row), (p.t, p.rank, p.row));
                eprintln!(
                    "seed {counter} point {i}: t {} rank {} row {} replay {:.2?} body {} B |y-y_hat| {:.2e} tol {:.2e}",
                    p.t, p.rank, p.row, replay, body.len(), (report.y - report.y_hat).abs(), report.tol
                );
                n_ok += 1;
            }
        }
        // Shape padding: the prompt grows to the floor and the answer gets its generated positions.
        let g = llama_engine::generate_traced(&gguf, 0, prompt, None, 8, 64, 1).unwrap();
        assert_eq!(g.n_prompt, 64);
        assert!(g.n_pos - g.n_prompt >= g.text.len().min(8) as u32, "{} generated for {} B", g.n_pos - g.n_prompt, g.text.len());
        assert_eq!(g.n_pos as usize, g.tokens.len());
        assert!(TraceRecord::new(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone()).is_some());

        // Prefix: the request id sits at the tier offset and its point verifies as the node would.
        let ctx = prefix_ctx(tier).expect("prefix context");
        let (head, rest) = crate::llm::format_prompt_parts(spec.name, prompt).expect("prefix template");
        let rh: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
        let toks = prefix_tokens(ctx, &rh);
        let prefix = llama_engine::PromptPrefix { tokens: &toks, head_tokens: ctx.offset, rest: &rest };
        let g = llama_engine::generate_traced(&gguf, 0, &head, Some(prefix), 8, 0, 0).unwrap();
        assert_eq!(&g.tokens[ctx.offset as usize..ctx.offset as usize + 32], toks.as_slice());
        let rec = TraceRecord::new(&m, g.n_prompt, g.ubatch.clone(), g.leaves.clone()).expect("complete trace");
        let resp = ResponseCtx { trace_root: rec.trace_root, n_prompt: rec.n_prompt, n_pos: rec.n_pos };
        for counter in 0..4u64 {
            let seed = *blake3::hash(&counter.to_le_bytes()).as_bytes();
            let t = draw_prefix(ctx, &seed);
            let (x, _) = llama_engine::replay(&gguf, 0, &g.tokens, g.n_prompt, t as u32, 0, 1).unwrap();
            assert_eq!(*blake3::hash(&x).as_bytes(), rec.leaf(t, 0), "seed {counter}: prefix X leaf");
            let point = build_prefix_point(&m, &rec, ctx, &rh, t, x, &index).unwrap();
            let body = opening_body([3u8; 32], vec![], Some(point)).unwrap();
            let decoded = Opening::decode(&body).unwrap();
            let opened = prefix::verify_prefix_point(&tier_ctx, ctx, &resp, &seed, &rh, decoded.prefix.as_ref().unwrap())
                .unwrap_or_else(|e| panic!("seed {counter}: prefix {e:?}"));
            assert_eq!(opened, t);
            eprintln!("seed {counter}: prefix position {t} verified, body {} B", body.len());
        }
        llama_engine::unload(0);
        assert_eq!(n_ok, 8 * POINTS);
    }
}
