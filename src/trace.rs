//! Inference trace of a sealed response: commitment, draw and opening. Byte-exact mirror of the
//! node's `collateral.rs` trace rules on top of the shared `keryx-verifiable-inference` crate.

use keryx_inference::TraceCommitment;
use keryx_verifiable_inference::draw;
use keryx_verifiable_inference::manifest::{self, Entry};
use keryx_verifiable_inference::merkle::{b3_chunk_cv, b3_path, b3_root, rt_range_proof, Hash, B3_CHUNK};
use keryx_verifiable_inference::opening::{self, ChunkRange, EntryOpening, Opening, Point};

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

/// Encoded opening body, `None` above the consensus size cap.
pub fn opening_body(response_txid: [u8; 32], points: Vec<Point>) -> Option<Vec<u8>> {
    let body = Opening { response_txid, points }.encode();
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

    /// `M_T`, entries and products pinned by the node (`TRACE_MANIFESTS_H14`).
    const PINNED: [(&str, u64, u64); 4] = [
        ("4c6212d16c7fc573b372e57e60176190b63ed3637a7bdfa93fbf6a7ca2bfd492", 249, 249),
        ("c586bd7c3ef5aeadd669a48e28beb135a9df43890bf4b30a26072cdebba1b045", 361, 241),
        ("1550e2f3a709f1653663111196edeb8088a0c50873687e70ef77bbbcdaa949c8", 329, 329),
        ("5ff3b42ff04356106ae7a56512321fae0f37bde171495b774bf944e4d0235c33", 497, 497),
    ];

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
        }
        assert!(TierManifest::for_tier(4).is_none());
        assert!(TierManifest::for_tier(9).is_none());
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
                let body = opening_body([3u8; 32], vec![point]).unwrap();
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
        let g = llama_engine::generate_traced(&gguf, 0, prompt, 24).unwrap();
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
                let body = opening_body([3u8; 32], vec![point]).unwrap();
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
        llama_engine::unload(0);
        assert_eq!(n_ok, 8 * POINTS);
    }
}
