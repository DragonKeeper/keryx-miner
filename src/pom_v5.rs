//! PoM v5 (v4 walk + context pad) — byte-exact mirror of the node's
//! `consensus/core/src/pom_v5.rs`. Salts, rounds and the pad read MUST stay bit-identical.

use crate::pom::{blake, mix64};
use crate::pom_v3::fold64;
use crate::pom_v4::{v4_initial_state, v4_state_root, v4_transition, PomProofV4, POM_V4_D, POM_V4_K};

pub const POM_V5_P_STRIDE: usize = 4;
pub const POM_V5_P_ENTRY_BYTES: usize = POM_V4_D;
pub const POM_V5_P_ROUNDS: u32 = 61_440;
pub const POM_V5_P_SALT: u64 = 0x0850_3190_C199_C415;
/// Context pad entries of each H14 tier; 0 = no pad. Mirror of the node's `POM_V5_PAD_ENTRIES_H14`.
pub const POM_V5_PAD_ENTRIES_H14: [u64; 5] = [22_511_616, 32_243_712, 34_570_240, 57_835_520, 0];

const PAD_DOMAIN: &[u8] = b"KERYX-POM-V5-PAD";
pub const PAD_IV: [u64; 4] = [0x510e_527f_ade6_82d1, 0x9b05_688c_2b3e_6c1f, 0x1f83_d9ab_fb41_bd6b, 0x5be0_cd19_137e_2179];

pub fn v5_pad_seed(tier: u8, model_id: &[u8; 32], r_t: &[u8; 32]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(PAD_DOMAIN.len() + 1 + 64);
    buf.extend_from_slice(PAD_DOMAIN);
    buf.push(tier);
    buf.extend_from_slice(model_id);
    buf.extend_from_slice(r_t);
    blake(&buf)
}

#[inline(always)]
fn round4(a: &mut u64, b: &mut u64, c: &mut u64, d: &mut u64) {
    *a = a.wrapping_add(*b);
    *d ^= *a;
    *d = d.rotate_left(32);
    *c = c.wrapping_add(*d);
    *b ^= *c;
    *b = b.rotate_left(24);
    *a = a.wrapping_add(*b);
    *d ^= *a;
    *d = d.rotate_left(16);
    *c = c.wrapping_add(*d);
    *b ^= *c;
    *b = b.rotate_left(63);
}

/// The four seed words the GPU fill kernel starts from, IVs already folded in.
pub fn v5_pad_seed_words(pad_seed: &[u8; 32]) -> [u64; 4] {
    let w = |i: usize| u64::from_le_bytes(pad_seed[i * 8..i * 8 + 8].try_into().unwrap());
    [w(0) ^ PAD_IV[0], w(1) ^ PAD_IV[1], w(2) ^ PAD_IV[2], w(3) ^ PAD_IV[3]]
}

pub fn v5_pad_entry(pad_seed: &[u8; 32], index: u64) -> [u8; POM_V5_P_ENTRY_BYTES] {
    let s = v5_pad_seed_words(pad_seed);
    let (mut a, mut b, mut c, mut d) = (s[0], s[1] ^ index, s[2], s[3]);
    for _ in 0..POM_V5_P_ROUNDS {
        round4(&mut a, &mut b, &mut c, &mut d);
    }
    let mut out = [0u8; POM_V5_P_ENTRY_BYTES];
    out[0..8].copy_from_slice(&a.to_le_bytes());
    out[8..16].copy_from_slice(&b.to_le_bytes());
    out[16..24].copy_from_slice(&c.to_le_bytes());
    out[24..32].copy_from_slice(&d.to_le_bytes());
    out
}

/// State row and pad index read after the transition of `step`, when `step` is a pad step.
pub fn v5_pad_read(seed: u64, state: &[u8], step: u32, n_pad: u64) -> Option<(usize, u64)> {
    if step as usize % POM_V5_P_STRIDE != 0 {
        return None;
    }
    let row = (step as usize / POM_V5_P_STRIDE) % POM_V4_D;
    let r = &state[row * POM_V4_D..(row + 1) * POM_V4_D];
    let h = (0..4).fold(0u64, |h, i| h ^ u64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().unwrap()));
    Some((row, mix64(seed ^ h ^ (step as u64).wrapping_mul(POM_V5_P_SALT)) % n_pad))
}

/// `final_state` of the v5 walk over the tiles of a v4 proof, pad entries supplied by `pad`.
pub fn v5_final_state<F: FnMut(u64) -> [u8; POM_V5_P_ENTRY_BYTES]>(seed: u64, proof: &PomProofV4, n_pad: u64, mut pad: F) -> u64 {
    let mut state = v4_initial_state(seed);
    for step in 1..=POM_V4_K {
        state = v4_transition(&state, &proof.tiles[step - 1], step as u32);
        if let Some((row, idx)) = v5_pad_read(seed, &state, step as u32, n_pad) {
            let entry = pad(idx);
            for (s, e) in state[row * POM_V4_D..(row + 1) * POM_V4_D].iter_mut().zip(entry.iter()) {
                *s ^= e;
            }
        }
    }
    fold64(&v4_state_root(&state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_matches_the_node_vectors() {
        let seed = v5_pad_seed(1, &[7; 32], &[9; 32]);
        assert_eq!(hex::encode(seed), NODE_PAD_SEED_HEX);
        assert_eq!(hex::encode(v5_pad_entry(&seed, 5)), NODE_PAD_ENTRY_5_HEX);
        let state: Vec<u8> = (0..(POM_V4_D * POM_V4_D) as u32).map(|i| (i * 37 + 11) as u8).collect();
        assert_eq!(v5_pad_read(0xABCD, &state, 12, 1_000_003), Some(NODE_PAD_READ));
        let tiles: Vec<Vec<u8>> =
            (0..POM_V4_K as u32).map(|t| (0..(POM_V4_D * POM_V4_D) as u32).map(|b| (t * 31 + b * 7 + 3) as u8).collect()).collect();
        let proof = PomProofV4 { tier: 1, tiles, merkle: Vec::new() };
        assert_eq!(v5_final_state(0x55AA, &proof, 4_099, |i| v5_pad_entry(&seed, i)), NODE_FINAL_STATE);
    }

    const NODE_PAD_SEED_HEX: &str = "cb678ed7a91fb53d40ad3b966d0219e92c75ead7761d6404bd533cc7ddf34f96";
    const NODE_PAD_ENTRY_5_HEX: &str = "f849c84e259e59e8fef1d78da4de56d6f9df69aad05f571404d9a6372b35911c";
    const NODE_PAD_READ: (usize, u64) = (3, 172_680);
    const NODE_FINAL_STATE: u64 = 10_975_683_372_132_145_722;
}
