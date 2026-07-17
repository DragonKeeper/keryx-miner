//! Proof-of-Model GPU mining — runs the `pom_mine` kernel in candle's CUDA context over the
//! resident weight blob to find a winning nonce. Foundation for the live mining loop (§6/3b).
//!
//! Loads the mining tier's GGUF raw (so we get per-tensor device pointers for the gather, like
//! `pom-q4-probe`) and builds the chunk-prefix gather index on the GPU. NOTE: this is a second
//! VRAM copy of the model (the inference engine holds its own). Fine for small tiers on the
//! testnet; the big tiers will share buffers later.
//!
//! The kernel's seed/pow folds are byte-identical to `pom::pom_block_seed`/`pom::pom_pow_value`,
//! so a nonce found here builds a `PomProof` (host) the node accepts.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

use log::{info, warn};

use candle_core::cuda_backend::cudarc::driver::{result, sys, DevicePtr, CudaSlice, CudaStream, LaunchConfig};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{CudaDevice, Device};

const PTX_SM90: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm90.ptx"));
const PTX_SM89: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm89.ptx"));
const PTX_SM86: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm86.ptx"));
const PTX_SM80: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm80.ptx"));
const PTX_SM75: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm75.ptx"));
const PTX_SM70: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm70.ptx"));
const PTX_SM61: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine_sm61.ptx"));
const FATBIN_LEGACY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_mine_legacy.fatbin"));
const FATBIN_NEXTGEN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_mine_nextgen.fatbin"));
const CHUNK_BYTES: usize = 32;
const POM_KERNEL_NAME: &str = "pom_mine";

const POM_PTX_CANDIDATES: [(&str, &str, &str); 7] = [
    ("pom_mine_mod_sm90", "sm_90", PTX_SM90),
    ("pom_mine_mod_sm89", "sm_89", PTX_SM89),
    ("pom_mine_mod_sm86", "sm_86", PTX_SM86),
    ("pom_mine_mod_sm80", "sm_80", PTX_SM80),
    ("pom_mine_mod_sm75", "sm_75", PTX_SM75),
    ("pom_mine_mod_sm70", "sm_70", PTX_SM70),
    ("pom_mine_mod_sm61", "sm_61", PTX_SM61),
];

#[derive(Clone, Debug)]
pub struct GpuKernelInfo {
    pub device_id: u32,
    pub cc_major: Option<i32>,
    pub cc_minor: Option<i32>,
    pub image: String,
    pub load_path: String,
}

fn gpu_kernel_info() -> &'static Mutex<HashMap<u32, GpuKernelInfo>> {
    static GPU_KERNEL_INFO: OnceLock<Mutex<HashMap<u32, GpuKernelInfo>>> = OnceLock::new();
    GPU_KERNEL_INFO.get_or_init(|| Mutex::new(HashMap::new()))
}

fn set_gpu_kernel_info(
    device_id: usize,
    cc: Option<(i32, i32)>,
    image: &str,
    load_path: &str,
) {
    let entry = GpuKernelInfo {
        device_id: device_id as u32,
        cc_major: cc.map(|x| x.0),
        cc_minor: cc.map(|x| x.1),
        image: image.to_string(),
        load_path: load_path.to_string(),
    };
    if let Ok(mut g) = gpu_kernel_info().lock() {
        g.insert(device_id as u32, entry);
    }
}

pub fn list_gpu_kernel_info() -> Vec<GpuKernelInfo> {
    let mut out = gpu_kernel_info()
        .lock()
        .map(|g| g.values().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    out.sort_by_key(|e| e.device_id);
    out
}

#[derive(Debug)]
struct LoadedPomKernel {
    cuda: CudaDevice,
    module: sys::CUmodule,
    function: sys::CUfunction,
}

impl Drop for LoadedPomKernel {
    fn drop(&mut self) {
        let module = self.module;
        if !module.is_null() {
            // Best-effort cleanup; a drop failure here would only leak the module.
            let _ = unsafe { result::module::unload(module) };
        }
    }
}

unsafe impl Send for LoadedPomKernel {}
unsafe impl Sync for LoadedPomKernel {}

impl LoadedPomKernel {
    fn from_fatbin(cuda: CudaDevice, label: &'static str, fatbin: &'static [u8]) -> candle_core::Result<Self> {
        if fatbin.is_empty() {
            return Err(candle_core::Error::Msg(format!("PoM GPU: {} fatbin is empty", label)));
        }
        let module = unsafe { result::module::load_data(fatbin.as_ptr() as *const c_void) }
            .map_err(candle_core::Error::wrap)?;
        let function = unsafe { result::module::get_function(module, CString::new(POM_KERNEL_NAME).unwrap()) }
            .map_err(candle_core::Error::wrap)?;
        Ok(Self { cuda, module, function })
    }

    fn from_ptx(cuda: CudaDevice, _label: &'static str, ptx: &'static str) -> candle_core::Result<Self> {
        let c_src = CString::new(ptx).map_err(candle_core::Error::wrap)?;
        let module = unsafe { result::module::load_data(c_src.as_ptr() as *const c_void) }
            .map_err(candle_core::Error::wrap)?;
        let function = unsafe { result::module::get_function(module, CString::new(POM_KERNEL_NAME).unwrap()) }
            .map_err(candle_core::Error::wrap)?;
        Ok(Self { cuda, module, function })
    }

    fn launch(
        &self,
        stream: &CudaStream,
        bases_dev: &CudaSlice<u64>,
        prefix_dev: &CudaSlice<u64>,
        t_count: u32,
        n_total_chunks: u64,
        p_words: &[u64; 4],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> candle_core::Result<Option<u64>> {
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        let winner = self.cuda.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let grid = ((batch + 255) / 256) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

        let (bases_ptr, _bases_guard) = bases_dev.device_ptr(stream);
        let (prefix_ptr, _prefix_guard) = prefix_dev.device_ptr(stream);
        let (winner_ptr, _winner_guard) = winner.device_ptr(stream);

        let mut params: [*mut c_void; 17] = [
            (&bases_ptr as *const _ as *mut c_void),
            (&prefix_ptr as *const _ as *mut c_void),
            (&t_count as *const _ as *mut c_void),
            (&n_total_chunks as *const _ as *mut c_void),
            (&k as *const _ as *mut c_void),
            (&p_words[0] as *const _ as *mut c_void),
            (&p_words[1] as *const _ as *mut c_void),
            (&p_words[2] as *const _ as *mut c_void),
            (&p_words[3] as *const _ as *mut c_void),
            (&timestamp as *const _ as *mut c_void),
            (&t[0] as *const _ as *mut c_void),
            (&t[1] as *const _ as *mut c_void),
            (&t[2] as *const _ as *mut c_void),
            (&t[3] as *const _ as *mut c_void),
            (&start as *const _ as *mut c_void),
            (&batch as *const _ as *mut c_void),
            (&winner_ptr as *const _ as *mut c_void),
        ];

        unsafe { result::launch_kernel(self.function, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes, stream.cu_stream(), &mut params) }
            .map_err(candle_core::Error::wrap)?;
        stream.synchronize().map_err(candle_core::Error::wrap)?;

        let w = self.cuda.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }
}

fn is_nextgen_device(device_id: usize) -> bool {
    let Ok(dev) = result::device::get(device_id as i32) else {
        return false;
    };
    let major = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(0);
    let minor = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    major > 8 || (major == 8 && minor >= 9)
}

fn gpu_compute_capability(device_id: usize) -> Option<(i32, i32)> {
    let dev = result::device::get(device_id as i32).ok()?;
    let major = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .ok()?;
    let minor = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .ok()?;
    Some((major, minor))
}

fn select_pom_kernel(cuda: CudaDevice, device_id: usize) -> candle_core::Result<LoadedPomKernel> {
    static FATBIN_STATUS_LOGGED: Once = Once::new();
    FATBIN_STATUS_LOGGED.call_once(|| {
        let legacy = FATBIN_LEGACY.len();
        let nextgen = FATBIN_NEXTGEN.len();
        if legacy > 0 || nextgen > 0 {
            info!(
                "PoM: prebuilt fatbins detected (legacy={} bytes, nextgen={} bytes); PTX fallback ladder currently active",
                legacy,
                nextgen
            );
        } else {
            info!("PoM: no prebuilt fatbins detected; using PTX fallback ladder");
        }
    });

    let is_nextgen_cc = is_nextgen_device(device_id);

    let fatbin_candidates: [(&str, &str, &[u8]); 2] = if is_nextgen_cc {
        [
            ("pom_mine_mod_nextgen", "nextgen fatbin", FATBIN_NEXTGEN),
            ("pom_mine_mod_legacy", "legacy fatbin", FATBIN_LEGACY),
        ]
    } else {
        [
            ("pom_mine_mod_legacy", "legacy fatbin", FATBIN_LEGACY),
            ("pom_mine_mod_nextgen", "nextgen fatbin", FATBIN_NEXTGEN),
        ]
    };

    for (module_name, label, fatbin) in fatbin_candidates {
        match LoadedPomKernel::from_fatbin(cuda.clone(), label, fatbin) {
            Ok(kernel) => {
                let cc = gpu_compute_capability(device_id);
                if let Some((major, minor)) = cc {
                    info!(
                        "PoM[gpu{} cc{}.{}]: startup loaded {} via {}",
                        device_id,
                        major,
                        minor,
                        label,
                        module_name,
                    );
                } else {
                    info!("PoM[gpu{}]: startup loaded {} via {}", device_id, label, module_name);
                }
                set_gpu_kernel_info(device_id, cc, label, module_name);
                return Ok(kernel);
            }
            Err(e) => {
                warn!("PoM[gpu{}]: {} load failed: {}", device_id, label, e);
            }
        }
    }

    for (module_name, label, ptx) in POM_PTX_CANDIDATES {
        match LoadedPomKernel::from_ptx(cuda.clone(), label, ptx) {
            Ok(kernel) => {
                let cc = gpu_compute_capability(device_id);
                if let Some((major, minor)) = cc {
                    info!(
                        "PoM[gpu{} cc{}.{}]: startup loaded {} PTX fallback via {}",
                        device_id,
                        major,
                        minor,
                        label,
                        module_name,
                    );
                } else {
                    info!("PoM[gpu{}]: startup loaded {} PTX fallback via {}", device_id, label, module_name);
                }
                set_gpu_kernel_info(
                    device_id,
                    cc,
                    &format!("{} PTX fallback", label),
                    module_name,
                );
                return Ok(kernel);
            }
            Err(e) => {
                warn!("PoM[gpu{}]: {} PTX load failed: {}", device_id, label, e);
            }
        }
    }

    Err(candle_core::Error::Msg(
        "PoM GPU: no compatible PTX image for this device/driver".into(),
    ))
}

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// Total VRAM (MB) of every CUDA device, in **CUDA device order** — the same ordering
/// `Device::new_cuda(id)` uses — so an entry `(id, mb)` is the VRAM of the device the miner would
/// mine/serve on for that `id`. Sourced from the CUDA driver, NOT nvidia-smi: nvidia-smi orders by
/// PCI position, which disagrees with CUDA's default `FASTEST_FIRST` ordering on a mixed rig, so a
/// line-order mapping would read the wrong card's VRAM. Returns an empty vec when no CUDA driver is
/// present (CPU-only / AMD hosts). Never panics — a driver-load failure inside cudarc is caught and
/// treated as "no devices".
pub fn query_all_gpus_vram() -> Vec<(usize, u64)> {
    use candle_core::cuda_backend::cudarc::driver::result;
    std::panic::catch_unwind(|| {
        if result::init().is_err() {
            return Vec::new();
        }
        let count = result::device::get_count().unwrap_or(0);
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for ordinal in 0..count {
            let Ok(dev) = result::device::get(ordinal) else {
                continue;
            };
            // SAFETY: `dev` is a valid device handle just returned by `device::get(ordinal)`.
            if let Ok(bytes) = unsafe { result::device::total_mem(dev) } {
                out.push((ordinal as usize, (bytes / (1024 * 1024)) as u64));
            }
        }
        out
    })
    .unwrap_or_default()
}

pub struct PomGpuMiner {
    stream: Arc<CudaStream>,
    kernel: LoadedPomKernel,
    bases_dev: CudaSlice<u64>,
    prefix_dev: CudaSlice<u64>,
    t_count: u32,
    n_total_chunks: u64,
    _tensors: Vec<QTensor>, // raw-loaded tensors kept alive so the gather pointers stay valid
    _shared: Vec<Arc<QTensor>>, // shared-with-inference tensors kept alive (zero-dup, Option C)
    _uploads: Vec<CudaSlice<u8>>, // host-resident llama tensors we uploaded ourselves (kept alive)
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into candle on a specific CUDA device, build the gather
    /// index, load the kernel.
    pub fn load(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — matches pom-rt-builder / the node R_T

        let mut tensors: Vec<QTensor> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        for name in &names {
            let qt = content.tensor(&mut file, name, &device)?;
            let chunks = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
            if chunks == 0 {
                tensors.push(qt);
                continue;
            }
            bases.push(qt.device_ptr()? as usize as u64);
            prefix.push(prefix.last().unwrap() + chunks);
            tensors.push(qt);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }

        let bases_dev = cuda.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = cuda.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        // Load the best prebuilt module for this card and keep the raw CUfunction cached.
        let kernel = select_pom_kernel(cuda.clone(), device_id)?;

        Ok(Self {
            stream,
            kernel,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _tensors: tensors,
            _shared: Vec::new(),
            _uploads: Vec::new(),
        })
    }

    /// Zero-dup load (Option C): build the gather over the SAME canonical name-sorted layout as
    /// `R_T`, but for each tensor reuse the inference engine's resident VRAM buffer when it holds
    /// it quantized (`shared`, the big matrices) instead of loading a second copy. Only the
    /// dequantized-in-inference tensors (token_embd, norms) are read raw here — small. `device`
    /// MUST be the same candle device the `shared` tensors live on (pointers are context-bound).
    pub fn load_shared(
        gguf_path: &str,
        device: &Device,
        shared: &std::collections::HashMap<String, Arc<QTensor>>,
    ) -> candle_core::Result<Self> {
        let cuda = match device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: shared load requires a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — must match pom-rt-builder / the node R_T

        let mut raw: Vec<QTensor> = Vec::new();
        let mut kept_shared: Vec<Arc<QTensor>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut shared_hits = 0usize;
        for name in &names {
            let (ptr, chunks) = if let Some(qt) = shared.get(name) {
                // Matrix already resident for inference → reuse its buffer (zero dup).
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                let p = qt.device_ptr()? as usize as u64;
                kept_shared.push(qt.clone());
                shared_hits += 1;
                (p, c)
            } else {
                // Dequantized-in-inference (token_embd, norms): read the raw quantized bytes.
                let qt = content.tensor(&mut file, name, device)?;
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                if c == 0 {
                    raw.push(qt);
                    continue;
                }
                let p = qt.device_ptr()? as usize as u64;
                raw.push(qt);
                (p, c)
            };
            if chunks == 0 {
                continue;
            }
            bases.push(ptr);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: shared load produced 0 chunks".into()));
        }
        info!("PoM zero-dup gather: {} shared tensors, {} raw-loaded, N={} chunks", shared_hits, raw.len(), n_total_chunks);

        let bases_dev = cuda.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = cuda.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let device_id = cuda_gpu_id(device).ok_or_else(|| {
            candle_core::Error::Msg("PoM GPU: shared load requires CUDA device id".into())
        })?;
        let kernel = select_pom_kernel(cuda.clone(), device_id)?;

        Ok(Self {
            stream,
            kernel,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _tensors: raw,
            _shared: kept_shared,
            _uploads: Vec::new(),
        })
    }

    /// Phase-2 zero-dup over the IN-PROCESS llama.cpp engine (candle hosts nothing): build the
    /// gather straight over the engine's resident device tensors in canonical name-sorted order
    /// (the wrapper pre-sorts; byte-identity to the on-disk GGUF proven by
    /// `tools/llama_zerodup_spike`). Host-resident tensors (e.g. `token_embd` on the CPU buffer)
    /// get a small device upload of our own. candle's `CudaDevice` here is pure CUDA plumbing
    /// (context/stream/kernel). `tier` selects the host possession index for the consensus byte-gate.
    pub fn load_llama(device_id: usize, tier: u8) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();
        let ts = crate::llama_engine::tensors()
            .ok_or_else(|| candle_core::Error::Msg("PoM GPU: llama engine tensors unavailable".into()))?;
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut uploads: Vec<CudaSlice<u8>> = Vec::new();
        let mut n_uploaded = 0usize;
        for (_name, ptr, nbytes, is_dev) in &ts {
            let chunks = (nbytes / CHUNK_BYTES) as u64;
            if chunks == 0 {
                continue;
            }
            let base = if *is_dev {
                *ptr
            } else {
                // Host-resident in ggml (CPU buffer): the walk needs device memory — upload our own
                // copy of the raw bytes (identical to the GGUF bytes, same as the pointer).
                let host: &[u8] = unsafe { std::slice::from_raw_parts(*ptr as *const u8, *nbytes) };
                let dev = cuda.clone_htod(host).map_err(candle_core::Error::wrap)?;
                let p = dev.device_ptr(&stream).0 as u64;
                uploads.push(dev);
                n_uploaded += 1;
                p
            };
            bases.push(base);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: llama engine produced 0 chunks".into()));
        }
        info!(
            "PoM llama zero-dup gather: {} tensors ({} host-resident uploaded), N={} chunks",
            bases.len(), n_uploaded, n_total_chunks
        );
        // BYTE GATE (consensus safety): the pool does not deep-verify every share, so a wrong
        // gather would mine garbage silently. Read back evenly-spaced chunks from the llama-owned
        // device memory and compare them byte-for-byte against the host index (GGUF pread) — any
        // mismatch refuses to mine. Full-model byte-identity for this llama build was proven once
        // by `tools/llama_zerodup_spike`; this guards every startup against regressions.
        if let Some(idx) = crate::pom::active_index_for_tier(tier) {
            if idx.n_chunks == n_total_chunks {
                use candle_core::cuda_backend::cudarc::driver::result as cures;
                let samples = 128u64;
                for kk in 0..=samples {
                    let off = if kk == samples { n_total_chunks - 1 } else { kk * (n_total_chunks / (samples + 1)) };
                    let j = prefix.partition_point(|&p| p <= off) - 1;
                    let dev_addr = bases[j] + (off - prefix[j]) * CHUNK_BYTES as u64;
                    let mut got = [0u8; CHUNK_BYTES];
                    unsafe { cures::memcpy_dtoh_sync(&mut got, dev_addr).map_err(candle_core::Error::wrap)? };
                    let want = idx.read_chunk_bytes(off);
                    if got != want {
                        return Err(candle_core::Error::Msg(format!(
                            "PoM llama byte gate FAILED at chunk {off} — llama-resident bytes differ from the GGUF; refusing to mine"
                        )));
                    }
                }
                info!("PoM llama byte gate: {} sampled chunks match the host index byte-for-byte.", samples + 1);
            }
        }

        let bases_dev = cuda.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = cuda.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let kernel = select_pom_kernel(cuda.clone(), device_id)?;

        Ok(Self {
            stream,
            kernel,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _tensors: Vec::new(),
            _shared: Vec::new(),
            _uploads: uploads,
        })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `target_le` is the header's compact target as 32 LE bytes.
    /// `h3` salts the pph words host-side (POM_H3_PPH_SALT) — the kernel itself is era-agnostic,
    /// it folds whatever words it receives, so no PTX change at the H3 gate.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool) -> candle_core::Result<Option<u64>> {
        let p_words = crate::pom::pph_words_for_era(pre_pow_hash, h3);
        self.kernel.launch(
            &self.stream,
            &self.bases_dev,
            &self.prefix_dev,
            self.t_count,
            self.n_total_chunks,
            &p_words,
            timestamp,
            target_le,
            start,
            batch,
        )
    }
}

// Per-GPU PoM miners. Host-side WeightIndex remains shared; only the CUDA-resident worker state
// is duplicated per device. This avoids all workers contending over a single GPU0-bound miner.
fn miners() -> &'static Mutex<HashMap<u32, Arc<PomGpuMiner>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

// Guards the one-time shared host index build. All workers may race into PoM activation, but the
// heavy GGUF -> WeightIndex build must happen exactly once for the process.
fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

/// Install the GPU miner for a specific CUDA device.
pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, Arc::new(m));
    }
}

/// Removes only `device_id`'s entry from a `device -> miner` map, leaving every other device's
/// entry untouched. Pulled out as a tiny generic helper (over the map's value type) purely so
/// this scoping behavior is unit-testable without a real, CUDA-backed `PomGpuMiner` — production
/// always calls it through `uninstall` against `HashMap<u32, Arc<PomGpuMiner>>`.
fn remove_device_entry<T>(map: &mut HashMap<u32, T>, device_id: u32) {
    map.remove(&device_id);
}

/// Drop the GPU miner for `device_id` only, releasing its hold on that device's mining-model VRAM
/// (shared Arcs + gather) so the inference engine can load another model there. Mining on that
/// device is paused during inference anyway.
///
/// Scoped to a single device on purpose: only the device colocated with inference (CUDA device 0
/// — see the `Device::new_cuda(0)` call in `slm::load_and_run_inference`) ever shares VRAM with
/// the inference engine via `load_shared`'s zero-dup path, or otherwise needs to make room for an
/// inference model swap. Other devices in a multi-GPU rig run fully standalone `PomGpuMiner`s
/// (`PomGpuMiner::load`) that never touch the inference engine's VRAM. A previous version of this
/// function called `g.clear()`, dropping every device's resident miner on every inference model
/// swap — needlessly forcing GPU1+ rigs to fully reload their GGUF from disk and rebuild the
/// gather index (`ensure_installed_inner`'s own doc comment calls this reload "Heavy") even though
/// nothing about them changed.
pub fn uninstall(device_id: u32) {
    if let Ok(mut g) = miners().lock() {
        remove_device_entry(&mut g, device_id);
    }
}

/// Whether the GPU miner is currently installed for `device_id`.
pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

/// True while the GPU miner is being (re)built — a heavy one-time model load that blocks the
/// mining worker. The PoW stall watchdog treats this like an inference pause, not a crash.
static LOADING: AtomicUsize = AtomicUsize::new(0);

/// Whether a PoM model load/rebuild is in progress (worker intentionally paused, not stalled).
pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

/// Convenience: search a nonce batch via the installed miner for a specific device.
pub fn mine(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool) -> Option<u64> {
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch, h3).ok().flatten()
}

/// Per-GPU mining-tier identity for rebuilds: `device_id -> (model_id, gguf_path)`. A heterogeneous
/// rig mines a different tier per GPU (the highest its VRAM holds), so this is keyed by device rather
/// than a single process-wide tier.
static MINING_TIERS: OnceLock<Mutex<HashMap<u32, ([u8; 32], String)>>> = OnceLock::new();

fn mining_tiers() -> &'static Mutex<HashMap<u32, ([u8; 32], String)>> {
    MINING_TIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record a GPU's mining tier so its miner can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(device_id: u32, model_id: [u8; 32], gguf_path: String) {
    if let Ok(mut g) = mining_tiers().lock() {
        g.insert(device_id, (model_id, gguf_path));
    }
}

/// Ensure the GPU miner is installed; if an inference evicted the mining model, reload it
/// (resident again) and rebuild the zero-dup gather. Heavy (model reload) but only when needed —
/// inference has priority, so mining reloads its model when it next gets the GPU. Returns true if
/// the miner is ready to mine.
pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    // Flag the heavy load so the stall watchdog stays benign while the worker is blocked here.
    LOADING.fetch_add(1, Ordering::Relaxed);
    let ok = ensure_installed_inner(device_id, daa);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

/// PoM tier index of the mining model at a given block DAA. Recomputed per block (not frozen at
/// index-build time) so the tier reindexing at the very-light hardfork (H2) is applied at the
/// exact boundary — e.g. Gemma 0→1 — rather than from a stale build-time value.
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let model_id = mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)?;
    crate::models::pom_tier_index(&model_id, daa)
}

/// The CUDA device that mines `model_id` (from the per-GPU tier assignment), if any. Inference for a
/// model is routed to the device that already holds it, so only that GPU pauses mining and the walk
/// can share the resident weights (zero-dup). Returns the lowest matching `device_id` when several
/// GPUs mine the same tier; `None` when no GPU is assigned this model.
pub fn device_for_model(model_id: &[u8; 32]) -> Option<u32> {
    let g = mining_tiers().lock().ok()?;
    g.iter().filter(|(_, (id, _))| id == model_id).map(|(dev, _)| *dev).min()
}

/// UI helper: current mining-model label by CUDA device id.
/// Returns entries sorted by device id.
pub fn list_mining_model_labels() -> Vec<(u32, String)> {
    let snapshot: Vec<(u32, [u8; 32])> = match mining_tiers().lock() {
        Ok(g) => g.iter().map(|(dev, (id, _))| (*dev, *id)).collect(),
        Err(_) => return Vec::new(),
    };

    let mut out: Vec<(u32, String)> = snapshot
        .into_iter()
        .map(|(dev, model_id)| {
            let label = crate::models::REGISTRY
                .iter()
                .copied()
                .find(|m| m.model_id == model_id)
                .map(|m| m.dir_name.to_string())
                .unwrap_or_else(|| hex::encode(model_id)[..8].to_string());
            (dev, label)
        })
        .collect();
    out.sort_by_key(|(dev, _)| *dev);
    out
}

/// Models that OOM'd when loading on a given GPU: `(device_id, model_id)`. Once banlisted, that GPU
/// never retries that model (avoids a hot-spin reloading a model that doesn't fit); the OOM handler
/// downgrades the GPU to a smaller downloaded tier instead.
static OOM_BANLIST: OnceLock<Mutex<HashSet<(u32, [u8; 32])>>> = OnceLock::new();

fn oom_banlist() -> &'static Mutex<HashSet<(u32, [u8; 32])>> {
    OOM_BANLIST.get_or_init(|| Mutex::new(HashSet::new()))
}

fn is_oom_banlisted(device_id: u32, model_id: &[u8; 32]) -> bool {
    oom_banlist().lock().map(|g| g.contains(&(device_id, *model_id))).unwrap_or(false)
}

fn oom_banlist_add(device_id: u32, model_id: [u8; 32]) {
    if let Ok(mut g) = oom_banlist().lock() {
        g.insert((device_id, model_id));
    }
}

/// After a GPU fails to load its assigned tier (OOM), reassign it to the largest **already-downloaded**
/// PoM model strictly smaller than the failed one that hasn't itself been banlisted on this GPU — so a
/// card whose VRAM estimate was optimistic (driver overhead + KV cache + fragmentation) mines a
/// smaller tier instead of idling. Returns true if a downgrade was applied. No extra prefetch is
/// needed: the candidate set is the served union (a mixed rig already downloaded the smaller tiers).
fn downgrade_after_oom(device_id: u32, failed_model: &[u8; 32], daa: u64) -> bool {
    let Some(failed_tier) = crate::models::pom_tier_index(failed_model, daa) else {
        return false;
    };
    let pick = crate::slm::served_pom_specs()
        .into_iter()
        .filter_map(|s| crate::models::pom_tier_index(&s.model_id, daa).map(|t| (t, s)))
        .filter(|(t, s)| *t < failed_tier && !is_oom_banlisted(device_id, &s.model_id))
        .max_by_key(|(t, _)| *t);
    match pick {
        Some((tier, spec)) => {
            let gguf = crate::slm::gguf_path_for(spec).to_string_lossy().into_owned();
            info!("PoM[gpu{}]: OOM on tier {} — downgrading to tier {} ({}).", device_id, failed_tier, tier, spec.name);
            set_mining_tier(device_id, spec.model_id, gguf);
            true
        }
        None => {
            log::warn!("PoM[gpu{}]: OOM and no smaller downloaded tier available — this GPU will not mine PoM (lower the tier flag or add VRAM).", device_id);
            false
        }
    }
}

/// CUDA ordinal of a candle device (None if not CUDA) — used to check whether the inference
/// engine's resident model lives on the same GPU as the PoM miner we're about to install, before
/// sharing its tensors in place.
fn cuda_gpu_id(d: &Device) -> Option<usize> {
    match d.location() {
        candle_core::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
        _ => None,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum MinerLoadFailureKind {
    PtxIncompatible,
    OomLikely,
    Other,
}

fn classify_miner_load_error(err: &str) -> MinerLoadFailureKind {
    let s = err.to_ascii_lowercase();
    if s.contains("invalid_ptx")
        || s.contains("invalid ptx")
        || s.contains("ptx") && (s.contains("compatible") || s.contains("no kernel image"))
    {
        return MinerLoadFailureKind::PtxIncompatible;
    }
    if s.contains("out of memory")
        || s.contains("cuda_error_out_of_memory")
        || s.contains("memory allocation")
        || s.contains("alloc") && s.contains("failed")
    {
        return MinerLoadFailureKind::OomLikely;
    }
    MinerLoadFailureKind::Other
}

fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {
    let (model_id, gguf) = match mining_tiers().lock().ok().and_then(|g| g.get(&device_id).cloned()) {
        Some(x) => x,
        None => return false,
    };
    // This GPU's tier at the current block DAA (recomputed per block, H2-gated).
    let tier = match crate::models::pom_tier_index(&model_id, daa) {
        Some(t) => t,
        None => return false,
    };
    if is_oom_banlisted(device_id, &model_id) {
        return false; // this model OOM'd on this GPU before — don't retry (avoids a hot reload spin).
    }
    // Build THIS tier's possession index once (host, heavy) — deferred from boot so the pre-PoM
    // legacy phase starts immediately, and keyed by tier so a mixed rig builds one index per
    // distinct tier it mines (shared across every GPU on that tier).
    if crate::pom::active_index_for_tier(tier).is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_index_for_tier(tier).is_none() {
            info!("PoM: building host weight index for tier {} (gpu{}) - this can take a while...", tier, device_id);
            match crate::pom::WeightIndex::build_from_gguf(&gguf) {
                Ok(idx) => {
                    info!("PoM: tier {} host index ready — N={} chunks", tier, idx.n_chunks);
                    crate::pom::set_index(tier, idx);
                }
                Err(e) => {
                    log::error!("PoM: host index build failed for tier {} on gpu{}: {}", tier, device_id, e);
                    return false;
                }
            }
        }
    }
    // One CUDA-resident PoM worker per GPU. This avoids all workers contending for a single
    // GPU0-bound miner object while still sharing the host-side index across the process.
    //
    // Zero-dup on the inference GPU: if the inference engine holds THIS exact model resident on
    // THIS device (split loader + `pom_force_split`), the walk shares its quantized tensors in
    // place (`load_shared`) rather than loading a second full VRAM copy — saving ~one model's
    // worth of VRAM on the serving GPU. Mining-only GPUs (no resident inference model to share)
    // fall back to a standalone copy. The N-guard below validates the gather against the host
    // index on every path, so a mismatch refuses to mine rather than producing bad proofs.
    // Load the miner (zero-dup on the inference GPU, else a standalone copy). A load OOM surfaces as
    // an Err or, in cudarc, a panic; catch both so the OOM handler can banlist + downgrade instead of
    // crashing the mining thread or hot-spinning on a model that doesn't fit this GPU.
    // Phase 2 (candle-independence): if the in-process llama.cpp engine can host THIS model on the
    // GPU that mines it (the inference GPU), the walk gathers over ITS resident tensors and candle
    // hosts nothing. `ensure_loaded` is a process-global singleton — only the inference GPU brings
    // it up; every other mining GPU falls through to the candle paths below. Absent `.so` ⇒ false,
    // so the candle behaviour is byte-identical to before.
    let inference_gpu = device_for_model(&model_id).unwrap_or(0);
    let mut use_llama =
        device_id == inference_gpu && crate::llama_engine::ensure_loaded(&gguf, device_id as usize);
    // BYTE-COMPAT GATE: llama.cpp repacks some architectures on load (e.g. Gemma-3 materialises a
    // separate output.weight from its tied embeddings), so its resident chunk count differs from
    // the canonical GGUF the walk MUST gather and R_T pins. When that happens the zero-dup walk is
    // impossible — free llama's VRAM and fall back to candle for the walk on this card.
    if use_llama {
        let host_n = crate::pom::active_index_for_tier(tier).map(|i| i.n_chunks);
        let llama_n = crate::llama_engine::tensors().map(|ts| {
            ts.iter().map(|(_, _, nbytes, _)| (*nbytes / CHUNK_BYTES) as u64).sum::<u64>()
        });
        if let (Some(hn), Some(ln)) = (host_n, llama_n) {
            if ln != hn {
                info!(
                    "PoM[gpu{}]: llama-resident layout N={} != canonical N={} (llama repacks this model arch) — using candle for the walk on this card.",
                    device_id, ln, hn
                );
                crate::llama_engine::unload();
                use_llama = false;
            }
        }
    }
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if use_llama {
            info!("PoM[gpu{}]: zero-dup — walking the llama.cpp engine's resident weights (candle dormant)", device_id);
            PomGpuMiner::load_llama(device_id as usize, tier)
        } else {
            match crate::slm::pom_shared(&model_id) {
                Some((inf_dev, shared)) if cuda_gpu_id(&inf_dev) == Some(device_id as usize) => {
                    info!("PoM[gpu{}]: zero-dup — sharing the inference engine's resident weights (no 2nd VRAM copy)", device_id);
                    PomGpuMiner::load_shared(&gguf, &inf_dev, &shared)
                }
                _ => PomGpuMiner::load(&gguf, device_id as usize),
            }
        }
    }));
    let gm = match loaded {
        Ok(Ok(gm)) => gm,
        Ok(Err(e)) => {
            let e_msg = e.to_string();
            match classify_miner_load_error(&e_msg) {
                MinerLoadFailureKind::PtxIncompatible => {
                    log::error!(
                        "PoM[gpu{}]: PTX incompatibility while loading miner (not OOM): {}. \
                         Check driver/PTX compatibility; skipping OOM downgrade.",
                        device_id,
                        e_msg
                    );
                }
                MinerLoadFailureKind::OomLikely => {
                    log::error!(
                        "PoM[gpu{}]: device miner build failed (OOM likely): {} — banlisting this model and downgrading.",
                        device_id,
                        e_msg
                    );
                    oom_banlist_add(device_id, model_id);
                    downgrade_after_oom(device_id, &model_id, daa);
                }
                MinerLoadFailureKind::Other => {
                    log::error!(
                        "PoM[gpu{}]: device miner build failed (non-OOM): {} — not applying OOM downgrade.",
                        device_id,
                        e_msg
                    );
                }
            }
            return false;
        }
        Err(_) => {
            log::error!("PoM[gpu{}]: device miner load panicked (likely OOM) — banlisting this model and downgrading.", device_id);
            oom_banlist_add(device_id, model_id);
            downgrade_after_oom(device_id, &model_id, daa);
            return false;
        }
    };
    let n = gm.n_chunks();
    // N-guard: the gather must match the host index, else blocks would be rejected.
    if let Some(idx) = crate::pom::active_index_for_tier(tier) {
        if n != idx.n_chunks {
            log::error!("PoM[gpu{}]: gather N={} != tier {} index N={} — refusing to mine", device_id, n, tier, idx.n_chunks);
            return false;
        }
    }
    install(device_id, gm);
    info!("PoM[gpu{}]: GPU miner ready — N={} chunks resident (matches shared index)", device_id, n);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise `remove_device_entry` directly with a dummy value type, rather than going
    // through `install`/`uninstall`, because `PomGpuMiner` can only be constructed via `load`/
    // `load_shared`, both of which require real CUDA hardware (`Device::new_cuda`) unavailable in
    // CI/unit-test environments. `remove_device_entry` holds the entire scoping logic that
    // `uninstall` delegates to, so this still covers the behavior that matters: only the targeted
    // device's entry is removed, every other device's entry survives untouched.

    #[test]
    fn remove_device_entry_only_clears_target_device() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(0, "gpu0-miner");
        map.insert(1, "gpu1-miner");
        map.insert(2, "gpu2-miner");

        remove_device_entry(&mut map, 0);

        assert!(!map.contains_key(&0));
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
        assert_eq!(map.get(&2), Some(&"gpu2-miner"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn remove_device_entry_on_missing_device_is_a_no_op() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(1, "gpu1-miner");

        remove_device_entry(&mut map, 0);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
    }
}
