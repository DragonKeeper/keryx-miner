//! Block task of the mining devices: window seeds, the task trace of each device and its root.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use keryx_verifiable_inference::block_task::window;

use crate::trace::ResponseTrace;

/// The task trace a device commits for one window.
pub struct DeviceTask {
    pub window: u64,
    pub seed: [u8; 32],
    pub model_id: [u8; 32],
    pub tier: u8,
    pub trace: ResponseTrace,
}

impl DeviceTask {
    pub fn root(&self) -> [u8; 32] {
        self.trace.record.trace_root
    }
}

static SEEDS: Mutex<BTreeMap<u64, [u8; 32]>> = Mutex::new(BTreeMap::new());
static TASKS: Mutex<BTreeMap<(u32, u64), Arc<DeviceTask>>> = Mutex::new(BTreeMap::new());
static FAILED: Mutex<BTreeMap<u32, u64>> = Mutex::new(BTreeMap::new());

/// Whether a block at `daa` carries the task.
pub fn active(daa: u64) -> bool {
    daa >= crate::pom::block_task_activation_daa()
}

/// Records the seed block of window `w`.
pub fn set_seed(w: u64, seed: [u8; 32]) {
    let mut s = SEEDS.lock().unwrap_or_else(|p| p.into_inner());
    s.insert(w, seed);
    while s.len() > 4 {
        s.pop_first();
    }
}

pub fn seed(w: u64) -> Option<[u8; 32]> {
    SEEDS.lock().unwrap_or_else(|p| p.into_inner()).get(&w).copied()
}

/// The task of `device` (mining `model_id` at `tier`) for the window of a block at `daa`, traced on
/// first use. `None` while the window seed is unknown or the task cannot be traced.
pub fn task_for(device: u32, model_id: &[u8; 32], tier: u8, daa: u64) -> Option<Arc<DeviceTask>> {
    let w = window(daa)?;
    if let Some(t) = TASKS.lock().unwrap_or_else(|p| p.into_inner()).get(&(device, w)) {
        if t.model_id == *model_id && t.tier == tier {
            return Some(Arc::clone(t));
        }
    }
    let seed = seed(w)?;
    if FAILED.lock().unwrap_or_else(|p| p.into_inner()).get(&device) == Some(&w) {
        return None;
    }
    let Some(trace) = crate::llm::run_block_task(model_id, tier, &seed) else {
        log::warn!("Block task: gpu{} cannot trace the task of window {} — not mining this window", device, w);
        FAILED.lock().unwrap_or_else(|p| p.into_inner()).insert(device, w);
        return None;
    };
    let task = Arc::new(DeviceTask { window: w, seed, model_id: *model_id, tier, trace });
    let mut tasks = TASKS.lock().unwrap_or_else(|p| p.into_inner());
    tasks.retain(|&(d, tw), _| d != device || tw + 1 >= w);
    tasks.insert((device, w), Arc::clone(&task));
    Some(task)
}
