//! Thread-safe C ABI. Each call takes the SDK lock, builds the engine it needs, and drops it.

use std::sync::{Mutex, MutexGuard};

use crate::bitnet::{self, argmax};
use crate::gpu::{GpuEngine, GPU_ADD, GPU_XOR};
use crate::ising::{graph_from_raw, score_cut};
use crate::scenarios::eval_script;

const VERSION: &[u8] = b"AetherDCC Engine v2.0\0";

fn lock() -> MutexGuard<'static, ()> {
    static SDK: Mutex<()> = Mutex::new(());
    SDK.lock().unwrap_or_else(|poison| poison.into_inner())
}

#[no_mangle]
pub extern "C" fn dcc_sdk_version() -> *const i8 {
    VERSION.as_ptr().cast()
}

#[no_mangle]
pub unsafe extern "C" fn dcc_sdk_infer_digit10(
    pixels: *const f32,
    class_out: *mut i32,
    scores_out: *mut i32,
    latency_us: *mut f64,
) -> i32 {
    if pixels.is_null() || class_out.is_null() || scores_out.is_null() || latency_us.is_null() {
        return -1;
    }
    let _guard = lock();
    let mut image = [0.0f32; 64];
    std::ptr::copy_nonoverlapping(pixels, image.as_mut_ptr(), 64);
    let mut gpu = GpuEngine::new();
    let sample = gpu.bitnet(&image);
    let class = argmax(sample.scores);
    *class_out = class;
    std::ptr::copy_nonoverlapping(sample.scores.as_ptr(), scores_out, 10);
    *latency_us = sample.dispatch_us;
    let host = bitnet::reference(&image);
    i32::from(host.scores != sample.scores || host.class != class)
}

#[no_mangle]
pub unsafe extern "C" fn dcc_sdk_ghost_cipher(
    block: *const u8,
    key: *const u8,
    salt: *const u8,
    out: *mut u8,
    latency_us: *mut f64,
) -> i32 {
    if block.is_null() || key.is_null() || salt.is_null() || out.is_null() || latency_us.is_null() {
        return -1;
    }
    let _guard = lock();
    let mut plain = [0u8; 16];
    let mut key_bytes = [0u8; 16];
    let mut salt_bytes = [0u8; 16];
    std::ptr::copy_nonoverlapping(block, plain.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(key, key_bytes.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(salt, salt_bytes.as_mut_ptr(), 16);
    let mut gpu = GpuEngine::new();
    let (xored, first) = gpu.simd_timed(GPU_XOR, &plain, &key_bytes);
    let (cipher, second) = gpu.simd_timed(GPU_ADD, &xored, &salt_bytes);
    std::ptr::copy_nonoverlapping(cipher.as_ptr(), out, 16);
    *latency_us = first + second;
    let mut errors = 0i32;
    for lane in 0..16 {
        let expect = xored[lane].wrapping_add(salt_bytes[lane]);
        errors += (cipher[lane] ^ expect).count_ones() as i32;
    }
    errors
}

#[no_mangle]
pub unsafe extern "C" fn dcc_sdk_eval_script(script_id: i32, result: *mut i32) -> i32 {
    if result.is_null() {
        return -1;
    }
    let _guard = lock();
    match eval_script(script_id) {
        Ok(value) => {
            *result = value;
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn dcc_sdk_solve_ising64(
    neighbors: *const i32,
    coupling: *const i32,
    bias: *const i32,
    spins_out: *mut i32,
    energy_out: *mut i32,
    latency_us: *mut f64,
) -> i32 {
    if neighbors.is_null()
        || coupling.is_null()
        || bias.is_null()
        || spins_out.is_null()
        || energy_out.is_null()
        || latency_us.is_null()
    {
        return -1;
    }
    let _guard = lock();
    let neighbor_buf = std::slice::from_raw_parts(neighbors, 64 * 8);
    let couple_buf = std::slice::from_raw_parts(coupling, 64 * 8);
    let bias_buf = std::slice::from_raw_parts(bias, 64);
    let graph = match graph_from_raw(neighbor_buf, couple_buf, bias_buf) {
        Ok(graph) => graph,
        Err(_) => return -1,
    };
    let mut gpu = GpuEngine::new();
    let sample = gpu.ising(&graph);
    let (_cut, energy) = score_cut(&graph, &sample.spins);
    std::ptr::copy_nonoverlapping(sample.spins.as_ptr(), spins_out, 64);
    *energy_out = energy;
    *latency_us = sample.dispatch_us;
    0
}
