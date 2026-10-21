//! Thread-safe C ABI for AetherDCC Engine v2.0.
//! Provides both high-performance context handles (dcc_context_*) and
//! backward-compatible warmed singleton API endpoints.

use std::sync::{Mutex, MutexGuard};

use crate::bitnet::{self, argmax};
use crate::gpu::{GpuEngine, GPU_ADD, GPU_XOR};
use crate::ising::{graph_from_raw, score_cut};
use crate::scenarios::eval_script;

const VERSION: &[u8] = b"AetherDCC Engine v2.0\0";

/// High-performance stateful context holding a warmed compute device
pub struct DccContext {
    pub gpu: GpuEngine,
}

fn cached_gpu() -> MutexGuard<'static, Option<GpuEngine>> {
    static CACHED: Mutex<Option<GpuEngine>> = Mutex::new(None);
    let mut guard = CACHED.lock().unwrap_or_else(|poison| poison.into_inner());
    if guard.is_none() {
        if let Ok(engine) = GpuEngine::try_new() {
            *guard = Some(engine);
        }
    }
    guard
}

#[no_mangle]
pub extern "C" fn dcc_sdk_version() -> *const i8 {
    VERSION.as_ptr().cast()
}

// =========================================================================
// High-Performance Context API (Sub-20µs Steady State)
// =========================================================================

#[no_mangle]
pub extern "C" fn dcc_context_create() -> *mut DccContext {
    match GpuEngine::try_new() {
        Ok(gpu) => Box::into_raw(Box::new(DccContext { gpu })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn dcc_context_destroy(ctx: *mut DccContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

#[no_mangle]
pub unsafe extern "C" fn dcc_context_infer_digit10(
    ctx: *mut DccContext,
    pixels: *const f32,
    class_out: *mut i32,
    scores_out: *mut i32,
    latency_us: *mut f64,
) -> i32 {
    if ctx.is_null() || pixels.is_null() || class_out.is_null() || scores_out.is_null() || latency_us.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;
    let mut image = [0.0f32; 64];
    std::ptr::copy_nonoverlapping(pixels, image.as_mut_ptr(), 64);
    let sample = ctx.gpu.bitnet(&image);
    let class = argmax(sample.scores);
    *class_out = class;
    std::ptr::copy_nonoverlapping(sample.scores.as_ptr(), scores_out, 10);
    *latency_us = sample.dispatch_us;
    let host = bitnet::reference(&image);
    i32::from(host.scores != sample.scores || host.class != class)
}

#[no_mangle]
pub unsafe extern "C" fn dcc_context_ghost_cipher(
    ctx: *mut DccContext,
    block: *const u8,
    key: *const u8,
    salt: *const u8,
    out: *mut u8,
    latency_us: *mut f64,
) -> i32 {
    if ctx.is_null() || block.is_null() || key.is_null() || salt.is_null() || out.is_null() || latency_us.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;
    let mut plain = [0u8; 16];
    let mut key_bytes = [0u8; 16];
    let mut salt_bytes = [0u8; 16];
    std::ptr::copy_nonoverlapping(block, plain.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(key, key_bytes.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(salt, salt_bytes.as_mut_ptr(), 16);
    let (xored, first) = ctx.gpu.simd_timed(GPU_XOR, &plain, &key_bytes);
    let (cipher, second) = ctx.gpu.simd_timed(GPU_ADD, &xored, &salt_bytes);
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
pub unsafe extern "C" fn dcc_context_solve_ising64(
    ctx: *mut DccContext,
    neighbors: *const i32,
    coupling: *const i32,
    bias: *const i32,
    spins_out: *mut i32,
    energy_out: *mut i32,
    latency_us: *mut f64,
) -> i32 {
    if ctx.is_null()
        || neighbors.is_null()
        || coupling.is_null()
        || bias.is_null()
        || spins_out.is_null()
        || energy_out.is_null()
        || latency_us.is_null()
    {
        return -1;
    }
    let ctx = &mut *ctx;
    let neighbor_buf = std::slice::from_raw_parts(neighbors, 64 * 8);
    let couple_buf = std::slice::from_raw_parts(coupling, 64 * 8);
    let bias_buf = std::slice::from_raw_parts(bias, 64);
    let graph = match graph_from_raw(neighbor_buf, couple_buf, bias_buf) {
        Ok(graph) => graph,
        Err(_) => return -1,
    };
    let sample = ctx.gpu.ising(&graph);
    let (_cut, energy) = score_cut(&graph, &sample.spins);
    std::ptr::copy_nonoverlapping(sample.spins.as_ptr(), spins_out, 64);
    *energy_out = energy;
    *latency_us = sample.dispatch_us;
    0
}

// =========================================================================
// Warmed Backward-Compatible Endpoints (Zero Initialization Penalty)
// =========================================================================

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
    let mut guard = cached_gpu();
    let gpu = match guard.as_mut() {
        Some(engine) => engine,
        None => return -1,
    };
    let mut image = [0.0f32; 64];
    std::ptr::copy_nonoverlapping(pixels, image.as_mut_ptr(), 64);
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
    let mut guard = cached_gpu();
    let gpu = match guard.as_mut() {
        Some(engine) => engine,
        None => return -1,
    };
    let mut plain = [0u8; 16];
    let mut key_bytes = [0u8; 16];
    let mut salt_bytes = [0u8; 16];
    std::ptr::copy_nonoverlapping(block, plain.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(key, key_bytes.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(salt, salt_bytes.as_mut_ptr(), 16);
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
    let mut guard = cached_gpu();
    let gpu = match guard.as_mut() {
        Some(engine) => engine,
        None => return -1,
    };
    let neighbor_buf = std::slice::from_raw_parts(neighbors, 64 * 8);
    let couple_buf = std::slice::from_raw_parts(coupling, 64 * 8);
    let bias_buf = std::slice::from_raw_parts(bias, 64);
    let graph = match graph_from_raw(neighbor_buf, couple_buf, bias_buf) {
        Ok(graph) => graph,
        Err(_) => return -1,
    };
    let sample = gpu.ising(&graph);
    let (_cut, energy) = score_cut(&graph, &sample.spins);
    std::ptr::copy_nonoverlapping(sample.spins.as_ptr(), spins_out, 64);
    *energy_out = energy;
    *latency_us = sample.dispatch_us;
    0
}

// =========================================================================
// Cryptographic SPN Block Cipher API
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn dcc_sdk_ghost_spn_encrypt(
    block: *const u8,
    key: *const u8,
    salt: *const u8,
    out: *mut u8,
    rounds: i32,
) -> i32 {
    if block.is_null() || key.is_null() || salt.is_null() || out.is_null() {
        return -1;
    }
    let mut plain = [0u8; 16];
    let mut key_bytes = [0u8; 16];
    let mut salt_bytes = [0u8; 16];
    std::ptr::copy_nonoverlapping(block, plain.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(key, key_bytes.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(salt, salt_bytes.as_mut_ptr(), 16);
    let cipher = crate::crypto::ghost_spn_encrypt_block(&plain, &key_bytes, &salt_bytes, rounds as usize);
    std::ptr::copy_nonoverlapping(cipher.as_ptr(), out, 16);
    0
}

#[no_mangle]
pub unsafe extern "C" fn dcc_sdk_ghost_spn_decrypt(
    cipher: *const u8,
    key: *const u8,
    salt: *const u8,
    out: *mut u8,
    rounds: i32,
) -> i32 {
    if cipher.is_null() || key.is_null() || salt.is_null() || out.is_null() {
        return -1;
    }
    let mut cipher_bytes = [0u8; 16];
    let mut key_bytes = [0u8; 16];
    let mut salt_bytes = [0u8; 16];
    std::ptr::copy_nonoverlapping(cipher, cipher_bytes.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(key, key_bytes.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(salt, salt_bytes.as_mut_ptr(), 16);
    let plain = crate::crypto::ghost_spn_decrypt_block(&cipher_bytes, &key_bytes, &salt_bytes, rounds as usize);
    std::ptr::copy_nonoverlapping(plain.as_ptr(), out, 16);
    0
}

#[no_mangle]
pub unsafe extern "C" fn dcc_sdk_ghost_sac_score(key: *const u8, salt: *const u8) -> f64 {
    if key.is_null() || salt.is_null() {
        return -1.0;
    }
    let mut key_bytes = [0u8; 16];
    let mut salt_bytes = [0u8; 16];
    std::ptr::copy_nonoverlapping(key, key_bytes.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(salt, salt_bytes.as_mut_ptr(), 16);
    crate::crypto::calculate_sac_percentage(&key_bytes, &salt_bytes)
}
