//! One D3D11 device on Intel UHD 0x9B41. SIMD, vision, and Ising share it.
//! Two devices are never constructed together.

use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use crate::cpu_alu::decode_counts;
use crate::ising::{IsingGraph, ISING_DEG, ISING_NODES, ISING_PUSH, ISING_WAVES};

const D3D_DRIVER_TYPE_UNKNOWN: u32 = 0;
const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
const D3D_FEATURE_LEVEL_11_1: u32 = 0xb100;
const D3D11_SDK_VERSION: u32 = 7;
const D3DCOMPILE_SKIP_OPTIMIZATION: u32 = 1 << 2;
const D3D11_USAGE_DEFAULT: u32 = 0;
const D3D11_USAGE_STAGING: u32 = 3;
const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
const D3D11_BIND_UNORDERED_ACCESS: u32 = 0x80;
const D3D11_CPU_ACCESS_READ: u32 = 0x20000;
const D3D11_RESOURCE_MISC_BUFFER_STRUCTURED: u32 = 0x40;
const D3D11_SRV_DIMENSION_BUFFER: u32 = 1;
const D3D11_UAV_DIMENSION_BUFFER: u32 = 1;
const D3D11_MAP_READ: u32 = 1;

pub const GPU_ADD: u32 = 1;
pub const GPU_SUB: u32 = 2;
pub const GPU_XOR: u32 = 3;
pub const GPU_AND: u32 = 4;
pub const GPU_OR: u32 = 5;

const VISION_PIXELS: usize = 64;
const VISION_CLASSES: usize = 4;
const VISION_MAX_BURST: u32 = 16;
const VISION_PEDESTAL: u32 = (VISION_PIXELS as u32) * VISION_MAX_BURST / 4;

#[link(name = "d3d11", kind = "raw-dylib")]
extern "system" {
    fn D3D11CreateDevice(
        adapter: *mut c_void,
        driver_type: u32,
        software: *mut c_void,
        flags: u32,
        feature_levels: *const u32,
        feature_level_count: u32,
        sdk_version: u32,
        device: *mut *mut c_void,
        feature_level: *mut u32,
        context: *mut *mut c_void,
    ) -> i32;
}

#[link(name = "dxgi", kind = "raw-dylib")]
extern "system" {
    fn CreateDXGIFactory(riid: *const Guid, factory: *mut *mut c_void) -> i32;
}

#[link(name = "d3dcompiler_47", kind = "raw-dylib")]
extern "system" {
    fn D3DCompile(
        src: *const c_void,
        src_len: usize,
        name: *const i8,
        defines: *const c_void,
        include: *mut c_void,
        entry: *const i8,
        target: *const i8,
        flags1: u32,
        flags2: u32,
        code: *mut *mut c_void,
        errors: *mut *mut c_void,
    ) -> i32;
}

#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

const IID_IDXGI_FACTORY: Guid = Guid {
    data1: 0x7b71_66ec,
    data2: 0x21c7,
    data3: 0x44ae,
    data4: [0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69],
};

#[repr(C)]
struct AdapterDesc {
    description: [u16; 128],
    vendor_id: u32,
    device_id: u32,
    sub_sys_id: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    luid_low: u32,
    luid_high: i32,
}

const _: () = assert!(std::mem::size_of::<AdapterDesc>() == 304);

#[repr(C)]
struct BufferDesc {
    byte_width: u32,
    usage: u32,
    bind_flags: u32,
    cpu_access: u32,
    misc_flags: u32,
    stride: u32,
}

#[repr(C)]
struct SrvDesc {
    format: u32,
    view_dimension: u32,
    first_element: u32,
    num_elements: u32,
    _union_tail: [u32; 2],
}

#[repr(C)]
struct UavDesc {
    format: u32,
    view_dimension: u32,
    first_element: u32,
    num_elements: u32,
    flags: u32,
}

#[repr(C)]
struct Mapped {
    data: *mut c_void,
    row_pitch: u32,
    depth_pitch: u32,
}

#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub feature_level: u32,
    pub is_reference_target: bool,
    pub vram_mb: usize,
}

struct ListedAdapter {
    pointer: *mut c_void,
    desc: AdapterDesc,
    name: String,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionSample {
    pub t_pos: [u32; VISION_CLASSES],
    pub t_neg: [u32; VISION_CLASSES],
    pub done_pos: [u32; VISION_CLASSES],
    pub done_neg: [u32; VISION_CLASSES],
    pub dispatch_us: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct IsingSample {
    pub spins: [i32; ISING_NODES],
    pub t_pos: [u32; ISING_NODES],
    pub t_neg: [u32; ISING_NODES],
    pub dispatch_us: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct BitnetSample {
    pub hidden_pos: [i32; 16],
    pub hidden_neg: [i32; 16],
    pub hidden: [i32; 16],
    pub ternary: [i32; 16],
    pub class_pos: [i32; 10],
    pub class_neg: [i32; 10],
    pub scores: [i32; 10],
    pub dispatch_us: f64,
}

fn check(hr: i32, what: &str) {
    if hr < 0 {
        panic!("{what} failed with HRESULT 0x{hr:08X}");
    }
}

unsafe fn vfn<T>(object: *mut c_void, index: usize) -> T {
    let table = *(object as *mut *const usize);
    let pointer = *table.add(index);
    std::mem::transmute_copy(&pointer)
}

unsafe fn release(object: *mut c_void) {
    if object.is_null() {
        return;
    }
    let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = vfn(object, 2);
    release_fn(object);
}

fn utf16z(words: &[u16]) -> String {
    let end = words.iter().position(|unit| *unit == 0).unwrap_or(words.len());
    String::from_utf16_lossy(&words[..end])
}

fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn c_str(text: &str) -> Vec<i8> {
    let mut bytes = Vec::with_capacity(text.len() + 1);
    bytes.extend(text.bytes().map(|byte| byte as i8));
    bytes.push(0);
    bytes
}

fn enumerate_adapters() -> Vec<ListedAdapter> {
    let mut factory = std::ptr::null_mut();
    check(unsafe { CreateDXGIFactory(&IID_IDXGI_FACTORY, &mut factory) }, "CreateDXGIFactory");
    let mut listed = Vec::new();
    for index in 0..8u32 {
        let mut adapter = std::ptr::null_mut();
        let enumerate: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32 =
            unsafe { vfn(factory, 7) };
        let hr = unsafe { enumerate(factory, index, &mut adapter) };
        if hr < 0 {
            break;
        }
        let mut desc = unsafe { std::mem::zeroed::<AdapterDesc>() };
        let get_desc: unsafe extern "system" fn(*mut c_void, *mut AdapterDesc) -> i32 =
            unsafe { vfn(adapter, 8) };
        check(unsafe { get_desc(adapter, &mut desc) }, "IDXGIAdapter::GetDesc");
        let name = utf16z(&desc.description);
        listed.push(ListedAdapter { pointer: adapter, desc, name });
    }
    unsafe { release(factory) };
    listed
}

pub fn available_adapters() -> Vec<AdapterInfo> {
    let mut out = Vec::new();
    let adapters = enumerate_adapters();
    for a in adapters {
        let is_reference = a.desc.vendor_id == 0x8086 && a.desc.device_id == 0x9B41;
        out.push(AdapterInfo {
            name: a.name,
            vendor_id: a.desc.vendor_id,
            device_id: a.desc.device_id,
            feature_level: D3D_FEATURE_LEVEL_11_0,
            is_reference_target: is_reference,
            vram_mb: a.desc.dedicated_video_memory / (1024 * 1024),
        });
        unsafe { release(a.pointer) };
    }
    out
}

fn adapter_rank(desc: &AdapterDesc, name: &str) -> u32 {
    let lower = name.to_ascii_lowercase();
    if lower.contains("basic render") || lower.contains("warp") {
        return 10;
    }
    let mut score = 100u32;
    // Prefer discrete GPUs with dedicated VRAM
    if desc.dedicated_video_memory > 256 * 1024 * 1024 {
        score += 500;
    }
    // High score for Intel UHD reference target if available
    if desc.vendor_id == 0x8086 && desc.device_id == 0x9B41 {
        score += 350;
    } else if desc.vendor_id == 0x8086 {
        score += 200; // Intel iGPU/Arc
    } else if desc.vendor_id == 0x10DE {
        score += 300; // NVIDIA
    } else if desc.vendor_id == 0x1002 {
        score += 300; // AMD
    }
    score
}

fn compile_hlsl_flags(source: &str, name: &str, flags: u32) -> *mut c_void {
    let name_c = c_str(name);
    let entry = c_str("cs_main");
    let target = c_str("cs_5_0");
    let mut code = std::ptr::null_mut();
    let mut errors = std::ptr::null_mut();
    let hr = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name_c.as_ptr(),
            std::ptr::null(),
            std::ptr::null_mut(),
            entry.as_ptr(),
            target.as_ptr(),
            flags,
            0,
            &mut code,
            &mut errors,
        )
    };
    if hr < 0 {
        let message = if errors.is_null() {
            format!("D3DCompile failed with HRESULT 0x{hr:08X}")
        } else {
            let get_pointer: unsafe extern "system" fn(*mut c_void) -> *const u8 = unsafe { vfn(errors, 3) };
            let get_size: unsafe extern "system" fn(*mut c_void) -> usize = unsafe { vfn(errors, 4) };
            let pointer = unsafe { get_pointer(errors) };
            let size = unsafe { get_size(errors) };
            let bytes = unsafe { std::slice::from_raw_parts(pointer, size) };
            let text = String::from_utf8_lossy(bytes).into_owned();
            unsafe { release(errors) };
            text
        };
        panic!("compute shader {name} failed:\n{message}");
    }
    if !errors.is_null() {
        unsafe { release(errors) };
    }
    code
}

fn blob_bytes(blob: *mut c_void) -> Vec<u8> {
    let bytecode: unsafe extern "system" fn(*mut c_void) -> *const u8 = unsafe { vfn(blob, 3) };
    let get_size: unsafe extern "system" fn(*mut c_void) -> usize = unsafe { vfn(blob, 4) };
    let size = unsafe { get_size(blob) };
    let ptr = unsafe { bytecode(blob) };
    unsafe { std::slice::from_raw_parts(ptr, size).to_vec() }
}

fn shader_digest(source: &str, flags: u32) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ u64::from(flags);
    for byte in source.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn shader_cache() -> &'static std::sync::Mutex<Vec<(String, u32, u64, Vec<u8>)>> {
    static CACHE: std::sync::Mutex<Vec<(String, u32, u64, Vec<u8>)>> = std::sync::Mutex::new(Vec::new());
    &CACHE
}

fn cache_lock() -> std::sync::MutexGuard<'static, Vec<(String, u32, u64, Vec<u8>)>> {
    shader_cache().lock().unwrap_or_else(|poison| poison.into_inner())
}

fn cache_get(name: &str, flags: u32, digest: u64) -> Option<Vec<u8>> {
    let cache = cache_lock();
    cache.iter().find(|(cached_name, cached_flags, cached_digest, _)| {
        cached_name == name && *cached_flags == flags && *cached_digest == digest
    }).map(|(_, _, _, bytes)| bytes.clone())
}

fn cache_store(name: &str, flags: u32, digest: u64, bytes: Vec<u8>) {
    let mut cache = cache_lock();
    if let Some(slot) = cache.iter_mut().find(|(cached_name, cached_flags, cached_digest, _)| {
        cached_name == name && *cached_flags == flags && *cached_digest == digest
    }) {
        slot.3 = bytes;
    } else {
        cache.push((name.to_string(), flags, digest, bytes));
    }
}

fn cache_evict(name: &str, flags: u32, digest: u64) {
    let mut cache = cache_lock();
    cache.retain(|(cached_name, cached_flags, cached_digest, _)| {
        !(cached_name == name && *cached_flags == flags && *cached_digest == digest)
    });
}

fn shader_path(name: &str, flags: u32, digest: u64) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("shader-cache");
    path.push(format!("{name}-{flags:x}-{digest:016x}.bin"));
    path
}

/// DXBC container: magic at 0, total size as u32le at offset 24.
fn dxbc_ok(bytes: &[u8]) -> bool {
    if bytes.len() < 32 || bytes.len() > u32::MAX as usize {
        return false;
    }
    if &bytes[0..4] != b"DXBC" {
        return false;
    }
    let declared = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    declared == bytes.len() as u32
}

fn tmp_seq() -> u64 {
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
    TMP_SEQ.fetch_add(1, Ordering::Relaxed)
}

fn write_shader_file(path: &std::path::Path, bytes: &[u8]) {
    let file_name = path.file_name().and_then(|name| name.to_str()).unwrap_or("shader.bin");
    let tmp = path.with_file_name(format!(
        "{file_name}.tmp.{}.{}",
        std::process::id(),
        tmp_seq()
    ));
    let write_tmp = || -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.flush()?;
        Ok(())
    };
    if write_tmp().is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    if std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn compile_shader_bytes(source: &str, name: &str, flags: u32) -> Vec<u8> {
    let blob = compile_hlsl_flags(source, name, flags);
    let bytes = blob_bytes(blob);
    unsafe { release(blob) };
    bytes
}

fn shader_bytes(source: &str, name: &str, flags: u32) -> Vec<u8> {
    let digest = shader_digest(source, flags);
    if let Some(bytes) = cache_get(name, flags, digest) {
        return bytes;
    }
    let path = shader_path(name, flags, digest);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(bytes) = std::fs::read(&path) {
        if dxbc_ok(&bytes) {
            cache_store(name, flags, digest, bytes.clone());
            return bytes;
        }
    }
    let bytes = compile_shader_bytes(source, name, flags);
    write_shader_file(&path, &bytes);
    cache_store(name, flags, digest, bytes.clone());
    bytes
}

fn try_create_cs(device: *mut c_void, bytes: &[u8]) -> Result<*mut c_void, i32> {
    let mut shader = std::ptr::null_mut();
    let create: unsafe extern "system" fn(*mut c_void, *const u8, usize, *mut c_void, *mut *mut c_void) -> i32 =
        unsafe { vfn(device, 18) };
    let hr = unsafe { create(device, bytes.as_ptr(), bytes.len(), std::ptr::null_mut(), &mut shader) };
    if hr < 0 {
        if !shader.is_null() {
            unsafe { release(shader) };
        }
        Err(hr)
    } else {
        Ok(shader)
    }
}

fn create_cs(device: *mut c_void, source: &str, name: &str, flags: u32) -> *mut c_void {
    let digest = shader_digest(source, flags);
    let path = shader_path(name, flags, digest);
    let bytes = shader_bytes(source, name, flags);
    if let Ok(shader) = try_create_cs(device, &bytes) {
        return shader;
    }
    cache_evict(name, flags, digest);
    let _ = std::fs::remove_file(&path);
    let fresh = compile_shader_bytes(source, name, flags);
    write_shader_file(&path, &fresh);
    cache_store(name, flags, digest, fresh.clone());
    match try_create_cs(device, &fresh) {
        Ok(shader) => shader,
        Err(hr) => panic!("CreateComputeShader failed with HRESULT 0x{hr:08X} after recompile of {name}"),
    }
}

fn create_buffer(device: *mut c_void, bytes: u32, usage: u32, bind: u32, cpu: u32, stride: u32) -> *mut c_void {
    let desc = BufferDesc {
        byte_width: bytes,
        usage,
        bind_flags: bind,
        cpu_access: cpu,
        misc_flags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
        stride,
    };
    let mut buffer = std::ptr::null_mut();
    let create: unsafe extern "system" fn(*mut c_void, *const BufferDesc, *const c_void, *mut *mut c_void) -> i32 =
        unsafe { vfn(device, 3) };
    check(unsafe { create(device, &desc, std::ptr::null(), &mut buffer) }, "CreateBuffer");
    buffer
}

fn create_srv(device: *mut c_void, buffer: *mut c_void, elements: u32) -> *mut c_void {
    let desc = SrvDesc {
        format: 0,
        view_dimension: D3D11_SRV_DIMENSION_BUFFER,
        first_element: 0,
        num_elements: elements,
        _union_tail: [0, 0],
    };
    let mut view = std::ptr::null_mut();
    let create: unsafe extern "system" fn(*mut c_void, *mut c_void, *const SrvDesc, *mut *mut c_void) -> i32 =
        unsafe { vfn(device, 7) };
    check(unsafe { create(device, buffer, &desc, &mut view) }, "CreateShaderResourceView");
    view
}

fn create_uav(device: *mut c_void, buffer: *mut c_void, elements: u32) -> *mut c_void {
    let desc = UavDesc {
        format: 0,
        view_dimension: D3D11_UAV_DIMENSION_BUFFER,
        first_element: 0,
        num_elements: elements,
        flags: 0,
    };
    let mut view = std::ptr::null_mut();
    let create: unsafe extern "system" fn(*mut c_void, *mut c_void, *const UavDesc, *mut *mut c_void) -> i32 =
        unsafe { vfn(device, 8) };
    check(unsafe { create(device, buffer, &desc, &mut view) }, "CreateUnorderedAccessView");
    view
}

fn simd_shader() -> &'static str {
    include_str!("shaders/ghost_vault.hlsl")
}

fn vision_shader() -> String {
    format!(
        r#"
StructuredBuffer<float> pixels : register(t0);
StructuredBuffer<float> weights : register(t1);
RWStructuredBuffer<uint4> results : register(u0);
groupshared uint slm_bank[64];
groupshared uint done_pos;
groupshared uint done_neg;
groupshared uint quota[64];
#define MAX_BURST {burst}u
#define PEDESTAL {pedestal}u
uint sensor_race(uint phase_id) {{
    uint prev = 0u;
    if (phase_id == 0u) {{
        InterlockedAdd(slm_bank[0], 1u, prev);
    }} else {{
        InterlockedAdd(slm_bank[16], 1u, prev);
    }}
    return 1u;
}}
[numthreads(65, 1, 1)]
void cs_main(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {{
    uint class_id = gid.x;
    if (tid == 0u) {{
        done_pos = 0u;
        done_neg = 0u;
        slm_bank[0] = 0u;
        slm_bank[16] = 0u;
    }}
    GroupMemoryBarrierWithGroupSync();
    float d = 0.0;
    if (tid < 64u) {{
        float x = pixels[tid];
        float w = weights[class_id * 64u + tid];
        d = clamp(x * w, -1.0, 1.0);
    }}
    uint pos_ticks = 0u;
    uint neg_ticks = 0u;
    if (tid < 64u) {{
        quota[tid] = (d > 0.0) ? (uint)(d * (float)MAX_BURST) : 0u;
    }}
    GroupMemoryBarrierWithGroupSync();
    uint owner;
    uint step;
    if (tid == 64u) {{
        [loop]
        for (step = 0u; step < PEDESTAL; step++) {{
            pos_ticks += sensor_race(0u);
        }}
    }}
    GroupMemoryBarrierWithGroupSync();
    [loop]
    for (owner = 0u; owner < 64u; owner++) {{
        [loop]
        for (step = 0u; step < MAX_BURST; step++) {{
            GroupMemoryBarrierWithGroupSync();
            if (step < quota[owner]) {{
                if (tid == owner) {{
                    uint prev;
                    InterlockedAdd(slm_bank[0], 1, prev);
                }} else if (tid == 64u) {{
                    pos_ticks += sensor_race(0u);
                }}
            }}
        }}
    }}
    if (tid < 64u) {{
        uint prev;
        InterlockedAdd(done_pos, 1, prev);
        quota[tid] = (d < 0.0) ? (uint)((-d) * (float)MAX_BURST) : 0u;
    }}
    GroupMemoryBarrierWithGroupSync();
    if (tid == 64u) {{
        [loop]
        for (step = 0u; step < PEDESTAL; step++) {{
            neg_ticks += sensor_race(1u);
        }}
    }}
    GroupMemoryBarrierWithGroupSync();
    [loop]
    for (owner = 0u; owner < 64u; owner++) {{
        [loop]
        for (step = 0u; step < MAX_BURST; step++) {{
            GroupMemoryBarrierWithGroupSync();
            if (step < quota[owner]) {{
                if (tid == owner) {{
                    uint prev;
                    InterlockedAdd(slm_bank[16], 1, prev);
                }} else if (tid == 64u) {{
                    neg_ticks += sensor_race(1u);
                }}
            }}
        }}
    }}
    if (tid < 64u) {{
        uint prev;
        InterlockedAdd(done_neg, 1, prev);
    }}
    GroupMemoryBarrierWithGroupSync();
    if (tid == 64u) {{
        results[class_id] = uint4(pos_ticks, neg_ticks, done_pos, done_neg);
    }}
}}
"#,
        burst = VISION_MAX_BURST,
        pedestal = VISION_PEDESTAL
    )
}

fn emit_span(kind: &str, lo: usize, hi: usize, indent: usize) -> String {
    let pad = "    ".repeat(indent);
    if hi - lo == 1 {
        let index = lo;
        let body = match kind {
            "add_p" => format!("InterlockedAdd(p{index}, {ISING_PUSH}u, prev);"),
            "add_n" => format!("InterlockedAdd(n{index}, {ISING_PUSH}u, prev);"),
            "load_s" => format!("spin = s{index};"),
            "store_s" => format!("s{index} = spin;"),
            "load_b" => format!("tp = p{index}; tn = n{index};"),
            other => panic!("unknown span {other}"),
        };
        return format!("{pad}{body}\n");
    }
    let mid = lo + (hi - lo) / 2;
    let mut out = String::new();
    out.push_str(&format!("{pad}if (dest < {mid}) {{\n"));
    out.push_str(&emit_span(kind, lo, mid, indent + 1));
    out.push_str(&format!("{pad}}} else {{\n"));
    out.push_str(&emit_span(kind, mid, hi, indent + 1));
    out.push_str(&format!("{pad}}}\n"));
    out
}

fn ising_shader() -> String {
    let mut src = String::new();
    src.push_str("StructuredBuffer<int> nbr : register(t0);\n");
    src.push_str("StructuredBuffer<int> cpl : register(t1);\n");
    src.push_str("StructuredBuffer<int> bias : register(t2);\n");
    src.push_str("RWStructuredBuffer<uint4> outbuf : register(u0);\n");
    for index in 0..ISING_NODES {
        src.push_str(&format!("groupshared uint p{index};\n"));
        src.push_str(&format!("groupshared uint n{index};\n"));
        src.push_str(&format!("groupshared int s{index};\n"));
    }
    src.push_str("void add_p(int dest, inout uint prev) {\n");
    src.push_str(&emit_span("add_p", 0, ISING_NODES, 1));
    src.push_str("}\nvoid add_n(int dest, inout uint prev) {\n");
    src.push_str(&emit_span("add_n", 0, ISING_NODES, 1));
    src.push_str("}\nvoid load_spin(uint id, out int spin) {\n    spin = 0;\n    int dest = (int)id;\n");
    src.push_str(&emit_span("load_s", 0, ISING_NODES, 1));
    src.push_str("}\nvoid store_spin(uint id, int spin) {\n    int dest = (int)id;\n");
    src.push_str(&emit_span("store_s", 0, ISING_NODES, 1));
    src.push_str("}\nvoid load_banks(uint id, out uint tp, out uint tn) {\n    tp = 0u;\n    tn = 0u;\n    int dest = (int)id;\n");
    src.push_str(&emit_span("load_b", 0, ISING_NODES, 1));
    src.push_str("}\n");
    src.push_str(&format!("[numthreads({ISING_NODES}, 1, 1)]\nvoid cs_main(uint tid : SV_GroupThreadID) {{\n"));
    src.push_str("    uint wave = 0u;\n    uint k = 0u;\n    uint prev = 0u;\n    uint tp = 0u;\n    uint tn = 0u;\n");
    src.push_str("    uint do_push = 0u;\n    uint to_pos = 0u;\n    uint code = 0u;\n    int spin = 0;\n    int nb = 0;\n    int c = 0;\n");
    src.push_str("    if (tid == 0u) {\n");
    for index in 0..ISING_NODES {
        src.push_str(&format!("        s{index} = 0;\n"));
    }
    src.push_str("    }\n    GroupMemoryBarrierWithGroupSync();\n    [loop]\n");
    src.push_str(&format!("    for (wave = 0u; wave < {ISING_WAVES}u; wave++) {{\n"));
    src.push_str("        if (tid == 0u) {\n");
    for index in 0..ISING_NODES {
        src.push_str(&format!("            p{index} = 0u;\n            n{index} = 0u;\n"));
    }
    src.push_str("        }\n        GroupMemoryBarrierWithGroupSync();\n");
    src.push_str("        if (wave == 0u) {\n            if (bias[tid] > 0) add_p((int)tid, prev);\n");
    src.push_str("            else if (bias[tid] < 0) add_n((int)tid, prev);\n        }\n");
    src.push_str("        GroupMemoryBarrierWithGroupSync();\n        load_spin(tid, spin);\n        [loop]\n");
    src.push_str(&format!("        for (k = 0u; k < {ISING_DEG}u; k++) {{\n"));
    src.push_str("            do_push = 0u;\n            to_pos = 0u;\n");
    src.push_str(&format!("            nb = nbr[tid * {ISING_DEG}u + k];\n"));
    src.push_str(&format!("            c = cpl[tid * {ISING_DEG}u + k];\n"));
    src.push_str("            if (spin != 0 && nb >= 0 && c != 0) {\n                do_push = 1u;\n");
    src.push_str("                if (c < 0) { if (spin < 0) { to_pos = 1u; } }\n");
    src.push_str("                else { if (spin > 0) { to_pos = 1u; } }\n            }\n");
    src.push_str("            if (do_push == 1u) {\n                if (to_pos == 1u) add_p(nb, prev);\n                else add_n(nb, prev);\n            }\n");
    src.push_str("        }\n        GroupMemoryBarrierWithGroupSync();\n        load_banks(tid, tp, tn);\n");
    src.push_str("        if (tp == 0u && tn == 0u) {\n        } else if (tp > tn) {\n            spin = 1;\n");
    src.push_str("        } else if (tn > tp) {\n            spin = -1;\n        } else {\n            spin = 0;\n        }\n");
    src.push_str("        store_spin(tid, spin);\n        GroupMemoryBarrierWithGroupSync();\n    }\n");
    src.push_str("    code = 0u;\n    if (spin > 0) code = 1u;\n    else if (spin < 0) code = 2u;\n");
    src.push_str("    outbuf[tid] = uint4(code, tp, tn, 0u);\n}\n");
    src
}

fn ising_contract(src: &str) {
    assert!(src.contains(&format!("[numthreads({ISING_NODES}, 1, 1)]")));
    assert!(src.contains(&format!("for (wave = 0u; wave < {ISING_WAVES}u; wave++)")));
    assert!(src.contains(&format!("for (k = 0u; k < {ISING_DEG}u; k++)")));
    assert_eq!(src.matches("groupshared uint p").count(), ISING_NODES);
    assert_eq!(src.matches("groupshared uint n").count(), ISING_NODES);
    assert_eq!(src.matches("groupshared int s").count(), ISING_NODES);
    assert_eq!(src.matches("InterlockedAdd").count(), ISING_NODES * 2);
    assert!(!src.contains("groupshared uint p["));
    assert!(!src.contains("groupshared uint n["));
    assert!(!src.contains("groupshared int s["));
    assert!(!src.contains("InterlockedAdd(p["));
    assert!(!src.contains("InterlockedAdd(n["));
    assert_eq!(src.matches("GroupMemoryBarrierWithGroupSync();").count(), 5);
}

pub struct GpuEngine {
    device: *mut c_void,
    context: *mut c_void,
    simd_shader: *mut c_void,
    vision_shader: *mut c_void,
    ising_shader: *mut c_void,
    sim_a: *mut c_void,
    sim_b: *mut c_void,
    sim_p: *mut c_void,
    sim_out: *mut c_void,
    sim_stage: *mut c_void,
    sim_a_srv: *mut c_void,
    sim_b_srv: *mut c_void,
    sim_p_srv: *mut c_void,
    sim_uav: *mut c_void,
    vis_pix: *mut c_void,
    vis_w: *mut c_void,
    vis_out: *mut c_void,
    vis_stage: *mut c_void,
    vis_pix_srv: *mut c_void,
    vis_w_srv: *mut c_void,
    vis_uav: *mut c_void,
    iso_n: *mut c_void,
    iso_c: *mut c_void,
    iso_b: *mut c_void,
    iso_out: *mut c_void,
    iso_stage: *mut c_void,
    iso_n_srv: *mut c_void,
    iso_c_srv: *mut c_void,
    iso_b_srv: *mut c_void,
    iso_uav: *mut c_void,
    bitnet_shader: *mut c_void,
    bn_pix: *mut c_void,
    bn_w1: *mut c_void,
    bn_w2: *mut c_void,
    bn_out: *mut c_void,
    bn_stage: *mut c_void,
    bn_pix_srv: *mut c_void,
    bn_w1_srv: *mut c_void,
    bn_w2_srv: *mut c_void,
    bn_uav: *mut c_void,
    pub info: AdapterInfo,
    _guard: MutexGuard<'static, ()>,
}

unsafe impl Send for GpuEngine {}

impl GpuEngine {
    pub fn try_new() -> Result<Self, String> {
        let guard = gpu_lock();
        let mut adapters = enumerate_adapters();
        if adapters.is_empty() {
            return Err("DXGI enumerated no adapters on this system".into());
        }
        adapters.sort_by(|a, b| adapter_rank(&b.desc, &b.name).cmp(&adapter_rank(&a.desc, &a.name)));

        let levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
        let mut last_err = String::new();

        for chosen in adapters {
            let mut device = std::ptr::null_mut();
            let mut context = std::ptr::null_mut();
            let mut feature_level = 0u32;
            let mut hr = unsafe {
                D3D11CreateDevice(
                    chosen.pointer,
                    D3D_DRIVER_TYPE_UNKNOWN,
                    std::ptr::null_mut(),
                    0,
                    levels.as_ptr(),
                    levels.len() as u32,
                    D3D11_SDK_VERSION,
                    &mut device,
                    &mut feature_level,
                    &mut context,
                )
            };
            if hr < 0 {
                hr = unsafe {
                    D3D11CreateDevice(
                        chosen.pointer,
                        D3D_DRIVER_TYPE_HARDWARE,
                        std::ptr::null_mut(),
                        0,
                        levels.as_ptr(),
                        levels.len() as u32,
                        D3D11_SDK_VERSION,
                        &mut device,
                        &mut feature_level,
                        &mut context,
                    )
                };
            }
            if hr < 0 {
                last_err = format!("D3D11CreateDevice on {} failed with HRESULT 0x{:08X}", chosen.name, hr);
                unsafe { release(chosen.pointer) };
                continue;
            }
            unsafe { release(chosen.pointer) };

            let is_reference_target = chosen.desc.vendor_id == 0x8086 && chosen.desc.device_id == 0x9B41;
            let info = AdapterInfo {
                name: chosen.name,
                vendor_id: chosen.desc.vendor_id,
                device_id: chosen.desc.device_id,
                feature_level,
                is_reference_target,
                vram_mb: chosen.desc.dedicated_video_memory / (1024 * 1024),
            };
            let sim_a = create_buffer(device, 64, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let sim_b = create_buffer(device, 64, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let sim_p = create_buffer(device, 16, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let sim_out = create_buffer(device, 1024, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, 4);
            let sim_stage = create_buffer(device, 1024, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 4);
            let vis_pix = create_buffer(device, 256, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let vis_w = create_buffer(device, 1024, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let vis_out = create_buffer(device, 64, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, 16);
            let vis_stage = create_buffer(device, 64, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 16);
            let iso_bytes = (ISING_NODES * ISING_DEG * 4) as u32;
            let iso_n = create_buffer(device, iso_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let iso_c = create_buffer(device, iso_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let iso_b = create_buffer(device, (ISING_NODES * 4) as u32, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let iso_out = create_buffer(device, (ISING_NODES * 16) as u32, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, 16);
            let iso_stage = create_buffer(device, (ISING_NODES * 16) as u32, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 16);
            let bn_pix = create_buffer(device, 256, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let bn_w1 = create_buffer(device, 4096, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let bn_w2 = create_buffer(device, 640, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 4);
            let bn_out = create_buffer(device, 416, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, 16);
            let bn_stage = create_buffer(device, 416, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 16);
            return Ok(Self {
                device,
                context,
                simd_shader: std::ptr::null_mut(),
                vision_shader: std::ptr::null_mut(),
                ising_shader: std::ptr::null_mut(),
                sim_a,
                sim_b,
                sim_p,
                sim_out,
                sim_stage,
                sim_a_srv: create_srv(device, sim_a, 16),
                sim_b_srv: create_srv(device, sim_b, 16),
                sim_p_srv: create_srv(device, sim_p, 4),
                sim_uav: create_uav(device, sim_out, 256),
                vis_pix,
                vis_w,
                vis_out,
                vis_stage,
                vis_pix_srv: create_srv(device, vis_pix, VISION_PIXELS as u32),
                vis_w_srv: create_srv(device, vis_w, (VISION_CLASSES * VISION_PIXELS) as u32),
                vis_uav: create_uav(device, vis_out, VISION_CLASSES as u32),
                iso_n,
                iso_c,
                iso_b,
                iso_out,
                iso_stage,
                iso_n_srv: create_srv(device, iso_n, (ISING_NODES * ISING_DEG) as u32),
                iso_c_srv: create_srv(device, iso_c, (ISING_NODES * ISING_DEG) as u32),
                iso_b_srv: create_srv(device, iso_b, ISING_NODES as u32),
                iso_uav: create_uav(device, iso_out, ISING_NODES as u32),
                bitnet_shader: std::ptr::null_mut(),
                bn_pix,
                bn_w1,
                bn_w2,
                bn_out,
                bn_stage,
                bn_pix_srv: create_srv(device, bn_pix, 64),
                bn_w1_srv: create_srv(device, bn_w1, 1024),
                bn_w2_srv: create_srv(device, bn_w2, 160),
                bn_uav: create_uav(device, bn_out, 26),
                info,
                _guard: guard,
            });
        }
        Err(if last_err.is_empty() { "No suitable D3D11 compute adapter found".into() } else { last_err })
    }

    pub fn new() -> Self {
        Self::try_new().expect("D3D11 compute engine initialization")
    }

    pub fn feature_name(&self) -> &'static str {
        if self.info.feature_level >= D3D_FEATURE_LEVEL_11_1 { "11.1" } else { "11.0" }
    }

    fn ensure_simd(&mut self) {
        if !self.simd_shader.is_null() {
            return;
        }
        self.simd_shader = create_cs(self.device, &simd_shader(), "dcc_stress_simd.hlsl", D3DCOMPILE_SKIP_OPTIMIZATION);
    }

    fn ensure_vision(&mut self) {
        if !self.vision_shader.is_null() {
            return;
        }
        self.vision_shader = create_cs(self.device, &vision_shader(), "dcc_stress_vision.hlsl", D3DCOMPILE_SKIP_OPTIMIZATION);
    }

    fn ensure_ising(&mut self) {
        if !self.ising_shader.is_null() {
            return;
        }
        println!("  compiling Ising cs_5_0 shader");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let source = ising_shader();
        ising_contract(&source);
        self.ising_shader = create_cs(self.device, &source, "dcc_stress_ising.hlsl", D3DCOMPILE_SKIP_OPTIMIZATION);
    }

    unsafe fn upload(&self, buffer: *mut c_void, bytes: *const u8, len: usize) {
        let update: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void, *const u8, u32, u32) =
            vfn(self.context, 48);
        update(self.context, buffer, 0, std::ptr::null(), bytes, len as u32, 0);
    }

    unsafe fn dispatch_copy(
        &self,
        shader: *mut c_void,
        srvs: &[*mut c_void],
        uav: *mut c_void,
        groups: u32,
        staging: *mut c_void,
        source: *mut c_void,
        words: usize,
    ) -> (Vec<u32>, f64) {
        let set_srv: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) = vfn(self.context, 67);
        let set_uav: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void, *const u32) = vfn(self.context, 68);
        let set_shader: unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32) = vfn(self.context, 69);
        let dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32) = vfn(self.context, 41);
        let copy: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) = vfn(self.context, 47);
        let started = Instant::now();
        set_shader(self.context, shader, std::ptr::null(), 0);
        set_srv(self.context, 0, srvs.len() as u32, srvs.as_ptr());
        set_uav(self.context, 0, 1, &uav, std::ptr::null());
        dispatch(self.context, groups, 1, 1);
        let null_view = std::ptr::null_mut();
        set_uav(self.context, 0, 1, &null_view, std::ptr::null());
        let nulls = [null_view, null_view, null_view];
        set_srv(self.context, 0, 3, nulls.as_ptr());
        set_shader(self.context, std::ptr::null_mut(), std::ptr::null(), 0);
        copy(self.context, staging, source);
        let map: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32 = vfn(self.context, 14);
        let unmap: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = vfn(self.context, 15);
        let mut mapped = Mapped { data: std::ptr::null_mut(), row_pitch: 0, depth_pitch: 0 };
        check(map(self.context, staging, 0, D3D11_MAP_READ, 0, &mut mapped), "Map staging");
        let slice = std::slice::from_raw_parts(mapped.data.cast::<u32>(), words);
        let out = slice.to_vec();
        unmap(self.context, staging, 0);
        let us = started.elapsed().as_secs_f64() * 1.0e6;
        crate::ThermalGovernor::after_dispatch();
        (out, us)
    }

    pub fn simd_bytes(&mut self, op: u32, a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
        self.ensure_simd();
        let a_words: [u32; 16] = std::array::from_fn(|index| u32::from(a[index]));
        let b_words: [u32; 16] = std::array::from_fn(|index| u32::from(b[index]));
        let params = [op, 0, 0, 0];
        let (words, _us) = unsafe {
            self.upload(self.sim_a, a_words.as_ptr().cast(), 64);
            self.upload(self.sim_b, b_words.as_ptr().cast(), 64);
            self.upload(self.sim_p, params.as_ptr().cast(), 16);
            let srvs = [self.sim_a_srv, self.sim_b_srv, self.sim_p_srv];
            self.dispatch_copy(self.simd_shader, &srvs, self.sim_uav, 16, self.sim_stage, self.sim_out, 256)
        };
        decode_simd(op, &words)
    }

    pub fn simd_timed(&mut self, op: u32, a: &[u8; 16], b: &[u8; 16]) -> ([u8; 16], f64) {
        self.ensure_simd();
        let a_words: [u32; 16] = std::array::from_fn(|index| u32::from(a[index]));
        let b_words: [u32; 16] = std::array::from_fn(|index| u32::from(b[index]));
        let params = [op, 0, 0, 0];
        let (words, us) = unsafe {
            self.upload(self.sim_a, a_words.as_ptr().cast(), 64);
            self.upload(self.sim_b, b_words.as_ptr().cast(), 64);
            self.upload(self.sim_p, params.as_ptr().cast(), 16);
            let srvs = [self.sim_a_srv, self.sim_b_srv, self.sim_p_srv];
            self.dispatch_copy(self.simd_shader, &srvs, self.sim_uav, 16, self.sim_stage, self.sim_out, 256)
        };
        (decode_simd(op, &words), us)
    }

    pub fn vision(&mut self, pixels: &[f32; 64], weights: &[[f32; 64]; 4]) -> VisionSample {
        self.ensure_vision();
        let mut packed = [0.0f32; VISION_CLASSES * VISION_PIXELS];
        for class in 0..VISION_CLASSES {
            packed[class * VISION_PIXELS..(class + 1) * VISION_PIXELS].copy_from_slice(&weights[class]);
        }
        let (words, us) = unsafe {
            self.upload(self.vis_pix, pixels.as_ptr().cast(), 256);
            self.upload(self.vis_w, packed.as_ptr().cast(), packed.len() * 4);
            let srvs = [self.vis_pix_srv, self.vis_w_srv];
            self.dispatch_copy(self.vision_shader, &srvs, self.vis_uav, VISION_CLASSES as u32, self.vis_stage, self.vis_out, 16)
        };
        let mut sample = VisionSample {
            t_pos: [0; 4],
            t_neg: [0; 4],
            done_pos: [0; 4],
            done_neg: [0; 4],
            dispatch_us: us,
        };
        for class in 0..4 {
            sample.t_pos[class] = words[class * 4];
            sample.t_neg[class] = words[class * 4 + 1];
            sample.done_pos[class] = words[class * 4 + 2];
            sample.done_neg[class] = words[class * 4 + 3];
        }
        sample
    }

    pub fn ising(&mut self, graph: &IsingGraph) -> IsingSample {
        self.ensure_ising();
        let mut neighbors = [0i32; ISING_NODES * ISING_DEG];
        let mut coupling = [0i32; ISING_NODES * ISING_DEG];
        for node in 0..ISING_NODES {
            for slot in 0..ISING_DEG {
                neighbors[node * ISING_DEG + slot] = graph.neighbors[node][slot];
                coupling[node * ISING_DEG + slot] = graph.coupling[node][slot];
            }
        }
        let (words, us) = unsafe {
            self.upload(self.iso_n, neighbors.as_ptr().cast(), neighbors.len() * 4);
            self.upload(self.iso_c, coupling.as_ptr().cast(), coupling.len() * 4);
            self.upload(self.iso_b, graph.bias.as_ptr().cast(), ISING_NODES * 4);
            let srvs = [self.iso_n_srv, self.iso_c_srv, self.iso_b_srv];
            self.dispatch_copy(self.ising_shader, &srvs, self.iso_uav, 1, self.iso_stage, self.iso_out, ISING_NODES * 4)
        };
        let mut sample = IsingSample {
            spins: [0; ISING_NODES],
            t_pos: [0; ISING_NODES],
            t_neg: [0; ISING_NODES],
            dispatch_us: us,
        };
        for node in 0..ISING_NODES {
            let code = words[node * 4];
            sample.spins[node] = match code {
                1 => 1,
                2 => -1,
                _ => 0,
            };
            sample.t_pos[node] = words[node * 4 + 1];
            sample.t_neg[node] = words[node * 4 + 2];
        }
        sample
    }

    fn ensure_bitnet(&mut self) {
        if !self.bitnet_shader.is_null() {
            return;
        }
        let source = include_str!("shaders/bitnet_digit.hlsl");
        self.bitnet_shader = create_cs(self.device, source, "bitnet_digit.hlsl", 0);
    }

    pub fn bitnet(&mut self, pixels: &[f32; 64]) -> BitnetSample {
        self.ensure_bitnet();
        let mut w1 = [0i32; 1024];
        for neuron in 0..16 {
            for pixel in 0..64 {
                w1[neuron * 64 + pixel] = i32::from(crate::bitnet::W1[neuron][pixel]);
            }
        }
        let mut w2 = [0i32; 160];
        for class in 0..10 {
            for neuron in 0..16 {
                w2[class * 16 + neuron] = i32::from(crate::bitnet::W2[class][neuron]);
            }
        }
        let (words, us) = unsafe {
            self.upload(self.bn_pix, pixels.as_ptr().cast(), 256);
            self.upload(self.bn_w1, w1.as_ptr().cast(), 4096);
            self.upload(self.bn_w2, w2.as_ptr().cast(), 640);
            let srvs = [self.bn_pix_srv, self.bn_w1_srv, self.bn_w2_srv];
            self.dispatch_copy(self.bitnet_shader, &srvs, self.bn_uav, 1, self.bn_stage, self.bn_out, 104)
        };
        let signed = |word: u32| word as i32;
        let mut sample = BitnetSample {
            hidden_pos: [0; 16],
            hidden_neg: [0; 16],
            hidden: [0; 16],
            ternary: [0; 16],
            class_pos: [0; 10],
            class_neg: [0; 10],
            scores: [0; 10],
            dispatch_us: us,
        };
        for neuron in 0..16 {
            let base = neuron * 4;
            sample.hidden_pos[neuron] = signed(words[base]);
            sample.hidden_neg[neuron] = signed(words[base + 1]);
            sample.hidden[neuron] = signed(words[base + 2]);
            sample.ternary[neuron] = signed(words[base + 3]);
        }
        for class in 0..10 {
            let base = (16 + class) * 4;
            sample.class_pos[class] = signed(words[base]);
            sample.class_neg[class] = signed(words[base + 1]);
            sample.scores[class] = signed(words[base + 2]);
        }
        sample
    }
}

impl Drop for GpuEngine {
    fn drop(&mut self) {
        unsafe {
            for object in [
                self.bn_uav, self.bn_w2_srv, self.bn_w1_srv, self.bn_pix_srv, self.bn_stage, self.bn_out,
                self.bn_w2, self.bn_w1, self.bn_pix, self.bitnet_shader,
                self.iso_uav, self.iso_b_srv, self.iso_c_srv, self.iso_n_srv, self.iso_stage, self.iso_out,
                self.iso_b, self.iso_c, self.iso_n, self.vis_uav, self.vis_w_srv, self.vis_pix_srv,
                self.vis_stage, self.vis_out, self.vis_w, self.vis_pix, self.sim_uav, self.sim_p_srv,
                self.sim_b_srv, self.sim_a_srv, self.sim_stage, self.sim_out, self.sim_p, self.sim_b,
                self.sim_a, self.ising_shader, self.vision_shader, self.simd_shader, self.context, self.device,
            ] {
                release(object);
            }
        }
    }
}

fn decode_simd(op: u32, words: &[u32]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for lane in 0..16 {
        let mut byte = 0u8;
        for bit in 0..8 {
            let base = lane * 16 + bit * 2;
            let wave = decode_counts(words[base], words[base + 1]);
            let on = match op {
                GPU_ADD | GPU_SUB => wave.sum_bit,
                GPU_XOR => wave.xor_bit,
                GPU_AND | GPU_OR => wave.sign_bit,
                _ => 0,
            };
            if on == 1 {
                byte |= 1 << bit;
            }
        }
        out[lane] = byte;
    }
    out
}

#[cfg(test)]
mod shader_cache_tests {
    use super::{cache_evict, create_cs, dxbc_ok, shader_bytes, shader_digest, shader_path, GpuEngine};

    const PROBE: &str = "[numthreads(1, 1, 1)] void cs_main(uint3 id : SV_DispatchThreadID) {}\n";
    const PROBE_NAME: &str = "dcc_cache_probe.hlsl";

    #[test]
    fn truncated_shader_cache_blob_is_recompiled() {
        crate::ThermalGovernor::reset();
        let flags = 0u32;
        let digest = shader_digest(PROBE, flags);
        let path = shader_path(PROBE_NAME, flags, digest);
        cache_evict(PROBE_NAME, flags, digest);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&path, b"DXBC not a shader").expect("plant truncated blob");
        let compiled = shader_bytes(PROBE, PROBE_NAME, flags);
        assert!(dxbc_ok(&compiled), "truncated blob was accepted");
        let on_disk = std::fs::read(&path).expect("read recompiled blob");
        assert!(dxbc_ok(&on_disk), "disk blob was not replaced");
        assert_ne!(on_disk, b"DXBC not a shader");

        let mut junk = vec![0u8; 64];
        junk[0..4].copy_from_slice(b"DXBC");
        junk[24..28].copy_from_slice(&64u32.to_le_bytes());
        assert!(dxbc_ok(&junk), "planted corrupt header should pass the size check");
        cache_evict(PROBE_NAME, flags, digest);
        std::fs::write(&path, &junk).expect("plant corrupt dxbc");
        let engine = GpuEngine::new();
        let shader = create_cs(engine.device, PROBE, PROBE_NAME, flags);
        assert!(!shader.is_null());
        unsafe { super::release(shader) };
        let repaired = std::fs::read(&path).expect("read repaired blob");
        assert!(dxbc_ok(&repaired));
        assert_ne!(repaired, junk);
        cache_evict(PROBE_NAME, flags, digest);
        let _ = std::fs::remove_file(&path);
    }
}

pub fn native_lane(op: u32, a: u8, b: u8) -> u8 {
    match op {
        GPU_ADD => a.wrapping_add(b),
        GPU_SUB => a.wrapping_sub(b),
        GPU_XOR => a ^ b,
        GPU_AND => a & b,
        GPU_OR => a | b,
        _ => 0,
    }
}
