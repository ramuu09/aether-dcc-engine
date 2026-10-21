//! Continuous-timing telemetry classifier.
//!
//! Workers are unpinned and below normal priority. Between sweeps they park.
//! Each channel takes three RDTSC differential pairs (`SAMPLES` is 3) and
//! keeps the median.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::MutexGuard;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::cpu_alu::SYNAPSE_CPUS;
use crate::machine::{cpu_lock, elevate};

const PHASE_IDLE: u32 = 0;
const PHASE_POS: u32 = 1;
const PHASE_NEG: u32 = 2;
const PHASE_STOP: u32 = 3;

const N_CHANNELS: usize = 4;
const N_SYNAPSES: usize = 3;
const RACE_OPS: u32 = 256;
const BURST_MAX: u32 = 32;
const SAMPLES: usize = 3;
const SETTLE: Duration = Duration::from_micros(250);

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 0x04;

pub const CLASS_NAMES: [&str; 4] = [
    "NORMAL_HEALTHY",
    "CPU_THERMAL_SURGE",
    "MEMORY_IO_PRESSURE",
    "SPOOFED_TELEMETRY_ANOMALY",
];

pub const DIAGNOSTIC_MATRIX: [[f64; 3]; 4] = [
    [-0.25, -0.25, 1.00],
    [1.00, -0.80, -0.85],
    [-0.45, 1.00, -0.85],
    [0.70, 0.70, 0.30],
];

pub struct Scenario {
    pub name: &'static str,
    pub inputs: [f64; 3],
    pub expected: usize,
}

pub const SCENARIOS: [Scenario; 8] = [
    Scenario { name: "Authenticated idle", inputs: [0.06, 0.05, 0.92], expected: 0 },
    Scenario { name: "Nominal quiet, auth on", inputs: [0.12, 0.10, 0.60], expected: 0 },
    Scenario { name: "CPU thermal surge", inputs: [0.98, 0.04, 0.02], expected: 1 },
    Scenario { name: "CPU surge, light noise", inputs: [0.88, 0.10, 0.05], expected: 1 },
    Scenario { name: "Memory IO pressure", inputs: [0.04, 0.98, 0.02], expected: 2 },
    Scenario { name: "Memory pressure, light noise", inputs: [0.10, 0.88, 0.04], expected: 2 },
    Scenario { name: "Spoofed dual spike", inputs: [0.96, 0.96, 0.90], expected: 3 },
    Scenario { name: "Spoofed, partial auth", inputs: [1.00, 0.92, 0.75], expected: 3 },
];

#[link(name = "kernel32", kind = "raw-dylib")]
extern "system" {
    fn GetCurrentThread() -> *mut core::ffi::c_void;
    fn SetThreadPriority(thread: *mut core::ffi::c_void, priority: i32) -> i32;
    fn VirtualAlloc(addr: *mut core::ffi::c_void, size: usize, kind: u32, protect: u32) -> *mut core::ffi::c_void;
    fn VirtualFree(addr: *mut core::ffi::c_void, size: usize, kind: u32) -> i32;
    fn GetSystemTimes(idle: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
}

#[allow(dead_code)]
const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;

#[repr(C, align(256))]
struct ChannelLine {
    arena: AtomicU64,
    _pad: [u8; 248],
}

#[repr(C)]
struct ControlHead {
    active_channel: AtomicU32,
    phase: AtomicU32,
    sync_epoch: AtomicU64,
    ready: AtomicU32,
    park_flag: AtomicU32,
    asleep: AtomicU32,
    sweep_req: AtomicU32,
    sweep_done: AtomicU32,
    sensor_ready: AtomicU32,
    stop: AtomicBool,
}

#[repr(C, align(256))]
struct SynapseSlot {
    x: AtomicU64,
    w: [AtomicU64; 4],
    _pad: [u8; 216],
}

#[repr(C, align(4096))]
struct CacheBank {
    lines: [ChannelLine; N_CHANNELS],
    _gap: [u8; 1024],
    control: ControlHead,
    raw_bits: [AtomicU64; N_CHANNELS],
    race_bits: AtomicU64,
    _after: [u8; 3072 - 2048 - std::mem::size_of::<ControlHead>() - 32 - 8],
    synapses: [SynapseSlot; N_SYNAPSES],
    _tail: [u8; 256],
}

const _: () = assert!(std::mem::size_of::<ChannelLine>() == 256);
const _: () = assert!(std::mem::size_of::<SynapseSlot>() == 256);
const _: () = assert!(std::mem::size_of::<ControlHead>() <= 64);
const _: () = assert!(std::mem::offset_of!(CacheBank, control) == 2048);
const _: () = assert!(std::mem::offset_of!(CacheBank, synapses) == 3072);
const _: () = assert!(std::mem::size_of::<CacheBank>() == 4096);
const _: () = assert!(std::mem::align_of::<CacheBank>() == 4096);

#[derive(Clone, Copy, Debug)]
struct ChannelScale {
    zero_bias: f64,
    y_pos: f64,
    y_neg: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassReading {
    pub outputs: [f64; 4],
    pub predicted: usize,
    pub latency_us: f64,
}

fn demote_worker() {
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), 0); // Normal priority prevents scheduler starvation
    }
}

fn pause() {
    core::arch::x86_64::_mm_pause();
}

fn read_tsc() -> u64 {
    unsafe {
        core::arch::x86_64::_mm_lfence();
        let ticks = core::arch::x86_64::_rdtsc();
        core::arch::x86_64::_mm_lfence();
        ticks
    }
}

fn hammer(arena: &AtomicU64, ops: u32) {
    let mut left = ops;
    while left != 0 {
        let observed = arena.fetch_add(1, Ordering::Relaxed);
        std::hint::black_box(observed);
        left -= 1;
    }
}

fn differential(pos: u64, neg: u64) -> f64 {
    let sum = pos as f64 + neg as f64;
    if sum == 0.0 { 0.0 } else { (pos as f64 - neg as f64) / sum }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

fn argmax4(values: [f64; 4]) -> usize {
    let mut best = 0usize;
    for index in 1..4 {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn clip_unit(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

fn normalize(raw: f64, scale: &ChannelScale) -> f64 {
    let y = if raw >= scale.zero_bias {
        (raw - scale.zero_bias) / (scale.y_pos - scale.zero_bias)
    } else {
        (raw - scale.zero_bias) / (scale.zero_bias - scale.y_neg)
    };
    y.clamp(-1.0, 1.0)
}

fn settle() {
    let start = Instant::now();
    while start.elapsed() < SETTLE {
        pause();
    }
}

fn arm(control: &ControlHead, channel: usize, phase: u32) {
    control.active_channel.store(channel as u32, Ordering::Relaxed);
    control.phase.store(phase, Ordering::Release);
    settle();
}

fn idle_phase(control: &ControlHead) {
    control.phase.store(PHASE_IDLE, Ordering::Release);
    settle();
}

fn sweep_bank(bank: &CacheBank) -> ([f64; 4], f64) {
    let control = &bank.control;
    let mut raw = [0.0; N_CHANNELS];
    let mut race_acc = 0.0f64;
    let mut race_n = 0.0f64;
    for channel in 0..N_CHANNELS {
        let mut ratios = [0.0; SAMPLES];
        for sample in 0..SAMPLES {
            arm(control, channel, PHASE_POS);
            let pos_mark = Instant::now();
            let pos_ticks = {
                let start = read_tsc();
                hammer(&bank.lines[channel].arena, RACE_OPS);
                read_tsc().wrapping_sub(start)
            };
            let pos_us = pos_mark.elapsed().as_secs_f64() * 1.0e6;
            arm(control, channel, PHASE_NEG);
            let neg_mark = Instant::now();
            let neg_ticks = {
                let start = read_tsc();
                hammer(&bank.lines[channel].arena, RACE_OPS);
                read_tsc().wrapping_sub(start)
            };
            let neg_us = neg_mark.elapsed().as_secs_f64() * 1.0e6;
            idle_phase(control);
            ratios[sample] = differential(pos_ticks, neg_ticks);
            race_acc += pos_us + neg_us;
            race_n += 1.0;
        }
        raw[channel] = median(&mut ratios);
    }
    idle_phase(control);
    (raw, race_acc / race_n.max(1.0))
}

fn synapse_worker(addr: usize, index: usize, _cpu: usize, main_thread: thread::Thread) {
    demote_worker();
    let bank = unsafe { &*(addr as *const CacheBank) };
    let control = &bank.control;
    let slot = &bank.synapses[index];
    let mut sample = 0.0f64;
    let mut weights = [0.0f64; N_CHANNELS];
    let mut seen_epoch = 0u64;
    loop {
        let phase = control.phase.load(Ordering::Acquire);
        if phase == PHASE_STOP || control.stop.load(Ordering::Acquire) {
            break;
        }
        if phase == PHASE_IDLE {
            let epoch = control.sync_epoch.load(Ordering::Acquire);
            if epoch != seen_epoch {
                sample = f64::from_bits(slot.x.load(Ordering::Acquire));
                for channel in 0..N_CHANNELS {
                    weights[channel] = f64::from_bits(slot.w[channel].load(Ordering::Acquire));
                }
                seen_epoch = epoch;
                control.ready.fetch_add(1, Ordering::Release);
                main_thread.unpark();
            }
            if control.park_flag.load(Ordering::Acquire) != 0 {
                control.asleep.fetch_add(1, Ordering::Release);
                main_thread.unpark();
                while !control.stop.load(Ordering::Acquire)
                    && control.phase.load(Ordering::Acquire) != PHASE_STOP
                    && control.park_flag.load(Ordering::Acquire) != 0
                {
                    thread::park_timeout(Duration::from_millis(2));
                }
                control.asleep.fetch_sub(1, Ordering::Release);
            } else {
                pause();
            }
            continue;
        }
        let channel = control.active_channel.load(Ordering::Relaxed) as usize;
        if channel >= N_CHANNELS {
            settle();
            continue;
        }
        let drive = clip_unit(sample * weights[channel]);
        let hot = (phase == PHASE_POS && drive > 0.0) || (phase == PHASE_NEG && drive < 0.0);
        let active = (drive.abs() * BURST_MAX as f64) as u32;
        if hot && active > 0 {
            hammer(&bank.lines[channel].arena, active);
            let coast = BURST_MAX - active;
            let mut left = coast;
            while left != 0 {
                pause();
                left -= 1;
            }
        } else {
            let mut left = BURST_MAX;
            while left != 0 {
                pause();
                left -= 1;
            }
        }
    }
}

fn sensor_worker(addr: usize, main_thread: thread::Thread) {
    demote_worker();
    let bank = unsafe { &*(addr as *const CacheBank) };
    let control = &bank.control;
    control.sensor_ready.store(1, Ordering::Release);
    main_thread.unpark();
    let mut seen = 0u32;
    loop {
        if control.stop.load(Ordering::Acquire) {
            break;
        }
        let job = control.sweep_req.load(Ordering::Acquire);
        if job == seen {
            thread::park();
            continue;
        }
        let (raw, race_us) = sweep_bank(bank);
        for channel in 0..N_CHANNELS {
            bank.raw_bits[channel].store(raw[channel].to_bits(), Ordering::Release);
        }
        bank.race_bits.store(race_us.to_bits(), Ordering::Release);
        control.sweep_done.store(job, Ordering::Release);
        main_thread.unpark();
        seen = job;
    }
}

/// If `Sentinel::new` panics while workers exist, join them before the page is freed.
struct SpawnGuard {
    page: *mut CacheBank,
    ctrl: *const ControlHead,
    threads: Vec<JoinHandle<()>>,
}

impl SpawnGuard {
    fn disarm(&mut self) -> Vec<JoinHandle<()>> {
        self.page = std::ptr::null_mut();
        self.ctrl = std::ptr::null();
        std::mem::take(&mut self.threads)
    }
}

impl Drop for SpawnGuard {
    fn drop(&mut self) {
        if self.page.is_null() {
            return;
        }
        unsafe {
            let control = &*self.ctrl;
            control.stop.store(true, Ordering::Release);
            control.phase.store(PHASE_STOP, Ordering::Release);
            control.park_flag.store(0, Ordering::Release);
        }
        for worker in &self.threads {
            worker.thread().unpark();
        }
        for worker in self.threads.drain(..) {
            let _ = worker.join();
        }
        unsafe {
            VirtualFree(self.page.cast(), 0, MEM_RELEASE);
            self.page = std::ptr::null_mut();
            self.ctrl = std::ptr::null();
        }
    }
}

pub struct Sentinel {
    page: *mut CacheBank,
    threads: Vec<JoinHandle<()>>,
    columns: [[f64; 4]; N_SYNAPSES],
    scales: [ChannelScale; N_CHANNELS],
    next_sweep: u32,
    _guard: MutexGuard<'static, ()>,
}

impl Sentinel {
    pub fn new() -> Result<Self, String> {
        elevate();
        let guard = cpu_lock();
        let bytes = std::mem::size_of::<CacheBank>();
        if bytes > 4096 {
            return Err(format!("sentinel bank is {bytes} bytes, expected a single page"));
        }
        let page = unsafe {
            VirtualAlloc(std::ptr::null_mut(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        };
        if page.is_null() {
            return Err("VirtualAlloc failed".into());
        }
        let page = page.cast::<CacheBank>();
        unsafe {
            let bank = &*page;
            std::ptr::write(std::ptr::addr_of!(bank.control).cast_mut(), ControlHead {
                active_channel: AtomicU32::new(0),
                phase: AtomicU32::new(PHASE_IDLE),
                sync_epoch: AtomicU64::new(0),
                ready: AtomicU32::new(0),
                park_flag: AtomicU32::new(1),
                asleep: AtomicU32::new(0),
                sweep_req: AtomicU32::new(0),
                sweep_done: AtomicU32::new(0),
                sensor_ready: AtomicU32::new(0),
                stop: AtomicBool::new(false),
            });
        }
        let mut spawning = SpawnGuard {
            page,
            ctrl: unsafe { std::ptr::addr_of!((*page).control) },
            threads: Vec::with_capacity(4),
        };
        unsafe {
            let bank = &*spawning.page;
            for line in &bank.lines {
                std::ptr::write(std::ptr::addr_of!(line.arena).cast_mut(), AtomicU64::new(0));
            }
            for slot in &bank.synapses {
                std::ptr::write(std::ptr::addr_of!(slot.x).cast_mut(), AtomicU64::new(0));
                for weight in &slot.w {
                    std::ptr::write(std::ptr::addr_of!(*weight).cast_mut(), AtomicU64::new(0));
                }
            }
            for word in &bank.raw_bits {
                std::ptr::write(std::ptr::addr_of!(*word).cast_mut(), AtomicU64::new(0));
            }
            std::ptr::write(std::ptr::addr_of!(bank.race_bits).cast_mut(), AtomicU64::new(0));
        }
        let main_thread = thread::current();
        let addr = spawning.page as usize;
        let sensor_wake = main_thread.clone();
        spawning.threads.push(
            thread::Builder::new()
                .name("dcc-stress-sentinel-sensor".into())
                .spawn(move || sensor_worker(addr, sensor_wake))
                .expect("spawn sentinel sensor"),
        );
        for (index, cpu) in SYNAPSE_CPUS.into_iter().enumerate() {
            let addr = spawning.page as usize;
            let wake = main_thread.clone();
            spawning.threads.push(
                thread::Builder::new()
                    .name(format!("dcc-stress-sentinel-synapse-{index}"))
                    .spawn(move || synapse_worker(addr, index, cpu, wake))
                    .expect("spawn sentinel synapse"),
            );
        }
        let threads = spawning.disarm();
        let mut layer = Self {
            page,
            threads,
            columns: diagnostic_columns(),
            scales: [ChannelScale { zero_bias: 0.0, y_pos: 1.0, y_neg: -1.0 }; N_CHANNELS],
            next_sweep: 1,
            _guard: guard,
        };
        layer.wait_asleep()?;
        layer.wait_sensor()?;
        layer.calibrate()?;
        Ok(layer)
    }

    pub fn classify(&mut self, inputs: [f64; 3]) -> Result<ClassReading, String> {
        let started = Instant::now();
        let raw = self.sweep(inputs)?;
        let mut outputs = [0.0; N_CHANNELS];
        for channel in 0..N_CHANNELS {
            outputs[channel] = normalize(raw[channel], &self.scales[channel]);
        }
        let latency_us = started.elapsed().as_secs_f64() * 1.0e6;
        crate::ThermalGovernor::after_contention(1);
        Ok(ClassReading {
            outputs,
            predicted: argmax4(outputs),
            latency_us,
        })
    }

    fn calibrate(&mut self) -> Result<(), String> {
        self.columns = [[1.0; N_CHANNELS]; N_SYNAPSES];
        let _warm = self.sweep([0.0, 0.0, 0.0])?;
        let mut last_error = String::new();
        for _try in 0..6 {
            let zero = self.sweep([0.0, 0.0, 0.0])?;
            let pos = self.sweep([1.0, 1.0, 1.0])?;
            let neg = self.sweep([-1.0, -1.0, -1.0])?;
            let mut ok = true;
            let mut scales = self.scales;
            for channel in 0..N_CHANNELS {
                let up = pos[channel] - zero[channel];
                let down = zero[channel] - neg[channel];
                if up < 0.03 || down < 0.03 {
                    ok = false;
                    last_error = format!(
                        "channel {channel} calibration span too small zero={:.4} pos={:.4} neg={:.4}",
                        zero[channel], pos[channel], neg[channel]
                    );
                    break;
                }
                scales[channel] = ChannelScale {
                    zero_bias: zero[channel],
                    y_pos: pos[channel],
                    y_neg: neg[channel],
                };
            }
            if ok {
                self.scales = scales;
                self.columns = diagnostic_columns();
                return Ok(());
            }
            thread::sleep(Duration::from_millis(10));
        }
        Err(last_error)
    }

    fn sweep(&mut self, inputs: [f64; 3]) -> Result<[f64; 4], String> {
        self.publish(inputs)?;
        let job = self.next_sweep;
        self.next_sweep = self.next_sweep.wrapping_add(1);
        if self.next_sweep == 0 {
            self.next_sweep = 1;
        }
        let control = unsafe { &(*self.page).control };
        control.sweep_req.store(job, Ordering::Release);
        self.threads[0].thread().unpark();
        let start = Instant::now();
        while control.sweep_done.load(Ordering::Acquire) != job {
            thread::park_timeout(Duration::from_millis(1));
            if start.elapsed().as_millis() > 8_000 {
                return Err("sentinel sweep timed out".into());
            }
        }
        let bank = unsafe { &*self.page };
        let mut raw = [0.0; N_CHANNELS];
        for channel in 0..N_CHANNELS {
            raw[channel] = f64::from_bits(bank.raw_bits[channel].load(Ordering::Acquire));
        }
        self.disengage()?;
        Ok(raw)
    }

    fn publish(&self, inputs: [f64; 3]) -> Result<(), String> {
        unsafe {
            let bank = &*self.page;
            for synapse in 0..N_SYNAPSES {
                let slot = &bank.synapses[synapse];
                slot.x.store(clip_unit(inputs[synapse]).to_bits(), Ordering::Relaxed);
                for channel in 0..N_CHANNELS {
                    slot.w[channel].store(self.columns[synapse][channel].to_bits(), Ordering::Relaxed);
                }
            }
            bank.control.ready.store(0, Ordering::Relaxed);
            bank.control.park_flag.store(0, Ordering::Release);
            bank.control.sync_epoch.fetch_add(1, Ordering::Release);
        }
        self.wake_all();
        self.wait_ready()
    }

    fn disengage(&self) -> Result<(), String> {
        unsafe {
            let control = &(*self.page).control;
            control.phase.store(PHASE_IDLE, Ordering::Release);
            control.park_flag.store(1, Ordering::Release);
        }
        self.wait_asleep()
    }

    fn wake_all(&self) {
        for worker in &self.threads {
            worker.thread().unpark();
        }
    }

    fn wait_asleep(&self) -> Result<(), String> {
        let control = unsafe { &(*self.page).control };
        let start = Instant::now();
        while control.asleep.load(Ordering::Acquire) < N_SYNAPSES as u32 {
            thread::park_timeout(Duration::from_millis(1));
            if start.elapsed().as_millis() > 4_000 {
                return Err("sentinel synapses did not park".into());
            }
        }
        Ok(())
    }

    fn wait_ready(&self) -> Result<(), String> {
        let control = unsafe { &(*self.page).control };
        let start = Instant::now();
        while control.ready.load(Ordering::Acquire) < N_SYNAPSES as u32 {
            thread::park_timeout(Duration::from_millis(1));
            if start.elapsed().as_millis() > 4_000 {
                return Err("sentinel synapses missed the epoch".into());
            }
        }
        Ok(())
    }

    fn wait_sensor(&self) -> Result<(), String> {
        let control = unsafe { &(*self.page).control };
        let start = Instant::now();
        while control.sensor_ready.load(Ordering::Acquire) == 0 {
            thread::park_timeout(Duration::from_millis(1));
            if start.elapsed().as_millis() > 4_000 {
                return Err("sentinel sensor did not start".into());
            }
        }
        Ok(())
    }
}

impl Drop for Sentinel {
    fn drop(&mut self) {
        if self.page.is_null() {
            return;
        }
        unsafe {
            let control = &(*self.page).control;
            control.stop.store(true, Ordering::Release);
            control.phase.store(PHASE_STOP, Ordering::Release);
            control.park_flag.store(0, Ordering::Release);
        }
        self.wake_all();
        for worker in std::mem::take(&mut self.threads) {
            let _ = worker.join();
        }
        unsafe {
            VirtualFree(self.page.cast(), 0, MEM_RELEASE);
            self.page = std::ptr::null_mut();
        }
    }
}

fn diagnostic_columns() -> [[f64; 4]; N_SYNAPSES] {
    let mut columns = [[0.0; N_CHANNELS]; N_SYNAPSES];
    for class in 0..N_CHANNELS {
        for synapse in 0..N_SYNAPSES {
            columns[synapse][class] = DIAGNOSTIC_MATRIX[class][synapse];
        }
    }
    columns
}

pub struct CpuStamp {
    pub idle: u64,
    pub kernel: u64,
    pub user: u64,
}

pub fn capture_cpu_stamp() -> CpuStamp {
    let mut idle = 0u64;
    let mut kernel = 0u64;
    let mut user = 0u64;
    let ok = unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) };
    if ok == 0 {
        return CpuStamp { idle: 0, kernel: 0, user: 1 };
    }
    CpuStamp { idle, kernel, user }
}

pub fn cpu_busy_fraction(previous: &CpuStamp, now: &CpuStamp) -> f64 {
    let idle = now.idle.saturating_sub(previous.idle);
    let kernel = now.kernel.saturating_sub(previous.kernel);
    let user = now.user.saturating_sub(previous.user);
    let busy = kernel.saturating_sub(idle).saturating_add(user);
    let total = kernel.saturating_add(user);
    if total == 0 { 0.0 } else { (busy as f64 / total as f64).clamp(0.0, 1.0) }
}

pub fn class_name(class: usize) -> &'static str {
    CLASS_NAMES.get(class).copied().unwrap_or("UNKNOWN")
}
