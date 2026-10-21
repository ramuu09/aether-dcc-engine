//! Quaternary `lock xadd` ALU.
//!
//! Pedestal 4 produces the studio plateaus. Pedestal 0 produces full-scale Â±1
//! and the cancelling paradox pair. Workers stay at below-normal priority,
//! unpinned, and parked on a condvar between jobs so logical CPU 0 is left
//! to the desktop.

use std::ffi::c_void;
use std::sync::atomic::{AtomicI8, AtomicU32, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::machine::{cpu_lock, elevate};

#[allow(dead_code)]
pub const SENSOR_CPU: usize = 0;
pub const SYNAPSE_CPUS: [usize; 3] = [2, 4, 6];
pub const BURST: u32 = 4;
pub const PEDESTAL: u32 = 4;
pub const CHANNELS: usize = 8;

const PAGE_BYTES: usize = 4096;
const CHANNEL_STRIDE: usize = 256;
const STEP_SHUTDOWN: u32 = u32::MAX;
const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 0x04;
const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
const TURN_TIMEOUT: Duration = Duration::from_secs(8);

#[link(name = "kernel32", kind = "raw-dylib")]
extern "system" {
    fn GetCurrentThread() -> *mut c_void;
    fn GetCurrentProcess() -> *mut c_void;
    fn SetThreadPriority(thread: *mut c_void, priority: i32) -> i32;
    fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
}

#[repr(C)]
struct ProcessMemoryCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set_size: usize,
    working_set_size: usize,
    quota_peak_paged_pool_usage: usize,
    quota_paged_pool_usage: usize,
    quota_peak_non_paged_pool_usage: usize,
    quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}

#[repr(C)]
struct Channel {
    pos: AtomicU32,
    _gap0: [u8; 60],
    neg: AtomicU32,
    _gap1: [u8; 60],
    _tail: [u8; 128],
}

#[repr(C)]
struct Page {
    channels: [Channel; CHANNELS],
    _tail: [u8; 2048],
}

const _: () = assert!(std::mem::size_of::<Channel>() == CHANNEL_STRIDE);
const _: () = assert!(std::mem::size_of::<Page>() == PAGE_BYTES);
const _: () = assert!(std::mem::offset_of!(Channel, neg) == 64);

struct TurnState {
    shutdown: bool,
    ready: u32,
    job: u32,
    finished: u32,
    token: u32,
    acks: [u32; 3],
}

struct Control {
    turn: Mutex<TurnState>,
    cv: Condvar,
    mask: AtomicU32,
    pedestal: AtomicU32,
    drives: [[AtomicI8; CHANNELS]; 3],
    t_pos: [AtomicU32; CHANNELS],
    t_neg: [AtomicU32; CHANNELS],
    sign_bit: [AtomicU8; CHANNELS],
    sum_bit: [AtomicU8; CHANNELS],
    xor_bit: [AtomicU8; CHANNELS],
    plateau: [AtomicI8; CHANNELS],
}

#[derive(Clone, Copy, Debug)]
pub struct Wave {
    pub t_pos: u32,
    pub t_neg: u32,
    pub y: f64,
    pub plateau: i32,
    pub sign_bit: u8,
    pub sum_bit: u8,
    pub xor_bit: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flags {
    pub z: bool,
    pub n: bool,
    pub c: bool,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum BitField {
    Sign,
    Sum,
    Xor,
}

pub fn contention_ratio(t_pos: u32, t_neg: u32) -> f64 {
    let sum = u64::from(t_pos) + u64::from(t_neg);
    if sum == 0 {
        0.0
    } else {
        (f64::from(t_pos) - f64::from(t_neg)) / sum as f64
    }
}

pub fn decode_counts(t_pos: u32, t_neg: u32) -> Wave {
    let delta = t_pos as i32 - t_neg as i32;
    let step = BURST as i32;
    let plateau = if delta > 2 * step {
        3
    } else if delta > 0 {
        1
    } else if delta < -2 * step {
        -3
    } else if delta < 0 {
        -1
    } else {
        0
    };
    let sum_bit = match plateau {
        3 | -1 => 1,
        _ => 0,
    };
    Wave {
        t_pos,
        t_neg,
        y: contention_ratio(t_pos, t_neg),
        plateau,
        sign_bit: u8::from(delta > 0),
        sum_bit,
        xor_bit: u8::from(delta.unsigned_abs() > BURST),
    }
}

fn bipolar(bit: u8) -> i8 {
    if bit == 0 { -1 } else { 1 }
}

fn demote_worker() {
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

fn lock_turn(control: &Control) -> std::sync::MutexGuard<'_, TurnState> {
    control.turn.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn active_planes(mask: u32) -> impl Iterator<Item = usize> {
    (0..CHANNELS).filter(move |plane| (mask & (1 << plane)) != 0)
}

fn sensor_worker(page: usize, control: usize) {
    demote_worker();
    let page = page as *mut Page;
    let control = unsafe { &*(control as *const Control) };
    {
        let mut state = lock_turn(control);
        state.ready += 1;
        control.cv.notify_all();
    }
    let mut seen_job = 0u32;
    loop {
        let mut state = lock_turn(control);
        while state.job == seen_job && !state.shutdown {
            state = control.cv.wait(state).unwrap_or_else(|poison| poison.into_inner());
        }
        if state.shutdown {
            state.token = STEP_SHUTDOWN;
            control.cv.notify_all();
            while state.acks.iter().any(|ack| *ack != STEP_SHUTDOWN) {
                state = control.cv.wait(state).unwrap_or_else(|poison| poison.into_inner());
            }
            break;
        }
        let job = state.job;
        drop(state);
        if run_sensor_job(page, control).is_err() {
            let mut state = lock_turn(control);
            state.shutdown = true;
            state.token = STEP_SHUTDOWN;
            control.cv.notify_all();
            break;
        }
        let mut state = lock_turn(control);
        state.finished = job;
        control.cv.notify_all();
        seen_job = job;
    }
}

fn run_sensor_job(page: *mut Page, control: &Control) -> Result<(), String> {
    let mask = control.mask.load(Ordering::Acquire);
    let pedestal = control.pedestal.load(Ordering::Acquire);
    unsafe {
        for plane in active_planes(mask) {
            let slot = &(*page).channels[plane];
            slot.pos.store(0, Ordering::Release);
            slot.neg.store(0, Ordering::Release);
        }
    }
    for step in 0..BURST {
        let published = {
            let mut state = lock_turn(control);
            state.token = state.token.wrapping_add(1);
            let published = state.token;
            control.cv.notify_all();
            published
        };
        if step < pedestal {
            unsafe {
                for plane in active_planes(mask) {
                    let slot = &(*page).channels[plane];
                    let _ = std::hint::black_box(slot.pos.fetch_add(1, Ordering::AcqRel));
                    let _ = std::hint::black_box(slot.neg.fetch_add(1, Ordering::AcqRel));
                }
            }
        }
        let mut state = lock_turn(control);
        let started = Instant::now();
        while state.acks.iter().any(|ack| *ack != published) && !state.shutdown {
            let (guard, wait) = control
                .cv
                .wait_timeout(state, TURN_TIMEOUT)
                .unwrap_or_else(|poison| poison.into_inner());
            state = guard;
            if wait.timed_out() || started.elapsed() > TURN_TIMEOUT {
                return Err("sensor waiting for synapse turn timed out after 8000 ms".to_string());
            }
        }
        if state.shutdown && state.acks.iter().any(|ack| *ack != published) {
            return Err("sensor shut down while waiting for a synapse turn".to_string());
        }
    }
    unsafe {
        for plane in active_planes(mask) {
            let slot = &(*page).channels[plane];
            let wave = decode_counts(
                slot.pos.load(Ordering::Acquire),
                slot.neg.load(Ordering::Acquire),
            );
            control.t_pos[plane].store(wave.t_pos, Ordering::Release);
            control.t_neg[plane].store(wave.t_neg, Ordering::Release);
            control.sign_bit[plane].store(wave.sign_bit, Ordering::Release);
            control.sum_bit[plane].store(wave.sum_bit, Ordering::Release);
            control.xor_bit[plane].store(wave.xor_bit, Ordering::Release);
            control.plateau[plane].store(wave.plateau as i8, Ordering::Release);
        }
    }
    Ok(())
}

fn synapse_worker(page: usize, control: usize, role: usize) {
    demote_worker();
    let page = page as *mut Page;
    let control = unsafe { &*(control as *const Control) };
    {
        let mut state = lock_turn(control);
        state.ready += 1;
        control.cv.notify_all();
    }
    let mut seen = 0u32;
    loop {
        let token = {
            let mut state = lock_turn(control);
            while state.token == seen && !state.shutdown {
                state = control.cv.wait(state).unwrap_or_else(|poison| poison.into_inner());
            }
            if state.shutdown || state.token == STEP_SHUTDOWN {
                state.acks[role] = STEP_SHUTDOWN;
                control.cv.notify_all();
                break;
            }
            state.token
        };
        let mask = control.mask.load(Ordering::Acquire);
        unsafe {
            for plane in active_planes(mask) {
                let drive = control.drives[role][plane].load(Ordering::Acquire);
                let slot = &(*page).channels[plane];
                if drive > 0 {
                    let _ = std::hint::black_box(slot.pos.fetch_add(1, Ordering::AcqRel));
                } else if drive < 0 {
                    let _ = std::hint::black_box(slot.neg.fetch_add(1, Ordering::AcqRel));
                }
            }
        }
        seen = token;
        let mut state = lock_turn(control);
        state.acks[role] = token;
        control.cv.notify_all();
    }
}

pub struct Alu {
    page: *mut Page,
    control: *mut Control,
    threads: Vec<JoinHandle<()>>,
    next_job: u32,
    jobs: u64,
    fault: Option<String>,
    _guard: MutexGuard<'static, ()>,
}

impl Alu {
    pub fn new() -> Self {
        elevate();
        let guard = cpu_lock();
        let page = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                PAGE_BYTES,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        if page.is_null() {
            panic!("VirtualAlloc failed");
        }
        let page = page.cast::<Page>();
        unsafe {
            for channel in &mut (*page).channels {
                std::ptr::write(&mut channel.pos, AtomicU32::new(0));
                std::ptr::write(&mut channel.neg, AtomicU32::new(0));
            }
        }
        let control = Box::into_raw(Box::new(Control {
            turn: Mutex::new(TurnState {
                shutdown: false,
                ready: 0,
                job: 0,
                finished: 0,
                token: 0,
                acks: [0, 0, 0],
            }),
            cv: Condvar::new(),
            mask: AtomicU32::new(0),
            pedestal: AtomicU32::new(PEDESTAL),
            drives: std::array::from_fn(|_| std::array::from_fn(|_| AtomicI8::new(0))),
            t_pos: std::array::from_fn(|_| AtomicU32::new(0)),
            t_neg: std::array::from_fn(|_| AtomicU32::new(0)),
            sign_bit: std::array::from_fn(|_| AtomicU8::new(0)),
            sum_bit: std::array::from_fn(|_| AtomicU8::new(0)),
            xor_bit: std::array::from_fn(|_| AtomicU8::new(0)),
            plateau: std::array::from_fn(|_| AtomicI8::new(0)),
        }));
        let mut threads = Vec::with_capacity(4);
        let page_addr = page as usize;
        let control_addr = control as usize;
        threads.push(
            thread::Builder::new()
                .name("dcc-stress-sensor".into())
                .spawn(move || sensor_worker(page_addr, control_addr))
                .expect("spawn sensor"),
        );
        for role in 0..3 {
            let page_addr = page as usize;
            let control_addr = control as usize;
            threads.push(
                thread::Builder::new()
                    .name(format!("dcc-stress-synapse-{role}"))
                    .spawn(move || synapse_worker(page_addr, control_addr, role))
                    .expect("spawn synapse"),
            );
        }
        let alu = Self {
            page,
            control,
            threads,
            next_job: 1,
            jobs: 0,
            fault: None,
            _guard: guard,
        };
        let ready = unsafe { &*alu.control };
        {
            let mut state = lock_turn(ready);
            let started = Instant::now();
            while state.ready < 4 {
                let (guard, wait) = ready
                    .cv
                    .wait_timeout(state, TURN_TIMEOUT)
                    .unwrap_or_else(|poison| poison.into_inner());
                state = guard;
                if wait.timed_out() || started.elapsed() > TURN_TIMEOUT {
                    panic!("workers did not start within 8000 ms");
                }
            }
        }
        alu
    }

    pub fn jobs(&self) -> u64 {
        self.jobs
    }

    pub fn fault(&self) -> Option<&str> {
        self.fault.as_deref()
    }

    pub fn probe(&mut self, drives: [i8; 3], pedestal: u32) -> Result<Wave, String> {
        let mut plane = [[0i8; CHANNELS]; 3];
        for role in 0..3 {
            plane[role][0] = drives[role];
        }
        Ok(self.contend(0b0000_0001, plane, pedestal)?[0])
    }

    fn logic_byte(&mut self, a: u8, b: u8, bias: i8, field: BitField) -> Result<u8, String> {
        let mut drives = [[0i8; CHANNELS]; 3];
        for bit in 0..CHANNELS {
            drives[0][bit] = bipolar((a >> bit) & 1);
            drives[1][bit] = bipolar((b >> bit) & 1);
            drives[2][bit] = bias;
        }
        let waves = self.contend(0xFF, drives, PEDESTAL)?;
        Ok(pack_byte(&waves, field))
    }

    fn xor_byte(&mut self, a: u8, b: u8) -> Result<u8, String> {
        let mut drives = [[0i8; CHANNELS]; 3];
        for bit in 0..CHANNELS {
            drives[0][bit] = bipolar((a >> bit) & 1);
            drives[1][bit] = -bipolar((b >> bit) & 1);
        }
        Ok(pack_byte(&self.contend(0xFF, drives, PEDESTAL)?, BitField::Xor))
    }

    fn not_byte(&mut self, a: u8) -> Result<u8, String> {
        let mut drives = [[0i8; CHANNELS]; 3];
        for bit in 0..CHANNELS {
            drives[0][bit] = -bipolar((a >> bit) & 1);
        }
        Ok(pack_byte(&self.contend(0xFF, drives, PEDESTAL)?, BitField::Sign))
    }

    fn add_byte(&mut self, a: u8, b: u8, mut cin: u8, subtract: bool) -> Result<(u8, u8), String> {
        let mut sum = 0u8;
        for bit in 0..CHANNELS {
            let mut drives = [[0i8; CHANNELS]; 3];
            drives[0][bit] = bipolar((a >> bit) & 1);
            let b_bit = (b >> bit) & 1;
            drives[1][bit] = if subtract { -bipolar(b_bit) } else { bipolar(b_bit) };
            drives[2][bit] = bipolar(cin);
            let raced = self.contend(1 << bit, drives, PEDESTAL)?;
            if raced[bit].sum_bit == 1 {
                sum |= 1 << bit;
            }
            cin = raced[bit].sign_bit;
        }
        Ok((sum, cin))
    }

    pub fn add_u16(&mut self, a: u16, b: u16) -> Result<(u16, Flags), String> {
        let a_bytes = a.to_le_bytes();
        let b_bytes = b.to_le_bytes();
        let (lo, cin) = self.add_byte(a_bytes[0], b_bytes[0], 0, false)?;
        let (hi, cout) = self.add_byte(a_bytes[1], b_bytes[1], cin, false)?;
        let value = u16::from_le_bytes([lo, hi]);
        Ok((value, Flags::from_parts(value, cout)))
    }

    pub fn sub_u16(&mut self, a: u16, b: u16) -> Result<(u16, Flags), String> {
        let a_bytes = a.to_le_bytes();
        let b_bytes = b.to_le_bytes();
        let (lo, cin) = self.add_byte(a_bytes[0], b_bytes[0], 1, true)?;
        let (hi, cout) = self.add_byte(a_bytes[1], b_bytes[1], cin, true)?;
        let value = u16::from_le_bytes([lo, hi]);
        Ok((value, Flags::from_parts(value, cout)))
    }

    pub fn and_u16(&mut self, a: u16, b: u16) -> Result<(u16, Flags), String> {
        let value = u16::from_le_bytes([
            self.logic_byte(a as u8, b as u8, -1, BitField::Sign)?,
            self.logic_byte((a >> 8) as u8, (b >> 8) as u8, -1, BitField::Sign)?,
        ]);
        Ok((value, Flags::from_parts(value, 0)))
    }

    pub fn or_u16(&mut self, a: u16, b: u16) -> Result<(u16, Flags), String> {
        let value = u16::from_le_bytes([
            self.logic_byte(a as u8, b as u8, 1, BitField::Sign)?,
            self.logic_byte((a >> 8) as u8, (b >> 8) as u8, 1, BitField::Sign)?,
        ]);
        Ok((value, Flags::from_parts(value, 0)))
    }

    pub fn xor_u16(&mut self, a: u16, b: u16) -> Result<(u16, Flags), String> {
        let value = u16::from_le_bytes([
            self.xor_byte(a as u8, b as u8)?,
            self.xor_byte((a >> 8) as u8, (b >> 8) as u8)?,
        ]);
        Ok((value, Flags::from_parts(value, 0)))
    }

    pub fn not_u16(&mut self, a: u16) -> Result<(u16, Flags), String> {
        let value = u16::from_le_bytes([self.not_byte(a as u8)?, self.not_byte((a >> 8) as u8)?]);
        Ok((value, Flags::from_parts(value, 0)))
    }

    pub fn cmp_u16(&mut self, a: u16, b: u16) -> Result<(u16, Flags), String> {
        self.sub_u16(a, b)
    }

    fn contend(
        &mut self,
        mask: u8,
        drives: [[i8; CHANNELS]; 3],
        pedestal: u32,
    ) -> Result<[Wave; CHANNELS], String> {
        if let Some(error) = &self.fault {
            return Err(error.clone());
        }
        let control = unsafe { &*self.control };
        for role in 0..3 {
            for plane in 0..CHANNELS {
                control.drives[role][plane].store(drives[role][plane], Ordering::Release);
            }
        }
        control.mask.store(u32::from(mask), Ordering::Release);
        control.pedestal.store(pedestal, Ordering::Release);
        let job = self.next_job;
        self.next_job = self.next_job.wrapping_add(1);
        if self.next_job == 0 {
            self.next_job = 1;
        }
        {
            let mut state = lock_turn(control);
            state.job = job;
            control.cv.notify_all();
            let started = Instant::now();
            while state.finished != job && !state.shutdown {
                let (guard, wait) = control
                    .cv
                    .wait_timeout(state, TURN_TIMEOUT)
                    .unwrap_or_else(|poison| poison.into_inner());
                state = guard;
                if wait.timed_out() || started.elapsed() > TURN_TIMEOUT {
                    let error = "contention job timed out after 8000 ms".to_string();
                    self.fault = Some(error.clone());
                    return Err(error);
                }
            }
            if state.shutdown && state.finished != job {
                let error = "contention worker shut down before the job finished".to_string();
                self.fault = Some(error.clone());
                return Err(error);
            }
        }
        self.jobs += 1;
        let mut waves = [decode_counts(0, 0); CHANNELS];
        for plane in active_planes(u32::from(mask)) {
            let t_pos = control.t_pos[plane].load(Ordering::Acquire);
            let t_neg = control.t_neg[plane].load(Ordering::Acquire);
            let mut wave = decode_counts(t_pos, t_neg);
            wave.sign_bit = control.sign_bit[plane].load(Ordering::Acquire);
            wave.sum_bit = control.sum_bit[plane].load(Ordering::Acquire);
            wave.xor_bit = control.xor_bit[plane].load(Ordering::Acquire);
            wave.plateau = i32::from(control.plateau[plane].load(Ordering::Acquire));
            waves[plane] = wave;
        }
        Ok(waves)
    }
}

impl Drop for Alu {
    fn drop(&mut self) {
        let control = unsafe { &*self.control };
        {
            let mut state = lock_turn(control);
            state.shutdown = true;
            state.token = STEP_SHUTDOWN;
            control.cv.notify_all();
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        unsafe {
            drop(Box::from_raw(self.control));
            if VirtualFree(self.page.cast(), 0, MEM_RELEASE) == 0 {
                panic!("VirtualFree failed");
            }
        }
    }
}

impl Flags {
    pub fn from_parts(value: u16, cout: u8) -> Self {
        Self {
            z: value == 0,
            n: (value & 0x8000) != 0,
            c: cout == 1,
        }
    }
}

fn pack_byte(waves: &[Wave; CHANNELS], field: BitField) -> u8 {
    let mut byte = 0u8;
    for bit in 0..CHANNELS {
        let on = match field {
            BitField::Sign => waves[bit].sign_bit,
            BitField::Sum => waves[bit].sum_bit,
            BitField::Xor => waves[bit].xor_bit,
        };
        if on == 1 {
            byte |= 1 << bit;
        }
    }
    byte
}

pub fn working_set_bytes() -> usize {
    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        0
    } else {
        counters.working_set_size
    }
}

pub fn native_add_flags(a: u16, b: u16) -> (u16, Flags) {
    let wide = u32::from(a) + u32::from(b);
    let value = wide as u16;
    let cout = u8::from(wide >> 16 != 0);
    (value, Flags::from_parts(value, cout))
}

pub fn native_sub_flags(a: u16, b: u16) -> (u16, Flags) {
    let wide = u32::from(a) + u32::from(!b) + 1;
    let value = wide as u16;
    let cout = u8::from(wide >> 16 != 0);
    (value, Flags::from_parts(value, cout))
}

#[allow(dead_code)]
pub fn flag_mismatches(got: Flags, expect: Flags) -> u64 {
    u64::from(got.z != expect.z) + u64::from(got.n != expect.n) + u64::from(got.c != expect.c)
}

pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "xorshift64 seed");
        Self { state: seed }
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

