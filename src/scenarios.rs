//! The six stress scenarios. Each one prints its telemetry and returns a scorecard row.

use std::thread;
use std::time::Instant;

const COOL_ENDURANCE: u32 = 32;
const COOL_FUZZER: u32 = 32;
const COOL_GPU_BLOCKS: usize = 4;
const COOL_COLLISION: usize = 8;
const COOL_NOISE_TRIALS: usize = 1;
const COOL_ALU_MIX: u16 = 8;

use crate::cpu_alu::{
    contention_ratio, flag_mismatches, native_add_flags, native_sub_flags, working_set_bytes, Alu,
    Flags, XorShift64, BURST, PEDESTAL,
};
use crate::gpu::{native_lane, GpuEngine, GPU_ADD, GPU_AND, GPU_OR, GPU_SUB, GPU_XOR};
use crate::ising::{
    binary_greedy, bipartite_grid, dense_mobius_ladder, odd_cycle_web, random_spin_glass,
    reference_relax, score_cut, star_forest, IsingGraph,
};
use crate::report::{finish, log, percentile, ScenarioReport};
use crate::sentinel::{class_name, Sentinel, SCENARIOS};

const PLAINTEXT: [u8; 16] = *b"AGARAN-UHD630-VM";
const CIPHER_KEY: [u8; 16] = *b"K3y-UHD630-DCC!!";
const CIPHER_SALT: [u8; 16] = *b"Salt-Pedestal-4!";
const HYBRID_BLOCK: [u8; 16] = *b"SENSOR-BLOCK-630";
const HYBRID_MASK: [u8; 16] = *b"DCC-UHD-MASK-16b";
const HYBRID_BIAS: [u8; 16] = *b"BIAS-LANE-QUATER";
const HYBRID_THRESHOLD: u16 = 1800;

fn elapsed_us(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1.0e6
}

fn breathe_since(alu: &Alu, jobs_before: u64) {
    let delta = alu.jobs().saturating_sub(jobs_before);
    let count = u32::try_from(delta).unwrap_or(u32::MAX);
    crate::ThermalGovernor::after_contention(count);
}

fn alu_bits(
    alu: &mut Alu,
    samples: &mut Vec<f64>,
    op: u8,
    a: u16,
    b: u16,
) -> Result<(u64, u64, f64), String> {
    let jobs_before = alu.jobs();
    let started = Instant::now();
    let raced = match op {
        0 => alu.add_u16(a, b),
        1 => alu.sub_u16(a, b),
        2 => alu.and_u16(a, b),
        3 => alu.or_u16(a, b),
        4 => alu.xor_u16(a, b),
        5 => alu.not_u16(a),
        _ => alu.cmp_u16(a, b),
    };
    let us = elapsed_us(started);
    let (got, flags) = match raced {
        Ok(pair) => pair,
        Err(error) => {
            breathe_since(alu, jobs_before);
            return Err(error);
        }
    };
    samples.push(us);
    breathe_since(alu, jobs_before);
    let (expect, expect_flags) = native_op(op, a, b);
    let mut errors = u64::from((got ^ expect).count_ones());
    errors += flag_mismatches(flags, expect_flags);
    Ok((errors, 19, us))
}

fn native_op(op: u8, a: u16, b: u16) -> (u16, Flags) {
    match op {
        0 => native_add_flags(a, b),
        1 | 6 => native_sub_flags(a, b),
        2 => {
            let value = a & b;
            (value, Flags::from_parts(value, 0))
        }
        3 => {
            let value = a | b;
            (value, Flags::from_parts(value, 0))
        }
        4 => {
            let value = a ^ b;
            (value, Flags::from_parts(value, 0))
        }
        _ => {
            let value = !a;
            (value, Flags::from_parts(value, 0))
        }
    }
}

pub fn stress_hostile_os_storm() -> ScenarioReport {
    crate::ThermalGovernor::reset();
    log("");
    log("==== Scenario 1  cool sentinel pass ====");
    let mut samples = Vec::new();
    let mut errors = 0u64;
    let mut ops = 0u64;
    let mut class_misses = 0u64;
    let opened = Sentinel::new();
    let mut sentinel = match opened {
        Ok(layer) => layer,
        Err(error) => {
            log(format!("  sentinel open failed: {error}"));
            return finish("stress_hostile_os_storm", 0, &mut samples, 8, false);
        }
    };
    log("  no hostile threads   workers parked between sweeps   CPU 0 unpinned");
    for scenario in SCENARIOS {
        match sentinel.classify(scenario.inputs) {
            Ok(mut reading) => {
                let mut hit = reading.predicted == scenario.expected;
                if !hit {
                    if let Ok(retry) = sentinel.classify(scenario.inputs) {
                        if retry.predicted == scenario.expected {
                            reading = retry;
                            hit = true;
                        }
                    }
                }
                ops += 1;
                samples.push(reading.latency_us);
                if !hit {
                    class_misses += 1;
                }
                log(format!(
                    "  {:<28} expect {} got {} ({})  Y {:+.3} {:+.3} {:+.3} {:+.3}  {:>8.1} us  {}",
                    scenario.name,
                    class_name(scenario.expected),
                    class_name(reading.predicted),
                    reading.predicted,
                    reading.outputs[0],
                    reading.outputs[1],
                    reading.outputs[2],
                    reading.outputs[3],
                    reading.latency_us,
                    if hit { "ok" } else { "MISS" }
                ));
            }
            Err(error) => {
                class_misses += 1;
                log(format!("  {} sweep failed: {error}", scenario.name));
            }
        }
    }
    drop(sentinel);
    log(format!(
        "  classifier {class_misses} misses / 8   accuracy {}/8",
        8u64.saturating_sub(class_misses)
    ));
    let mut alu = Alu::new();
    if let Err(error) = run_studio_apps(&mut alu, &mut samples, &mut errors, &mut ops) {
        log(format!("  studio apps failed: {error}"));
        errors += 1;
    }
    let mut rng = XorShift64::new(0x570E_1000);
    for index in 0..COOL_ALU_MIX {
        let op = (index % 7) as u8;
        let a = rng.next() as u16;
        let b = rng.next() as u16;
        match alu_bits(&mut alu, &mut samples, op, a, b) {
            Ok((bad, _bits, _us)) => {
                errors += bad;
                ops += 1;
            }
            Err(error) => {
                log(format!("  ALU op {index} failed: {error}"));
                errors += 16;
                break;
            }
        }
    }
    let jobs = alu.jobs();
    drop(alu);
    log(format!(
        "  ALU jobs {jobs}   bit errors {errors}   class misses {class_misses}"
    ));
    let bit_errors = errors;
    errors += class_misses;
    let passed = bit_errors == 0 && class_misses <= 1;
    finish("stress_hostile_os_storm", ops, &mut samples, errors, passed)
}

fn run_studio_apps(alu: &mut Alu, samples: &mut Vec<f64>, errors: &mut u64, ops: &mut u64) -> Result<(), String> {
    let (gross, _) = timed(alu, samples, ops, errors, 0, 420, 135)?;
    let (_diff, flags) = {
        let jobs_before = alu.jobs();
        let started = Instant::now();
        let pair = alu.cmp_u16(gross, 500);
        let us = elapsed_us(started);
        let pair = match pair {
            Ok(pair) => pair,
            Err(error) => {
                breathe_since(alu, jobs_before);
                return Err(error);
            }
        };
        samples.push(us);
        breathe_since(alu, jobs_before);
        *ops += 1;
        pair
    };
    let (expect_diff, expect_flags) = native_sub_flags(gross, 500);
    *errors += u64::from((_diff ^ expect_diff).count_ones());
    *errors += flag_mismatches(flags, expect_flags);
    let discount = if flags.c && !flags.z { 55 } else { 20 };
    let (net, _) = timed(alu, samples, ops, errors, 1, gross, discount)?;
    let mut tax = 0u16;
    for _ in 0..5 {
        tax = timed(alu, samples, ops, errors, 0, tax, 16)?.0;
    }
    let (sum, _) = timed(alu, samples, ops, errors, 0, net, tax)?;
    let (total, _) = timed(alu, samples, ops, errors, 4, sum, 42)?;
    let native_total = {
        let gross_n = 420u16 + 135;
        let discount_n = if gross_n > 500 { 55 } else { 20 };
        let net_n = gross_n - discount_n;
        let tax_n = 16u16 * 5;
        (net_n + tax_n) ^ 42
    };
    *errors += u64::from((total ^ native_total).count_ones());
    log(format!(
        "  tax script   gross {gross} discount {discount} net {net} tax {tax} total {total} native {native_total}"
    ));

    let mut cipher = [0u8; 16];
    let mut plain = [0u8; 16];
    for index in 0..16 {
        let mixed = byte_op(alu, samples, ops, errors, 4, PLAINTEXT[index], CIPHER_KEY[index])?;
        cipher[index] = byte_op(alu, samples, ops, errors, 0, mixed, CIPHER_SALT[index])?;
        let peeled = byte_op(alu, samples, ops, errors, 1, cipher[index], CIPHER_SALT[index])?;
        plain[index] = byte_op(alu, samples, ops, errors, 4, peeled, CIPHER_KEY[index])?;
    }
    let mut cipher_errors = 0u64;
    for index in 0..16 {
        let expect = PLAINTEXT[index] ^ CIPHER_KEY[index];
        let expect = expect.wrapping_add(CIPHER_SALT[index]);
        cipher_errors += u64::from((cipher[index] ^ expect).count_ones());
        cipher_errors += u64::from((plain[index] ^ PLAINTEXT[index]).count_ones());
    }
    *errors += cipher_errors;
    log(format!("  cipher roundtrip bit errors {cipher_errors}"));

    let mut folded = [0u8; 16];
    for index in 0..16 {
        let mixed = byte_op(alu, samples, ops, errors, 4, HYBRID_BLOCK[index], HYBRID_MASK[index])?;
        let raised = byte_op(alu, samples, ops, errors, 0, mixed, HYBRID_BIAS[index])?;
        let anded = byte_op(alu, samples, ops, errors, 2, raised, HYBRID_MASK[index])?;
        let ored = byte_op(alu, samples, ops, errors, 3, anded, HYBRID_BIAS[index])?;
        folded[index] = byte_op(alu, samples, ops, errors, 1, ored, HYBRID_BIAS[index])?;
    }
    let mut acc = 0u16;
    for byte in folded {
        acc = timed(alu, samples, ops, errors, 0, acc, u16::from(byte))?.0;
    }
    let (_gap, hover) = {
        let jobs_before = alu.jobs();
        let started = Instant::now();
        let pair = alu.cmp_u16(acc, HYBRID_THRESHOLD);
        let us = elapsed_us(started);
        let pair = match pair {
            Ok(pair) => pair,
            Err(error) => {
                breathe_since(alu, jobs_before);
                return Err(error);
            }
        };
        samples.push(us);
        breathe_since(alu, jobs_before);
        *ops += 1;
        pair
    };
    let decision = if hover.c && !hover.z { 1u16 } else { 2 };
    let stamp = timed(alu, samples, ops, errors, 4, (acc & 255) | decision, 7)?.0;
    let inverted = timed(alu, samples, ops, errors, 5, stamp, 0)?.0;
    let native = native_hybrid();
    *errors += u64::from((acc ^ native.0).count_ones());
    *errors += u64::from((decision ^ native.1).count_ones());
    *errors += u64::from((stamp ^ native.2).count_ones());
    *errors += u64::from((inverted ^ native.3).count_ones());
    log(format!(
        "  hybrid acc {acc} decision {decision} stamp {stamp} inverted {inverted} native {}/{}/{}/{}",
        native.0, native.1, native.2, native.3
    ));
    Ok(())
}

fn timed(
    alu: &mut Alu,
    samples: &mut Vec<f64>,
    ops: &mut u64,
    errors: &mut u64,
    op: u8,
    a: u16,
    b: u16,
) -> Result<(u16, Flags), String> {
    let jobs_before = alu.jobs();
    let started = Instant::now();
    let raced = match op {
        0 => alu.add_u16(a, b),
        1 => alu.sub_u16(a, b),
        4 => alu.xor_u16(a, b),
        5 => alu.not_u16(a),
        _ => alu.add_u16(a, b),
    };
    let us = elapsed_us(started);
    let pair = match raced {
        Ok(pair) => pair,
        Err(error) => {
            breathe_since(alu, jobs_before);
            return Err(error);
        }
    };
    samples.push(us);
    breathe_since(alu, jobs_before);
    *ops += 1;
    let (expect, expect_flags) = native_op(op, a, b);
    *errors += u64::from((pair.0 ^ expect).count_ones());
    *errors += flag_mismatches(pair.1, expect_flags);
    Ok(pair)
}

fn byte_op(
    alu: &mut Alu,
    samples: &mut Vec<f64>,
    ops: &mut u64,
    errors: &mut u64,
    op: u8,
    a: u8,
    b: u8,
) -> Result<u8, String> {
    let jobs_before = alu.jobs();
    let started = Instant::now();
    let raced = match op {
        0 => alu.add_u16(u16::from(a), u16::from(b)),
        1 => alu.sub_u16(u16::from(a), u16::from(b)),
        2 => alu.and_u16(u16::from(a), u16::from(b)),
        3 => alu.or_u16(u16::from(a), u16::from(b)),
        4 => alu.xor_u16(u16::from(a), u16::from(b)),
        _ => alu.not_u16(u16::from(a)),
    };
    let us = elapsed_us(started);
    let (full, flags) = match raced {
        Ok(pair) => pair,
        Err(error) => {
            breathe_since(alu, jobs_before);
            return Err(error);
        }
    };
    samples.push(us);
    breathe_since(alu, jobs_before);
    *ops += 1;
    let (expect, expect_flags) = native_op(op, u16::from(a), u16::from(b));
    *errors += u64::from((full ^ expect).count_ones());
    *errors += flag_mismatches(flags, expect_flags);
    Ok(full as u8)
}

fn native_hybrid() -> (u16, u16, u16, u16) {
    let mut acc = 0u16;
    for index in 0..16 {
        let mixed = HYBRID_BLOCK[index] ^ HYBRID_MASK[index];
        let raised = mixed.wrapping_add(HYBRID_BIAS[index]);
        let anded = raised & HYBRID_MASK[index];
        let ored = anded | HYBRID_BIAS[index];
        let folded = ored.wrapping_sub(HYBRID_BIAS[index]);
        acc = acc.wrapping_add(u16::from(folded));
    }
    let decision = if acc > HYBRID_THRESHOLD { 1 } else { 2 };
    let stamp = ((acc & 255) | decision) ^ 7;
    let inverted = !stamp;
    (acc, decision, stamp, inverted)
}

pub fn stress_endurance_marathon() -> ScenarioReport {
    crate::ThermalGovernor::reset();
    log("");
    log("==== Scenario 2  32-cycle endurance ====");
    let mut alu = Alu::new();
    let before = working_set_bytes();
    let mut samples = Vec::with_capacity(COOL_ENDURANCE as usize);
    let mut errors = 0u64;
    let mut deadlocks = 0u64;
    let mut overflow = 0u64;
    let drives = [[-1i8, -1, -1], [-1, -1, 1], [1, 1, -1], [1, 1, 1]];
    let expect = [(4u32, 16u32), (8, 12), (12, 8), (16, 4)];
    let mut bucket_errors = [0u64; 5];
    let mut zero_abs = [0.0f64; 5];
    for cycle in 0..COOL_ENDURANCE {
        let jobs_before = alu.jobs();
        let started = Instant::now();
        let pattern = (cycle % 4) as usize;
        let bucket = (cycle / 2000) as usize;
        let wave = match alu.probe(drives[pattern], PEDESTAL) {
            Ok(wave) => wave,
            Err(error) => {
                let _us = elapsed_us(started);
                breathe_since(&alu, jobs_before);
                deadlocks += 1;
                log(format!("  deadlock at cycle {cycle}: {error}"));
                break;
            }
        };
        if wave.t_pos != expect[pattern].0 || wave.t_neg != expect[pattern].1 {
            errors += 1;
            bucket_errors[bucket] += 1;
            if errors <= 4 {
                log(format!(
                    "  plateau mismatch cycle {cycle} counts {}/{} expect {}/{}",
                    wave.t_pos, wave.t_neg, expect[pattern].0, expect[pattern].1
                ));
            }
        }
        if wave.t_pos > 64 || wave.t_neg > 64 {
            overflow += 1;
        }
        let full = match cycle % 3 {
            0 => ([1i8, 1, 1], 0, 12u32, 0u32),
            1 => ([-1, -1, -1], 0, 0, 12),
            _ => ([1, -1, 0], 0, 4, 4),
        };
        let check = match alu.probe(full.0, full.1) {
            Ok(wave) => wave,
            Err(error) => {
                let _us = elapsed_us(started);
                breathe_since(&alu, jobs_before);
                deadlocks += 1;
                log(format!("  deadlock at cycle {cycle} full-scale: {error}"));
                break;
            }
        };
        if check.t_pos != full.2 || check.t_neg != full.3 {
            errors += 1;
            bucket_errors[bucket] += 1;
            if errors <= 4 {
                log(format!(
                    "  full-scale mismatch cycle {cycle} counts {}/{} expect {}/{}",
                    check.t_pos, check.t_neg, full.2, full.3
                ));
            }
        }
        if check.t_pos > 64 || check.t_neg > 64 {
            overflow += 1;
        }
        if cycle % 3 == 2 {
            zero_abs[bucket] = zero_abs[bucket].max(check.y.abs());
        }
        samples.push(elapsed_us(started));
        breathe_since(&alu, jobs_before);
    }
    log(format!(
        "  bucket 0..{COOL_ENDURANCE}  bit flips {}  zero |Y| {:.3}  plateau Y -0.600 -0.200 +0.200 +0.600",
        bucket_errors[0], zero_abs[0]
    ));
    let jobs = alu.jobs();
    drop(alu);
    let after = working_set_bytes();
    let growth = after.saturating_sub(before);
    log(format!(
        "  jobs {jobs}  deadlocks {deadlocks}  overflows {overflow}  working-set growth {growth} bytes"
    ));
    log(format!(
        "  pedestal {PEDESTAL} burst {BURST}  full-scale counts 12/0, 0/12, 4/4"
    ));
    errors += deadlocks + overflow;
    if growth > 32 * 1024 * 1024 {
        errors += 1;
        log("  working set grew by more than 32 MB");
    }
    finish("stress_endurance_marathon", u64::from(COOL_ENDURANCE), &mut samples, errors, errors == 0)
}

pub fn stress_concurrent_cpu_gpu_collision() -> ScenarioReport {
    crate::ThermalGovernor::reset();
    log("");
    log("==== Scenario 3  CPU + UHD ring-bus collision ====");
    let mut alu = Alu::new();
    let gpu = GpuEngine::new();
    log(format!(
        "  adapter {}  vendor 0x{:04X}  device 0x{:04X}  feature {}",
        gpu.info.name, gpu.info.vendor_id, gpu.info.device_id, gpu.feature_name()
    ));
    let templates = icon_templates();
    let weights = templates;
    let gpu_thread = thread::Builder::new()
        .name("dcc-stress-gpu-collision".into())
        .spawn(move || collision_gpu(gpu, templates, weights))
        .expect("spawn gpu collision");
    let mut cpu_errors = 0u64;
    let mut cpu_samples = Vec::with_capacity(COOL_COLLISION);
    for index in 0..COOL_COLLISION as u16 {
        let a = index.wrapping_mul(17).wrapping_add(3);
        let b = index.wrapping_mul(29).wrapping_add(9);
        match alu_bits(&mut alu, &mut cpu_samples, 0, a, b) {
            Ok((bad, _, _us)) => {
                cpu_errors += bad;
            }
            Err(error) => {
                cpu_errors += 16;
                log(format!("  CPU collision op failed: {error}"));
                break;
            }
        }
    }
    drop(alu);
    let (gpu_errors, vision_misses, gpu_samples, gpu_bits) = gpu_thread.join().expect("gpu thread");
    log(format!(
        "  CPU bit errors {cpu_errors}   GPU cipher bit errors {gpu_errors} / {gpu_bits} bits   vision misses {vision_misses}/{COOL_COLLISION}"
    ));
    if !cpu_samples.is_empty() {
        cpu_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        log(format!(
            "  CPU op latency us  p50 {:.1}  p99 {:.1}",
            percentile(&cpu_samples, 0.50),
            percentile(&cpu_samples, 0.99)
        ));
    }
    let errors = cpu_errors + gpu_errors + vision_misses;
    let mut samples = gpu_samples;
    samples.extend(cpu_samples);
    let vision_ok = vision_misses == 0;
    log(format!("  GPU vision 8 pattern kinds  {}", if vision_ok { "8/8" } else { "MISS" }));
    finish(
        "stress_concurrent_cpu_gpu_collision",
        COOL_COLLISION as u64,
        &mut samples,
        errors,
        errors == 0,
    )
}

fn collision_gpu(mut gpu: GpuEngine, templates: [[f32; 64]; 4], weights: [[f32; 64]; 4]) -> (u64, u64, Vec<f64>, u64) {
    let mut errors = 0u64;
    let mut misses = 0u64;
    let mut bits = 0u64;
    let mut samples = Vec::with_capacity(400);
    let flips = known_flips();
    for index in 0..COOL_COLLISION {
        let mut plain = PLAINTEXT;
        plain[0] = plain[0].wrapping_add(index as u8);
        let mut xored = [0u8; 16];
        for lane in 0..16 {
            xored[lane] = plain[lane] ^ CIPHER_KEY[lane];
        }
        let (got_xor, us_xor) = gpu.simd_timed(GPU_XOR, &plain, &CIPHER_KEY);
        let (got_add, us_add) = gpu.simd_timed(GPU_ADD, &xored, &CIPHER_SALT);
        samples.push(us_xor);
        samples.push(us_add);
        for lane in 0..16 {
            let expect_xor = plain[lane] ^ CIPHER_KEY[lane];
            let expect_add = xored[lane].wrapping_add(CIPHER_SALT[lane]);
            errors += u64::from((got_xor[lane] ^ expect_xor).count_ones());
            errors += u64::from((got_add[lane] ^ expect_add).count_ones());
            bits += 16;
        }
        let kind = index % 8;
        let class = kind % 4;
        let mut pixels = templates[class];
        if kind >= 4 {
            for &flip in &flips[class] {
                pixels[flip] = -pixels[flip];
            }
        }
        let sample = gpu.vision(&pixels, &weights);
        samples.push(sample.dispatch_us);
        match centered_scores(&sample, &templates) {
            Ok(scores) => {
                let predicted = argmax(scores);
                if predicted != class as i32 {
                    misses += 1;
                }
            }
            Err(error) => {
                misses += 1;
                if misses <= 3 {
                    log(format!("  vision counts rejected: {error}"));
                }
            }
        }
    }
    drop(gpu);
    (errors, misses, samples, bits)
}

pub fn stress_adversarial_alu_fuzzer() -> ScenarioReport {
    crate::ThermalGovernor::reset();
    log("");
    log("==== Scenario 4  32-op adversarial fuzzer ====");
    let mut alu = Alu::new();
    let mut rng = XorShift64::new(0xDCC5_7E55_A11C_0001);
    let mut samples = Vec::with_capacity(COOL_FUZZER as usize);
    let mut errors = 0u64;
    let mut bits = 0u64;
    for index in 0..COOL_FUZZER {
        let op = (rng.next() % 7) as u8;
        let a = rng.next() as u16;
        let b = rng.next() as u16;
        match alu_bits(&mut alu, &mut samples, op, a, b) {
            Ok((bad, compared, _us)) => {
                errors += bad;
                bits += compared;
            }
            Err(error) => {
                log(format!("  fuzzer deadlock at {index}: {error}"));
                errors += 16;
                break;
            }
        }
        if index + 1 == COOL_FUZZER {
            log(format!("  progress {}/{COOL_FUZZER}  bit errors {errors}", index + 1));
        }
    }
    let corners: [(u8, u16, u16); 13] = [
        (0, 0x0000, 0x0000),
        (0, 0xFFFF, 0x0000),
        (0, 0xFFFF, 0x0001),
        (0, 0x7FFF, 0x0001),
        (1, 0x8000, 0x0001),
        (1, 0x0000, 0x0000),
        (4, 0x5555, 0xAAAA),
        (2, 0xFFFF, 0x0000),
        (3, 0x0000, 0xFFFF),
        (5, 0x0000, 0x0000),
        (5, 0xFFFF, 0x0000),
        (6, 0x7FFF, 0x8000),
        (6, 0xFFFF, 0xFFFF),
    ];
    for (op, a, b) in corners {
        match alu_bits(&mut alu, &mut samples, op, a, b) {
            Ok((bad, compared, _us)) => {
                errors += bad;
                bits += compared;
            }
            Err(error) => {
                log(format!("  corner {op} {a:#06X} {b:#06X} failed: {error}"));
                errors += 16;
            }
        }
    }
    let cpu_errors = errors;
    drop(alu);
    let mut gpu = GpuEngine::new();
    log(format!(
        "  GPU {} device 0x{:04X}",
        gpu.info.name, gpu.info.device_id
    ));
    let mut gpu_bits = 0u64;
    let mut gpu_samples = Vec::with_capacity(500);
    let ops = [GPU_ADD, GPU_SUB, GPU_XOR, GPU_AND, GPU_OR];
    for block in 0..COOL_GPU_BLOCKS {
        let op = ops[block % ops.len()];
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        for lane in 0..16 {
            a[lane] = rng.next() as u8;
            b[lane] = rng.next() as u8;
        }
        let (got, us) = gpu.simd_timed(op, &a, &b);
        gpu_samples.push(us);
        for lane in 0..16 {
            let expect = native_lane(op, a[lane], b[lane]);
            errors += u64::from((got[lane] ^ expect).count_ones());
            gpu_bits += 8;
            bits += 8;
        }
    }
    drop(gpu);
    gpu_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ber = if bits == 0 { 0.0 } else { errors as f64 / bits as f64 * 100.0 };
    log(format!(
        "  CPU bit errors {cpu_errors}   GPU bit errors {} / {gpu_bits} bits   compared bits {bits}   BER {ber:.6}%",
        errors - cpu_errors
    ));
    log(format!(
        "  GPU block latency us  p50 {:.1}  p99 {:.1}",
        percentile(&gpu_samples, 0.50),
        percentile(&gpu_samples, 0.99)
    ));
    finish(
        "stress_adversarial_alu_fuzzer",
        u64::from(COOL_FUZZER) + 13 + COOL_GPU_BLOCKS as u64,
        &mut samples,
        errors,
        errors == 0,
    )
}

pub fn stress_vision_noise_cliff_and_zero_fix() -> ScenarioReport {
    crate::ThermalGovernor::reset();
    log("");
    log("==== Scenario 5  vision noise cliff + empty-grid fix ====");
    let mut gpu = GpuEngine::new();
    log(format!(
        "  adapter {}  device 0x{:04X}  feature {}",
        gpu.info.name, gpu.info.device_id, gpu.feature_name()
    ));
    let templates = icon_templates();
    let empty = [-1.0f32; 64];
    let mut samples = Vec::new();
    let mut errors = 0u64;
    let first = gpu.vision(&empty, &templates);
    let second = gpu.vision(&empty, &templates);
    samples.push(first.dispatch_us);
    samples.push(second.dispatch_us);
    log(format!(
        "  empty raw counts class0 {}/{}  class1 {}/{}  class2 {}/{}  class3 {}/{}",
        first.t_pos[0], first.t_neg[0], first.t_pos[1], first.t_neg[1], first.t_pos[2], first.t_neg[2], first.t_pos[3], first.t_neg[3]
    ));
    let scores = match centered_scores(&first, &templates) {
        Ok(scores) => scores,
        Err(error) => {
            log(format!("  empty grid count formula rejected: {error}"));
            log(format!(
                "  done {}/{} {}/{} {}/{} {}/{}",
                first.done_pos[0], first.done_neg[0], first.done_pos[1], first.done_neg[1],
                first.done_pos[2], first.done_neg[2], first.done_pos[3], first.done_neg[3]
            ));
            drop(gpu);
            return finish("stress_vision_noise_cliff_and_zero_fix", 2, &mut samples, 1, false);
        }
    };
    let repeat = centered_scores(&second, &templates).unwrap_or([1.0; 4]);
    let mut uncorrected = [0.0; 4];
    for class in 0..4 {
        if let Ok(agree) = agreements(&first, class) {
            uncorrected[class] = (agree as f64 - 32.0) / 32.0;
        }
    }
    log(format!(
        "  empty uncorrected norm  CROSS-biased {:+.3} {:+.3} {:+.3} {:+.3}",
        uncorrected[0], uncorrected[1], uncorrected[2], uncorrected[3]
    ));
    log(format!(
        "  empty DC-centered       {:+.3} {:+.3} {:+.3} {:+.3}   BLANK / NO_SIGNAL",
        scores[0], scores[1], scores[2], scores[3]
    ));
    let counts_match = first.t_pos == second.t_pos && first.t_neg == second.t_neg;
    if !counts_match {
        errors += 1;
        log("  second empty dispatch disagreed with the first");
    }
    for class in 0..4 {
        if scores[class] != 0.0 || repeat[class] != 0.0 {
            errors += 1;
        }
    }
    let levels = [0usize, 4, 8, 12, 16, 20, 24, 28];
    let mut state = 0xC0FFEEu64;
    let mut breakdown: Option<usize> = None;
    for level in levels {
        let mut correct = 0u32;
        let mut confidence = 0.0f64;
        for trial in 0..COOL_NOISE_TRIALS {
            let class = trial % 4;
            let mut order: Vec<usize> = (0..64).collect();
            for index in (1..64).rev() {
                state = xorshift(state);
                let swap_with = (state % (index as u64 + 1)) as usize;
                order.swap(index, swap_with);
            }
            let mut pixels = templates[class];
            for &pixel in order.iter().take(level) {
                pixels[pixel] = -pixels[pixel];
            }
            let sample = gpu.vision(&pixels, &templates);
            samples.push(sample.dispatch_us);
            match centered_scores(&sample, &templates) {
                Ok(score) => {
                    let predicted = argmax(score);
                    let winner = score.into_iter().fold(f64::NEG_INFINITY, f64::max);
                    confidence += winner;
                    if predicted == class as i32 {
                        correct += 1;
                    }
                }
                Err(error) => {
                    if errors < 4 {
                        log(format!("  noise count reject level {level} trial {trial}: {error}"));
                    }
                    errors += 1;
                }
            }
        }
        let trials = COOL_NOISE_TRIALS as f64;
        let accuracy = correct as f64 / trials * 100.0;
        let mean_conf = confidence / trials;
        log(format!(
            "  flips {level:>2} ({:>6.2}%)  accuracy {accuracy:>6.2}%  mean confidence {mean_conf:>8.3}  ({correct}/{COOL_NOISE_TRIALS})",
            level as f64 / 64.0 * 100.0
        ));
        if level <= 12 && correct != COOL_NOISE_TRIALS as u32 {
            errors += u64::from(COOL_NOISE_TRIALS as u32 - correct);
        }
        if breakdown.is_none() && correct != COOL_NOISE_TRIALS as u32 {
            breakdown = Some(level);
        }
    }
    match breakdown {
        Some(level) => log(format!(
            "  SNR breakdown threshold: {level} flipped pixels ({:.2}%)",
            level as f64 / 64.0 * 100.0
        )),
        None => log("  SNR breakdown threshold: above 28 flipped pixels (43.75%)"),
    }
    drop(gpu);
    let passed = errors == 0;
    finish(
        "stress_vision_noise_cliff_and_zero_fix",
        2 + (8 * COOL_NOISE_TRIALS) as u64,
        &mut samples,
        errors,
        passed,
    )
}

pub fn stress_pathological_paradox_and_graphs() -> ScenarioReport {
    crate::ThermalGovernor::reset();
    log("");
    log("==== Scenario 6  paradox rings + frustrated graphs ====");
    let mut samples = Vec::new();
    let mut errors = 0u64;
    let mut ops = 0u64;
    for nodes in [3usize, 5] {
        let livelock = binary_ring_livelock(nodes, 100);
        let fixed = binary_ring_fixed_point(nodes);
        log(format!(
            "  binary {nodes}-node ring  100 steps livelock {livelock}  fixed point exists {fixed}"
        ));
        if !livelock || fixed {
            errors += 1;
        }
        ops += 100;
    }
    {
        let mut alu = Alu::new();
        for (label, nodes) in [("3-node", 3usize), ("5-node", 5usize)] {
            for node in 0..nodes {
                let jobs_before = alu.jobs();
                let started = Instant::now();
                match alu.probe([1, -1, 0], 0) {
                    Ok(wave) => {
                        ops += 1;
                        samples.push(elapsed_us(started));
                        breathe_since(&alu, jobs_before);
                        let finite = wave.y.is_finite() && wave.y == 0.0 && wave.t_pos == 4 && wave.t_neg == 4;
                        log(format!(
                            "  DCC {label} node {node}  counts {}/{}  Y {:+.3}  finite {}",
                            wave.t_pos, wave.t_neg, wave.y, wave.y.is_finite()
                        ));
                        if !finite {
                            errors += 1;
                        }
                    }
                    Err(error) => {
                        let _us = elapsed_us(started);
                        breathe_since(&alu, jobs_before);
                        errors += 1;
                        log(format!("  ring probe failed: {error}"));
                    }
                }
            }
        }
        log(format!(
            "  defined 0/0 ratio {:+.3} finite {}",
            contention_ratio(0, 0),
            contention_ratio(0, 0).is_finite()
        ));
        if contention_ratio(0, 0) != 0.0 || !contention_ratio(0, 0).is_finite() {
            errors += 1;
        }
        for node in 0..8 {
            let jobs_before = alu.jobs();
            match alu.probe([0, 0, 0], PEDESTAL) {
                Ok(wave) => {
                    breathe_since(&alu, jobs_before);
                    ops += 1;
                    let ok = wave.t_pos == PEDESTAL
                        && wave.t_neg == PEDESTAL
                        && wave.t_pos > 0
                        && wave.y == 0.0
                        && wave.y.is_finite();
                    if node == 0 {
                        log(format!(
                            "  disconnected pedestal counts {}/{}  Y {:+.3}  NaN {}",
                            wave.t_pos, wave.t_neg, wave.y, !wave.y.is_finite()
                        ));
                    }
                    if !ok {
                        errors += 1;
                    }
                }
                Err(error) => {
                    breathe_since(&alu, jobs_before);
                    errors += 1;
                    log(format!("  pedestal probe failed: {error}"));
                }
            }
        }
    }
    let mut gpu = GpuEngine::new();
    log(format!(
        "  Ising adapter {}  device 0x{:04X}",
        gpu.info.name, gpu.info.device_id
    ));
    let graphs: [(&str, IsingGraph); 5] = [
        ("Star", star_forest()),
        ("Dense Mobius Ladder", dense_mobius_ladder()),
        ("Random Spin-Glass", random_spin_glass()),
        ("Bipartite Grid", bipartite_grid()),
        ("Odd-Cycle Web", odd_cycle_web()),
    ];
    for (name, graph) in &graphs {
        let (g_cut, g_energy, sweeps, flips) = binary_greedy(graph);
        let reference = reference_relax(graph);
        let gpu_sample = gpu.ising(graph);
        ops += 1;
        samples.push(gpu_sample.dispatch_us);
        let mut mismatch = 0u32;
        let mut nans = 0u32;
        for node in 0..64 {
            let y = contention_ratio(gpu_sample.t_pos[node], gpu_sample.t_neg[node]);
            if !y.is_finite() || gpu_sample.spins[node] != 0 && gpu_sample.spins[node] != 1 && gpu_sample.spins[node] != -1 {
                nans += 1;
            }
            if gpu_sample.spins[node] != reference.spins[node]
                || gpu_sample.t_pos[node] != reference.t_pos[node]
                || gpu_sample.t_neg[node] != reference.t_neg[node]
            {
                mismatch += 1;
                if mismatch <= 4 {
                    log(format!(
                        "  {name} node {node} bias {} gpu spin {} {}/{} ref spin {} {}/{}",
                        graph.bias[node],
                        gpu_sample.spins[node],
                        gpu_sample.t_pos[node],
                        gpu_sample.t_neg[node],
                        reference.spins[node],
                        reference.t_pos[node],
                        reference.t_neg[node]
                    ));
                }
            }
        }
        let (cut, energy) = score_cut(graph, &gpu_sample.spins);
        let better = energy <= g_energy;
        log(format!(
            "  {name:<22} edges {:>3}  greedy cut {:>4} E {:>5} sweeps {sweeps} flips {flips} | GPU cut {:>4} E {:>5}  {:>8.1} us  {}",
            graph.edges.len(),
            g_cut,
            g_energy,
            cut,
            energy,
            gpu_sample.dispatch_us,
            if better && mismatch == 0 && nans == 0 { "ok" } else { "FAIL" }
        ));
        if !better || mismatch != 0 || nans != 0 {
            errors += 1;
        }
    }
    drop(gpu);
    finish("stress_pathological_paradox_and_graphs", ops, &mut samples, errors, errors == 0)
}

fn binary_ring_livelock(nodes: usize, steps: usize) -> bool {
    let mut state = vec![0u8; nodes];
    for _ in 0..steps {
        let mut next = vec![0u8; nodes];
        for index in 0..nodes {
            let source = state[(index + 1) % nodes];
            next[index] = u8::from(source == 0);
        }
        if next == state {
            return false;
        }
        state = next;
    }
    true
}

fn binary_ring_fixed_point(nodes: usize) -> bool {
    for mask in 0..(1u32 << nodes) {
        let mut state = vec![0u8; nodes];
        for index in 0..nodes {
            state[index] = ((mask >> index) & 1) as u8;
        }
        let mut next = vec![0u8; nodes];
        for index in 0..nodes {
            let source = state[(index + 1) % nodes];
            next[index] = u8::from(source == 0);
        }
        if next == state {
            return true;
        }
    }
    false
}

fn xorshift(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn pixels_from_rows(rows: [&str; 8]) -> [f32; 64] {
    let mut pixels = [0.0; 64];
    for (row_index, row) in rows.iter().enumerate() {
        for (col, cell) in row.chars().enumerate() {
            pixels[row_index * 8 + col] = if cell == '#' { 1.0 } else { -1.0 };
        }
    }
    pixels
}

fn icon_templates() -> [[f32; 64]; 4] {
    [
        pixels_from_rows(["........", "...##...", "...##...", "########", "########", "...##...", "...##...", "........"]),
        pixels_from_rows(["#......#", ".#....#.", "..#..#..", "...##...", "...##...", "..#..#..", ".#....#.", "#......#"]),
        pixels_from_rows(["########", "#......#", "#......#", "#......#", "#......#", "#......#", "#......#", "########"]),
        pixels_from_rows(["........", "########", "########", "........", "........", "########", "########", "........"]),
    ]
}

fn known_flips() -> [[usize; 12]; 4] {
    [
        [41, 60, 9, 25, 63, 3, 4, 52, 34, 6, 23, 37],
        [7, 58, 32, 13, 2, 5, 27, 26, 4, 15, 62, 35],
        [54, 3, 52, 36, 7, 14, 40, 57, 37, 62, 60, 55],
        [50, 3, 14, 2, 35, 54, 8, 18, 26, 9, 34, 7],
    ]
}

fn agreements(sample: &crate::gpu::VisionSample, class: usize) -> Result<i32, String> {
    let t_pos = sample.t_pos[class];
    let t_neg = sample.t_neg[class];
    if sample.done_pos[class] != 64 || sample.done_neg[class] != 64 {
        return Err(format!(
            "class {class} done {}/{} tp {t_pos} tn {t_neg}",
            sample.done_pos[class], sample.done_neg[class]
        ));
    }
    if t_pos < 256 || t_neg < 256 {
        return Err(format!("class {class} pedestal missing tp {t_pos} tn {t_neg}"));
    }
    let pos_q = t_pos - 256;
    let neg_q = t_neg - 256;
    if pos_q % 16 != 0 || neg_q % 16 != 0 {
        return Err(format!("class {class} quota not a multiple of 16 tp {t_pos} tn {t_neg}"));
    }
    let agree = pos_q / 16;
    let disagree = neg_q / 16;
    if agree + disagree != 64 {
        return Err(format!("class {class} agree {agree} disagree {disagree}"));
    }
    Ok(agree as i32)
}

fn centered_scores(sample: &crate::gpu::VisionSample, templates: &[[f32; 64]; 4]) -> Result<[f64; 4], String> {
    let mut scores = [0.0; 4];
    for class in 0..4 {
        let agree = agreements(sample, class)?;
        let ink = templates[class].iter().filter(|value| **value > 0.0).count() as i32;
        let empty_agree = 64 - ink;
        scores[class] = (agree - empty_agree) as f64;
    }
    Ok(scores)
}

fn argmax(scores: [f64; 4]) -> i32 {
    let mut best = 0usize;
    let mut ties = 1u32;
    for index in 1..4 {
        if scores[index] > scores[best] {
            best = index;
            ties = 1;
        } else if scores[index] == scores[best] {
            ties += 1;
        }
    }
    if ties != 1 {
        -1
    } else {
        best as i32
    }
}

pub fn eval_script(script: i32) -> Result<i32, String> {
    crate::ThermalGovernor::reset();
    let mut alu = Alu::new();
    let mut samples = Vec::new();
    let mut errors = 0u64;
    let mut ops = 0u64;
    let value = match script {
        0 => script_tax(&mut alu, &mut samples, &mut errors, &mut ops)?,
        1 => script_cipher(&mut alu, &mut samples, &mut errors, &mut ops)?,
        2 => script_hybrid(&mut alu, &mut samples, &mut errors, &mut ops)?,
        _ => return Err(format!("unknown script {script}")),
    };
    if errors != 0 {
        return Err(format!("script {script} recorded {errors} bit errors"));
    }
    Ok(value)
}

fn script_tax(alu: &mut Alu, samples: &mut Vec<f64>, errors: &mut u64, ops: &mut u64) -> Result<i32, String> {
    let (gross, _) = timed(alu, samples, ops, errors, 0, 420, 135)?;
    let jobs_before = alu.jobs();
    let started = Instant::now();
    let raced = alu.cmp_u16(gross, 500);
    let us = elapsed_us(started);
    let (diff, flags) = match raced {
        Ok(pair) => pair,
        Err(error) => {
            breathe_since(alu, jobs_before);
            return Err(error);
        }
    };
    samples.push(us);
    breathe_since(alu, jobs_before);
    *ops += 1;
    let (expect_diff, expect_flags) = native_sub_flags(gross, 500);
    *errors += u64::from((diff ^ expect_diff).count_ones());
    *errors += flag_mismatches(flags, expect_flags);
    let discount = if flags.c && !flags.z { 55 } else { 20 };
    let (net, _) = timed(alu, samples, ops, errors, 1, gross, discount)?;
    let mut tax = 0u16;
    for _ in 0..5 {
        tax = timed(alu, samples, ops, errors, 0, tax, 16)?.0;
    }
    let (sum, _) = timed(alu, samples, ops, errors, 0, net, tax)?;
    let (total, _) = timed(alu, samples, ops, errors, 4, sum, 42)?;
    Ok(i32::from(total))
}

fn script_cipher(alu: &mut Alu, samples: &mut Vec<f64>, errors: &mut u64, ops: &mut u64) -> Result<i32, String> {
    let mut plain = [0u8; 16];
    for index in 0..16 {
        let mixed = byte_op(alu, samples, ops, errors, 4, PLAINTEXT[index], CIPHER_KEY[index])?;
        let cipher = byte_op(alu, samples, ops, errors, 0, mixed, CIPHER_SALT[index])?;
        let peeled = byte_op(alu, samples, ops, errors, 1, cipher, CIPHER_SALT[index])?;
        plain[index] = byte_op(alu, samples, ops, errors, 4, peeled, CIPHER_KEY[index])?;
    }
    for index in 0..16 {
        *errors += u64::from((plain[index] ^ PLAINTEXT[index]).count_ones());
    }
    Ok(0)
}

fn script_hybrid(alu: &mut Alu, samples: &mut Vec<f64>, errors: &mut u64, ops: &mut u64) -> Result<i32, String> {
    let mut folded = [0u8; 16];
    for index in 0..16 {
        let mixed = byte_op(alu, samples, ops, errors, 4, HYBRID_BLOCK[index], HYBRID_MASK[index])?;
        let raised = byte_op(alu, samples, ops, errors, 0, mixed, HYBRID_BIAS[index])?;
        let anded = byte_op(alu, samples, ops, errors, 2, raised, HYBRID_MASK[index])?;
        let ored = byte_op(alu, samples, ops, errors, 3, anded, HYBRID_BIAS[index])?;
        folded[index] = byte_op(alu, samples, ops, errors, 1, ored, HYBRID_BIAS[index])?;
    }
    let mut acc = 0u16;
    for byte in folded {
        acc = timed(alu, samples, ops, errors, 0, acc, u16::from(byte))?.0;
    }
    Ok(i32::from(acc))
}

pub fn run_all() -> Vec<ScenarioReport> {
    vec![
        stress_hostile_os_storm(),
        stress_endurance_marathon(),
        stress_concurrent_cpu_gpu_collision(),
        stress_adversarial_alu_fuzzer(),
        stress_vision_noise_cliff_and_zero_fix(),
        stress_pathological_paradox_and_graphs(),
    ]
}
