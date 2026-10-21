//! Executive product benchmark: BitNet cliff, C ABI smoke, then the six stress scenarios.

use crate::abi::{
    dcc_sdk_eval_script, dcc_sdk_ghost_cipher, dcc_sdk_infer_digit10, dcc_sdk_solve_ising64, dcc_sdk_version,
};
use crate::bitnet::{self, argmax};
use crate::gpu::GpuEngine;
use crate::ising::{binary_greedy, graph_from_raw, star_forest};
use crate::report::{log, percentile, print_scorecard};
use crate::scenarios::run_all;

pub struct DigitQa {
    pub clean: u32,
    pub band_misses: u64,
    pub mismatches: u64,
    pub blank_max: i32,
    pub breakdown: Option<usize>,
    pub p50: f64,
    pub p99: f64,
}

pub fn digit_qa(gpu: &mut GpuEngine) -> DigitQa {
    crate::ThermalGovernor::reset();
    log("");
    log("==== BitNet 64â†’16â†’10  DC-centered digit net ====");
    let (zeros, total) = bitnet::weight_census();
    log(format!(
        "  ternary weights  {zeros}/{total} zeros  ({:.2}% skipped InterlockedAdd)",
        zeros as f64 / total as f64 * 100.0
    ));
    let mut samples = Vec::new();
    let mut mismatches = 0u64;
    let blank_pixels = [-1.0f32; 64];
    let first = gpu.bitnet(&blank_pixels);
    let second = gpu.bitnet(&blank_pixels);
    samples.push(first.dispatch_us);
    samples.push(second.dispatch_us);
    let host_blank = bitnet::reference(&blank_pixels);
    if first.scores != host_blank.scores || first.hidden != host_blank.hidden || first.scores != second.scores {
        mismatches += 1;
        log(format!(
            "  blank GPU scores {:?}  host {:?}",
            first.scores, host_blank.scores
        ));
    }
    let blank_max = first.scores.iter().map(|score| score.abs()).max().unwrap_or(0);
    log(format!(
        "  empty canvas scores {:+} {:+} {:+} {:+} {:+} {:+} {:+} {:+} {:+} {:+}   {}",
        first.scores[0], first.scores[1], first.scores[2], first.scores[3], first.scores[4],
        first.scores[5], first.scores[6], first.scores[7], first.scores[8], first.scores[9],
        if blank_max == 0 { "BLANK / NO_SIGNAL" } else { "BIAS" }
    ));
    let mut clean = 0u32;
    for digit in 0..10 {
        let pixels = bitnet::glyph_f32(digit);
        let sample = gpu.bitnet(&pixels);
        samples.push(sample.dispatch_us);
        let host = bitnet::reference(&pixels);
        if sample.scores != host.scores || sample.hidden != host.hidden {
            mismatches += 1;
            if mismatches <= 4 {
                log(format!("  digit {digit} GPU/host score mismatch {:?} vs {:?}", sample.scores, host.scores));
            }
        }
        if argmax(sample.scores) == digit as i32 {
            clean += 1;
        }
    }
    log(format!("  clean glyphs {clean}/10"));
    let levels = [0usize, 4, 8, 12, 16, 20, 24, 28];
    let mut state = 0xC0FFEEu64;
    let mut band_misses = 0u64;
    let mut breakdown = None;
    for level in levels {
        let mut correct = 0u32;
        let mut confidence = 0.0f64;
        for trial in 0..crate::COOL_DIGIT_TRIALS {
            let digit = trial % 10;
            let pixels = bitnet::corrupt(&bitnet::glyph_f32(digit), level, &mut state);
            let sample = gpu.bitnet(&pixels);
            samples.push(sample.dispatch_us);
            let host = bitnet::reference(&pixels);
            if sample.scores != host.scores {
                mismatches += 1;
            }
            let predicted = argmax(sample.scores);
            let winner = sample.scores.iter().copied().max().unwrap_or(0);
            confidence += f64::from(winner);
            if predicted == digit as i32 {
                correct += 1;
            }
        }
        let trials = crate::COOL_DIGIT_TRIALS as f64;
        let accuracy = correct as f64 / trials * 100.0;
        log(format!(
            "  flips {level:>2} ({:>6.2}%)  accuracy {accuracy:>6.2}%  mean winner score {:>7.2}  ({correct}/{})",
            level as f64 / 64.0 * 100.0,
            confidence / trials,
            crate::COOL_DIGIT_TRIALS
        ));
        if level <= 12 && correct != crate::COOL_DIGIT_TRIALS as u32 {
            band_misses += u64::from(crate::COOL_DIGIT_TRIALS as u32 - correct);
        }
        if breakdown.is_none() && correct != crate::COOL_DIGIT_TRIALS as u32 {
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
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let qa = DigitQa {
        clean,
        band_misses,
        mismatches,
        blank_max,
        breakdown,
        p50: percentile(&samples, 0.50),
        p99: percentile(&samples, 0.99),
    };
    log(format!(
        "  digit latency us  p50 {:.1}  p99 {:.1}  gpu/host mismatches {}",
        qa.p50, qa.p99, qa.mismatches
    ));
    qa
}

pub fn product_benchmark() -> i32 {
    log("AetherDCC Studio v1.0 â€” product benchmark");
    let version = unsafe { std::ffi::CStr::from_ptr(dcc_sdk_version()) };
    log(format!("  C ABI version {}", version.to_string_lossy()));
    let mut failures = 0u32;
    failures += abi_smoke();
    let mut gpu = GpuEngine::new();
    log(format!(
        "  adapter {}  vendor 0x{:04X}  device 0x{:04X}  feature {}",
        gpu.info.name, gpu.info.vendor_id, gpu.info.device_id, gpu.feature_name()
    ));
    let digits = digit_qa(&mut gpu);
    drop(gpu);
    let digits_ok = digits.clean == 10 && digits.band_misses == 0 && digits.mismatches == 0 && digits.blank_max == 0;
    if !digits_ok {
        failures += 1;
    }
    log("");
    log("==== Six-scenario stress suite ====");
    let rows = run_all();
    print_scorecard(&rows);
    if rows.iter().any(|row| !row.passed) {
        failures += 1;
    }
    log("");
    let ablation = match crate::estimator_ablation::ablation_benchmark() {
        Ok(report) => report,
        Err(error) => {
            log(format!("  ablation harness error: {error}"));
            failures += 1;
            log(format!(
                "  product pillars  bitnet {}  stress {}  ablation FAIL",
                if digits_ok { "PASS" } else { "FAIL" },
                if rows.iter().all(|row| row.passed) { "PASS" } else { "FAIL" }
            ));
            return i32::from(failures != 0);
        }
    };
    let ablation_ok = ablation.turn.errors == 0
        && ablation.chopper.ser < 0.001
        && ablation.naive.errors > ablation.chopper.errors;
    if !ablation_ok {
        failures += 1;
    }
    log(format!(
        "  product pillars  bitnet {}  stress {}  ablation {}  abi failures before the suite counted above",
        if digits_ok { "PASS" } else { "FAIL" },
        if rows.iter().all(|row| row.passed) { "PASS" } else { "FAIL" },
        if ablation_ok { "PASS" } else { "FAIL" }
    ));
    i32::from(failures != 0)
}

fn abi_smoke() -> u32 {
    log("");
    log("==== C ABI smoke ====");
    let mut failures = 0u32;
    let image = bitnet::glyph_f32(3);
    let mut class = -99i32;
    let mut scores = [0i32; 10];
    let mut latency = 0.0f64;
    let status = unsafe { dcc_sdk_infer_digit10(image.as_ptr(), &mut class, scores.as_mut_ptr(), &mut latency) };
    log(format!("  infer digit 3  status {status}  class {class}  {latency:.1} us"));
    if status != 0 || class != 3 {
        failures += 1;
    }
    let plain = *b"AGARAN-UHD630-VM";
    let key = *b"K3y-UHD630-DCC!!";
    let salt = *b"Salt-Pedestal-4!";
    let mut cipher = [0u8; 16];
    let status = unsafe {
        dcc_sdk_ghost_cipher(plain.as_ptr(), key.as_ptr(), salt.as_ptr(), cipher.as_mut_ptr(), &mut latency)
    };
    log(format!(
        "  ghost cipher status {status}  {latency:.1} us  {}",
        cipher.iter().map(|byte| format!("{byte:02X}")).collect::<Vec<_>>().join(" ")
    ));
    if status != 0 {
        failures += 1;
    }
    for (script, expect) in [(0, 622), (1, 0), (2, 136)] {
        let mut result = -1i32;
        let status = unsafe { dcc_sdk_eval_script(script, &mut result) };
        log(format!("  eval script {script}  status {status}  result {result}  expect {expect}"));
        if status != 0 || result != expect {
            failures += 1;
        }
    }
    let star = star_forest();
    let mut neighbors = [-1i32; 512];
    let mut coupling = [0i32; 512];
    let mut bias = [0i32; 64];
    for node in 0..64 {
        bias[node] = star.bias[node];
        for slot in 0..8 {
            neighbors[node * 8 + slot] = star.neighbors[node][slot];
            coupling[node * 8 + slot] = star.coupling[node][slot];
        }
    }
    let mut spins = [0i32; 64];
    let mut energy = 0i32;
    let status = unsafe {
        dcc_sdk_solve_ising64(
            neighbors.as_ptr(),
            coupling.as_ptr(),
            bias.as_ptr(),
            spins.as_mut_ptr(),
            &mut energy,
            &mut latency,
        )
    };
    let rebuilt = graph_from_raw(&neighbors, &coupling, &bias).expect("star raw graph");
    let (greedy_cut, greedy_energy, _, _) = binary_greedy(&rebuilt);
    log(format!(
        "  ising star status {status}  GPU E {energy}  greedy E {greedy_energy} cut {greedy_cut}  {latency:.1} us"
    ));
    if status != 0 || energy > greedy_energy {
        failures += 1;
    }
    failures
}
