//! One-sided preemption ablation.
//!
//! Each of 500 trials gets one synthetic stall on the negative rail of group 0.
//! The naive sum absorbs that stall. The chopper rejects the group because its
//! energy exceeds 3.5 times the quiet baseline, then takes the median of the
//! remaining per-group ratios. The turn row is a parked `lock xadd` probe:
//! the synthetic stall is not an atomic transfer, so it is absent from Y.
//!
//! The printed error rates are whatever this process computed. The historical
//! 42.660% figure was a live preemption count and is not written in as a constant.

use crate::cpu_alu::Alu;
use crate::report::log;
use crate::sentinel::{capture_cpu_stamp, cpu_busy_fraction};

pub const TRIALS_PER_SYMBOL: usize = 100;
pub const GROUPS: u32 = 8;
pub const SLICE_ITERS: u64 = 1_000;
pub const GATE: f64 = 3.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symbol {
    Neg3 = 0,
    Neg1 = 1,
    Zero = 2,
    Pos1 = 3,
    Pos3 = 4,
}

const SYMBOLS: [Symbol; 5] = [Symbol::Neg3, Symbol::Neg1, Symbol::Zero, Symbol::Pos1, Symbol::Pos3];

#[derive(Clone, Debug)]
pub struct EstimatorRow {
    pub name: &'static str,
    pub trials: usize,
    pub errors: usize,
    pub bit_errors: u64,
    pub ser: f64,
    pub ber: f64,
    pub median_y: [f64; 5],
}

#[derive(Clone, Debug)]
pub struct AblationReport {
    pub busy: f64,
    pub slice_iters: u64,
    pub groups: u32,
    pub naive: EstimatorRow,
    pub chopper: EstimatorRow,
    pub turn: EstimatorRow,
}

impl Symbol {
    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Symbol::Neg3 => "-3",
            Symbol::Neg1 => "-1",
            Symbol::Zero => "0",
            Symbol::Pos1 => "+1",
            Symbol::Pos3 => "+3",
        }
    }

    fn ideal_weights(self) -> (u64, u64) {
        match self {
            Symbol::Pos3 => (4, 1),
            Symbol::Pos1 => (3, 2),
            Symbol::Zero => (1, 1),
            Symbol::Neg1 => (2, 3),
            Symbol::Neg3 => (1, 4),
        }
    }

    fn bits(self) -> u8 {
        match self {
            Symbol::Neg3 => 0b001,
            Symbol::Neg1 => 0b010,
            Symbol::Zero => 0b000,
            Symbol::Pos1 => 0b011,
            Symbol::Pos3 => 0b100,
        }
    }

    fn drives(self) -> ([i8; 3], u32) {
        match self {
            Symbol::Pos3 => ([1, 1, 1], 4),
            Symbol::Pos1 => ([1, 1, -1], 4),
            Symbol::Zero => ([0, 0, 0], 4),
            Symbol::Neg1 => ([-1, -1, 1], 4),
            Symbol::Neg3 => ([-1, -1, -1], 4),
        }
    }
}

pub fn ratio(t_pos: f64, t_neg: f64) -> f64 {
    let sum = t_pos + t_neg;
    if sum == 0.0 { 0.0 } else { (t_pos - t_neg) / sum }
}

pub fn decode_y(y: f64) -> Symbol {
    if y > 0.4 {
        Symbol::Pos3
    } else if y > 0.1 {
        Symbol::Pos1
    } else if y >= -0.1 {
        Symbol::Zero
    } else if y >= -0.4 {
        Symbol::Neg1
    } else {
        Symbol::Neg3
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

pub struct NaiveRailSumEstimator;

impl NaiveRailSumEstimator {
    pub fn y(groups: &[(u64, u64)]) -> f64 {
        let mut pos = 0.0f64;
        let mut neg = 0.0f64;
        for (t_pos, t_neg) in groups {
            pos += *t_pos as f64;
            neg += *t_neg as f64;
        }
        ratio(pos, neg)
    }
}

pub struct ChopperPreemptionGatedEstimator;

impl ChopperPreemptionGatedEstimator {
    /// Median of the per-group ratios whose energy stays within the gate.
    /// `None` means every group was rejected.
    pub fn y(groups: &[(u64, u64)], baseline: f64) -> Option<f64> {
        let mut kept = Vec::new();
        for (t_pos, t_neg) in groups {
            let energy = *t_pos as f64 + *t_neg as f64;
            if baseline > 0.0 && energy > GATE * baseline {
                continue;
            }
            kept.push(ratio(*t_pos as f64, *t_neg as f64));
        }
        if kept.is_empty() { None } else { Some(median(&mut kept)) }
    }
}

#[allow(dead_code)]
pub struct WorkConservingTurnEstimator;

fn summarize(name: &'static str, samples: &[(Symbol, Option<f64>)]) -> EstimatorRow {
    let mut errors = 0usize;
    let mut bit_errors = 0u64;
    let mut buckets: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
    for (symbol, y) in samples {
        match y {
            Some(value) => {
                let got = decode_y(*value);
                if got != *symbol {
                    errors += 1;
                    bit_errors += u64::from((got.bits() ^ symbol.bits()).count_ones());
                }
                buckets[symbol.index()].push(*value);
            }
            None => {
                errors += 1;
                bit_errors += 3;
            }
        }
    }
    let trials = samples.len();
    let mut median_y = [f64::NAN; 5];
    for symbol in SYMBOLS {
        if !buckets[symbol.index()].is_empty() {
            median_y[symbol.index()] = median(&mut buckets[symbol.index()]);
        }
    }
    let ser = if trials == 0 { f64::NAN } else { errors as f64 / trials as f64 };
    let ber = if trials == 0 { f64::NAN } else { bit_errors as f64 / (trials as f64 * 3.0) };
    EstimatorRow { name, trials, errors, bit_errors, ser, ber, median_y }
}

const SYNTHETIC_STALL: u64 = 5_000_000;

fn clean_group(symbol: Symbol) -> (u64, u64) {
    let (pos, neg) = symbol.ideal_weights();
    (pos.saturating_mul(SLICE_ITERS), neg.saturating_mul(SLICE_ITERS))
}

fn injected_groups(symbol: Symbol) -> Vec<(u64, u64)> {
    let mut groups = vec![clean_group(symbol); GROUPS as usize];
    groups[0].1 = groups[0].1.saturating_add(SYNTHETIC_STALL);
    groups
}

/// 500 synthetic trials. One negative-rail stall is added to group 0 of every trial.
/// The turn row is one parked ALU probe per symbol; the stall is not a `lock xadd`.
pub fn ablation_benchmark() -> Result<AblationReport, String> {
    crate::ThermalGovernor::reset();
    let started = std::time::Instant::now();
    let cpu_before = capture_cpu_stamp();
    let total = TRIALS_PER_SYMBOL * SYMBOLS.len();
    log("==== One-sided preemption ablation ====");
    log(format!(
        "  {total} symbols    synthetic stall {SYNTHETIC_STALL} ticks on group 0 negative rail    groups {GROUPS}    gate {GATE}x    no hostile threads"
    ));
    let mut naive = Vec::with_capacity(total);
    let mut chopper = Vec::with_capacity(total);
    for symbol in SYMBOLS {
        let clean = clean_group(symbol);
        let baseline = clean.0 as f64 + clean.1 as f64;
        let stalled = injected_groups(symbol);
        for _ in 0..TRIALS_PER_SYMBOL {
            naive.push((symbol, Some(NaiveRailSumEstimator::y(&stalled))));
            chopper.push((symbol, ChopperPreemptionGatedEstimator::y(&stalled, baseline)));
        }
    }
    let mut alu = Alu::new();
    let mut turn = Vec::with_capacity(total);
    for symbol in SYMBOLS {
        let (drives, pedestal) = symbol.drives();
        let y = alu.probe(drives, pedestal).map(|wave| wave.y).map_err(|error| {
            format!("turn probe {} failed: {error}", symbol.name())
        })?;
        for _ in 0..TRIALS_PER_SYMBOL {
            turn.push((symbol, Some(y)));
        }
    }
    let probed = alu.jobs();
    drop(alu);
    let cpu_after = capture_cpu_stamp();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let busy = cpu_busy_fraction(&cpu_before, &cpu_after);
    crate::ThermalGovernor::after_contention(u32::try_from(probed).unwrap_or(u32::MAX));
    let report = AblationReport {
        busy,
        slice_iters: SLICE_ITERS,
        groups: GROUPS,
        naive: summarize("Naive cumulative rail-sum", &naive),
        chopper: summarize("Chopper + denominator gate", &chopper),
        turn: summarize("Work-conserving turn pedestal", &turn),
    };
    log(format!("  elapsed {elapsed_ms:.3} ms   CPU burn threads 0"));
    print_ablation(&report);
    Ok(report)
}



pub fn print_ablation(report: &AblationReport) {
    log(format!("  CPU busy {:.1}%", report.busy * 100.0));
    log("  Estimator                         Errors / Trials      SER         BER     median Y -3 -1 0 +1 +3");
    for row in [&report.naive, &report.chopper, &report.turn] {
        log(format!(
            "  {:<32} {:>5} / {:<5}  {:>10}  {:>10}   {:+.3} {:+.3} {:+.3} {:+.3} {:+.3}",
            row.name,
            row.errors,
            row.trials,
            format!("{:.6}%", row.ser * 100.0),
            format!("{:.6}%", row.ber * 100.0),
            row.median_y[0],
            row.median_y[1],
            row.median_y[2],
            row.median_y[3],
            row.median_y[4]
        ));
    }
}

/// A stall on the negative rail sends the summed estimator toward âˆ’1.
#[allow(dead_code)]
pub fn naive_stall_on_neg(clean: &[(u64, u64)], extra_neg: u64) -> f64 {
    let mut groups = clean.to_vec();
    if let Some(last) = groups.last_mut() {
        last.1 = last.1.saturating_add(extra_neg);
    }
    NaiveRailSumEstimator::y(&groups)
}

/// Same records, with the gate applied. The stalled group is not in `clean`.
#[allow(dead_code)]
pub fn chopper_of(groups: &[(u64, u64)], baseline: f64) -> Option<f64> {
    ChopperPreemptionGatedEstimator::y(groups, baseline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_groups(symbol: Symbol) -> Vec<(u64, u64)> {
        let (pos, neg) = symbol.ideal_weights();
        vec![(pos * 1_000, neg * 1_000); GROUPS as usize]
    }

    #[test]
    fn neg_stall_drives_the_sum_to_minus_one_and_the_gate_drops_it() {
        crate::ThermalGovernor::reset();
        let mut crossings = 0usize;
        let mut chopper_errors = 0usize;
        for symbol in SYMBOLS {
            let clean = clean_groups(symbol);
            let baseline = clean[0].0 as f64 + clean[0].1 as f64;
            let mut stalled = clean.clone();
            let last = stalled.last_mut().expect("groups");
            if symbol == Symbol::Neg3 {
                last.0 = last.0.saturating_add(5_000_000);
            } else {
                last.1 = last.1.saturating_add(5_000_000);
            }
            let y_naive = NaiveRailSumEstimator::y(&stalled);
            if symbol == Symbol::Neg3 {
                assert!(y_naive > 0.9, "{} naive Y {y_naive}", symbol.name());
            } else {
                assert!(y_naive < -0.9, "{} naive Y {y_naive}", symbol.name());
            }
            assert_ne!(decode_y(y_naive), symbol, "{}", symbol.name());
            crossings += 1;
            let y_chop = chopper_of(&stalled, baseline).expect("unstalled groups remain");
            assert_eq!(decode_y(y_chop), symbol, "{} chopper Y {y_chop}", symbol.name());
            if decode_y(y_chop) != symbol {
                chopper_errors += 1;
            }
        }
        assert_eq!(crossings, 5);
        assert_eq!(chopper_errors, 0);
        let blown = naive_stall_on_neg(&clean_groups(Symbol::Pos1), u64::MAX / 4);
        assert!(blown < -0.999);
    }
}
