use std::io::{self, Write};

#[derive(Clone, Debug)]
pub struct ScenarioReport {
    pub name: &'static str,
    pub ops: u64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    pub errors: u64,
    pub passed: bool,
}

pub fn log(line: impl AsRef<str>) {
    println!("{}", line.as_ref());
    let _ = io::stdout().flush();
}

pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

pub fn finish(name: &'static str, ops: u64, samples: &mut [f64], errors: u64, passed: bool) -> ScenarioReport {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let report = ScenarioReport {
        name,
        ops,
        p50_us: percentile(samples, 0.50),
        p95_us: percentile(samples, 0.95),
        p99_us: percentile(samples, 0.99),
        max_us: samples.last().copied().unwrap_or(0.0),
        errors,
        passed,
    };
    log(format!(
        "  latency us   p50 {:>10.1}   p95 {:>10.1}   p99 {:>10.1}   max {:>10.1}   n {}",
        report.p50_us, report.p95_us, report.p99_us, report.max_us, samples.len()
    ));
    log(format!(
        "  errors {errors}   status {}",
        if passed { "PASS" } else { "FAIL" }
    ));
    report
}

pub fn print_scorecard(rows: &[ScenarioReport]) {
    log("");
    log("================ MASTER STRESS-TEST SCORECARD ================");
    log(format!(
        "{:<44} {:>14} {:>22} {:>12} {:>8}",
        "Scenario", "Ops/Cycles", "Latency p50/p99 us", "Errors", "Status"
    ));
    log("-".repeat(106));
    for row in rows {
        log(format!(
            "{:<44} {:>14} {:>10.1} / {:>8.1} {:>12} {:>8}",
            row.name,
            row.ops,
            row.p50_us,
            row.p99_us,
            row.errors,
            if row.passed { "PASS" } else { "FAIL" }
        ));
    }
    let passed = rows.iter().filter(|row| row.passed).count();
    log("-".repeat(106));
    log(format!(
        "scenarios {}/{} passed",
        passed,
        rows.len()
    ));
}
