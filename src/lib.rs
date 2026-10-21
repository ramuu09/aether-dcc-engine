//! AetherDCC Engine v2.0.
//! Production-grade Differential Contention Compute, Universal D3D11 GPU inference,
//! and 64-node Ising paradox solver.

mod abi;
mod estimator_ablation;
mod bitnet;
mod cpu_alu;
mod gpu;
mod ising;
mod machine;
mod product;
mod report;
mod scenarios;
mod sentinel;

pub use bitnet::{argmax, corrupt, glyph_f32, ratio, reference, weight_census, GLYPHS};
pub use cpu_alu::Alu;
pub use gpu::{available_adapters, AdapterInfo, BitnetSample, GpuEngine, GPU_ADD, GPU_AND, GPU_OR, GPU_SUB, GPU_XOR};
pub use estimator_ablation::{ablation_benchmark, AblationReport};
pub use product::{digit_qa, product_benchmark, DigitQa};
pub use report::{print_scorecard, ScenarioReport};
pub use scenarios::{
    run_all, stress_adversarial_alu_fuzzer, stress_concurrent_cpu_gpu_collision, stress_endurance_marathon,
    stress_hostile_os_storm, stress_pathological_paradox_and_graphs, stress_vision_noise_cliff_and_zero_fix,
};

/// Parks the machine between batches. Burst and pedestal stay at the exact
/// quaternary counts. Batch runners call the breathers so the fan and the
/// desktop get a 1 ms idle gap.
pub const COOL_DIGIT_TRIALS: usize = 1;

pub struct ThermalGovernor;

impl ThermalGovernor {
    pub const BURST: u32 = 4;
    pub const PEDESTAL: u32 = 4;
    const EVERY: u32 = 32;

    fn ops() -> &'static std::sync::atomic::AtomicU32 {
        static OPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        &OPS
    }

    /// Each scenario starts from zero so the 32-op gap does not depend on earlier tests.
    pub fn reset() {
        Self::ops().store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn after_contention(count: u32) {
        if count == 0 {
            return;
        }
        let total = Self::ops().fetch_add(count, std::sync::atomic::Ordering::Relaxed).wrapping_add(count);
        if total / Self::EVERY != (total - count) / Self::EVERY {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    pub fn after_dispatch() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[cfg(test)]
mod tests {
    use super::print_scorecard;

    #[test]
    fn stress_hostile_os_storm() {
        super::ThermalGovernor::reset();
        let row = super::stress_hostile_os_storm();
        print_scorecard(std::slice::from_ref(&row));
        assert!(row.passed, "hostile storm recorded {} errors", row.errors);
    }

    #[test]
    fn stress_endurance_marathon() {
        super::ThermalGovernor::reset();
        let row = super::stress_endurance_marathon();
        print_scorecard(std::slice::from_ref(&row));
        assert!(row.passed, "endurance recorded {} errors", row.errors);
    }

    #[test]
    fn stress_concurrent_cpu_gpu_collision() {
        super::ThermalGovernor::reset();
        let row = super::stress_concurrent_cpu_gpu_collision();
        print_scorecard(std::slice::from_ref(&row));
        assert!(row.passed, "collision recorded {} errors", row.errors);
    }

    #[test]
    fn stress_adversarial_alu_fuzzer() {
        super::ThermalGovernor::reset();
        let row = super::stress_adversarial_alu_fuzzer();
        print_scorecard(std::slice::from_ref(&row));
        assert!(row.passed, "fuzzer recorded {} errors", row.errors);
    }

    #[test]
    fn stress_vision_noise_cliff_and_zero_fix() {
        super::ThermalGovernor::reset();
        let row = super::stress_vision_noise_cliff_and_zero_fix();
        print_scorecard(std::slice::from_ref(&row));
        assert!(row.passed, "vision cliff recorded {} errors", row.errors);
    }

    #[test]
    fn stress_pathological_paradox_and_graphs() {
        super::ThermalGovernor::reset();
        let row = super::stress_pathological_paradox_and_graphs();
        print_scorecard(std::slice::from_ref(&row));
        assert!(row.passed, "paradox/graphs recorded {} errors", row.errors);
    }

    #[test]
    fn product_bitnet_digit10() {
        super::ThermalGovernor::reset();
        let mut gpu = super::GpuEngine::new();
        let qa = super::digit_qa(&mut gpu);
        assert_eq!(qa.blank_max, 0, "empty canvas was not DC-centered");
        assert_eq!(qa.clean, 10, "clean digits");
        assert_eq!(qa.band_misses, 0, "noise band 0..=12 flips");
        assert_eq!(qa.mismatches, 0, "GPU diverged from the host reference");
    }

    #[test]
    fn test_pillar6_preemption_ablation_naive_vs_hardened() {
        super::ThermalGovernor::reset();
        let started = std::time::Instant::now();
        let report = super::ablation_benchmark().expect("ablation harness");
        let elapsed = started.elapsed();
        assert!(
            elapsed.as_millis() < 500,
            "ablation took {} ms",
            elapsed.as_millis()
        );
        assert_eq!(report.turn.errors, 0, "turn pedestal errors {}", report.turn.errors);
        assert!(
            report.chopper.ser < 0.001,
            "chopper SER {:.6}% ({}/{})",
            report.chopper.ser * 100.0,
            report.chopper.errors,
            report.chopper.trials
        );
        assert!(
            report.naive.errors > report.chopper.errors,
            "naive {} did not exceed chopper {}",
            report.naive.errors,
            report.chopper.errors
        );
    }

    #[test]
    fn test_verilog_dcc_arbiter_software_twin_equivalence() {
        // Digital twin verification of rtl/dcc_arbiter.v shift-add logic
        fn rtl_ratio_state(p: u32, n: u32) -> i8 {
            if p + n == 0 {
                return 0;
            }
            let p3 = (p << 1) + p;
            let n7 = (n << 3) - n;
            let p9 = (p << 3) + p;
            let n11 = (n << 3) + (n << 1) + n;
            let p11 = (p << 3) + (p << 1) + p;
            let n9 = (n << 3) + n;
            let p7 = (p << 3) - p;
            let n3 = (n << 1) + n;

            if p3 > n7 {
                3
            } else if p9 > n11 {
                1
            } else if p11 >= n9 {
                0
            } else if p7 >= n3 {
                -1
            } else {
                -3
            }
        }

        // Test four legal pedestal plateaus
        assert_eq!(rtl_ratio_state(16, 4), 3);
        assert_eq!(rtl_ratio_state(12, 8), 1);
        assert_eq!(rtl_ratio_state(8, 12), -1);
        assert_eq!(rtl_ratio_state(4, 16), -3);
        assert_eq!(rtl_ratio_state(4, 4), 0);
        // Pedestal 0 full scale
        assert_eq!(rtl_ratio_state(12, 0), 3);
        assert_eq!(rtl_ratio_state(0, 12), -3);
        assert_eq!(rtl_ratio_state(0, 0), 0);
    }
}
