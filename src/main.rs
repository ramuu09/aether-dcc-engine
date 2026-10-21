//! AetherDCC Engine v2.0 entry point.
//! Defaults to interactive Win32 studio. Headless mode supported via CLI flags.

mod studio;

fn print_help() {
    println!("AetherDCC Engine v2.0 — Production-Grade Contention Compute & AI Runtime");
    println!();
    println!("USAGE:");
    println!("    aether-studio.exe [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --benchmark, --product-benchmark   Execute full release benchmark & stress scorecard");
    println!("    --info, --devices                  Enumerate and display available GPU compute adapters");
    println!("    --test-alu                         Verify quaternary CPU ALU arithmetic & logic");
    println!("    --test-gpu                         Run smoke test for Direct3D 11 compute pipeline");
    println!("    --gui                              Launch interactive Win32 studio dashboard (default)");
    println!("    -h, --help                         Print this help menu");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return;
    }

    if args.iter().any(|arg| arg == "--info" || arg == "--devices") {
        println!("==== AetherDCC Engine v2.0 Hardware Diagnostics ====");
        println!("CPU Host: {} physical/logical cores", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        let adapters = aether_dcc::available_adapters();
        println!("DXGI Compute Adapters ({} found):", adapters.len());
        for (i, a) in adapters.iter().enumerate() {
            println!(
                "  [{}] {} (Vendor: 0x{:04X}, Device: 0x{:04X}, VRAM: {} MB) {}",
                i,
                a.name,
                a.vendor_id,
                a.device_id,
                a.vram_mb,
                if a.is_reference_target { "[Reference Target 0x9B41]" } else { "" }
            );
        }
        return;
    }

    if args.iter().any(|arg| arg == "--test-alu") {
        println!("==== Verifying Quaternary CPU ALU ====");
        let mut alu = aether_dcc::Alu::new();
        let (sum, flags) = alu.add_u16(420, 135).expect("ALU ADD");
        println!("ADD 420 + 135 = {} (expected 555)  Flags: Z={} C={}", sum, flags.z, flags.c);
        let (xor_val, _) = alu.xor_u16(sum, 42).expect("ALU XOR");
        println!("XOR {} ^ 42 = {} (expected 513)", sum, xor_val);
        assert_eq!(sum, 555);
        assert_eq!(xor_val, 513);
        println!("ALU Quaternary Verification: PASS");
        return;
    }

    if args.iter().any(|arg| arg == "--test-gpu") {
        println!("==== Initializing Universal Direct3D 11 Compute Pipeline ====");
        let mut gpu = aether_dcc::GpuEngine::new();
        println!("Selected Adapter: {} (Feature Level {})", gpu.info.name, gpu.feature_name());
        let blank = [-1.0f32; 64];
        let sample = gpu.bitnet(&blank);
        println!("BitNet inference on blank canvas dispatch time: {:.1} us", sample.dispatch_us);
        println!("GPU Pipeline Smoke Test: PASS");
        return;
    }

    if args.iter().any(|arg| arg == "--benchmark" || arg == "--product-benchmark") {
        let code = aether_dcc::product_benchmark();
        std::process::exit(code);
    }

    studio::launch();
}
