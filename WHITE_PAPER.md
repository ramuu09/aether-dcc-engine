# Differential Contention Compute on Commodity x86 and Intel UHD Silicon

**AetherDCC Studio v1.0.** Empirical record from `cargo run --release -- --product-benchmark` on an Intel i5-10210U whose DXGI adapter enumerates as `Intel(R) UHD Graphics`, vendor `0x8086`, device `0x9B41`, feature level 11.1. Date of the runs: 2026-10-18. Section 4 and Section 6.0 are the cool process this command prints now: workers parked between jobs, no burn threads, and a deterministic stall in the ablation. Section 6.1 through Section 6.8, and Section 4.5, are the retired long hostile campaign from the same date. Those tables are kept. They are not the cells the current command prints.

## Abstract

A binary gate records one threshold crossing. Under a full CPU load that crossing moves with temperature, clock, and the other cores on the ring bus, and the program has no second measurement that would cancel the motion. Differential Contention Compute (DCC) does not threshold a single interval. It races two banks of `lock xadd` on one 4096-byte page and keeps only the normalized difference

\[
Y = \frac{t_{\mathrm{pos}} - t_{\mathrm{neg}}}{t_{\mathrm{pos}} + t_{\mathrm{neg}}}.
\]

A scale factor shared by both banks cancels. A stall added to only one bank stays in the ratio and drives that trial toward \(\pm 1\). Section 4 measures the stall. With a pedestal of 4 and a burst of 4, three synapses land on four plateaus, \(Y \in \{-0.600,\ -0.200,\ +0.200,\ +0.600\}\), which the ALU reads as the signed magnitudes \(\{-3,\ -1,\ +1,\ +3\}\). The same ratio, with the pedestal removed, is the full-scale logic value \(\pm 1\) and the paradox value \(0\).

This paper records that construction on the i5-10210U and its integrated GPU, device `0x9B41`. A single `cs_5_0` dispatch evaluates a 64→16→10 ternary network whose zero weights never issue `InterlockedAdd`. An empty canvas scores +0.000 on every digit class. GhostVault runs a 128-bit XOR-then-ADD cipher in the same compute shader. Odd liar rings that have no Boolean fixed point settle at \(Y = +0.000\) in one ALU pass. A 64-node Ising relaxation on the same GPU meets or beats a binary greedy cut on energy. The cool product run in Section 6.0 kept bit errors at 0. The retired campaign in Section 6 compared 1,014,247 fuzzer bits at 0.000000% BER while eight unpinned hostile threads held the CPU at 100.0% busy. That campaign is no longer what the command runs.

## 1. Introduction: the binary wall and bus-contention physics

A synchronous Boolean machine treats time as a clock edge and treats contention as noise to be closed out by timing margins. On a laptop SoC the margin is not a constant. The ring bus, the L2 spatial prefetcher, and the memory controller all stretch a `lock xadd` by an amount that depends on who else is issuing locks. A threshold placed on one bank therefore moves when the machine gets hot or busy. That is the binary wall: one operand, one comparison, and no common mode to reject.

Contention is still a physical observable. Two atomic counters that share a page, a burst length, and a turn barrier see the same stretch. Their difference, divided by their sum, is a dimensionless field. DCC uses that field as the datum of an ALU, of a ternary synapse, and of a paraconsistent isolator. The implementation in this repository is native Rust with raw FFI to `kernel32`, `d3d11`, `dxgi`, and `d3dcompiler_47`. There is no third-party crate.

The digit experiment is a geometric block code on 64 pixels. It is not a claim about handwritten MNIST.

## 2. Mathematical formulation

### 2.1 Common-mode rejection

Let a turn commit integer counts \(t_{\mathrm{pos}}\) and \(t_{\mathrm{neg}}\) into two cache lines. The reported observable is

\[
Y =
\begin{cases}
0 & \text{if } t_{\mathrm{pos}} + t_{\mathrm{neg}} = 0, \\[4pt]
\dfrac{t_{\mathrm{pos}} - t_{\mathrm{neg}}}{t_{\mathrm{pos}} + t_{\mathrm{neg}}} & \text{otherwise.}
\end{cases}
\]

The zero-over-zero case is defined. The implementation returns `0.0` and does not produce a NaN. On the release run the defined ratio printed as `+0.000`, finite.

### 2.2 Cancellation of thermal and frequency drift

Suppose temperature \(T\) and clock frequency \(f\) scale both banks by the same positive factor \(\alpha(T, f)\). The measured counts are \(\alpha t_{\mathrm{pos}}\) and \(\alpha t_{\mathrm{neg}}\). Substitute:

\[
Y_{\mathrm{meas}}
  = \frac{\alpha t_{\mathrm{pos}} - \alpha t_{\mathrm{neg}}}{\alpha t_{\mathrm{pos}} + \alpha t_{\mathrm{neg}}}
  = \frac{\alpha\,(t_{\mathrm{pos}} - t_{\mathrm{neg}})}{\alpha\,(t_{\mathrm{pos}} + t_{\mathrm{neg}})}
  = \frac{t_{\mathrm{pos}} - t_{\mathrm{neg}}}{t_{\mathrm{pos}} + t_{\mathrm{neg}}}
  = Y,
\]

provided \(\alpha \neq 0\) and the denominator is nonzero. The factor sits in the numerator and the denominator and divides out. A drift that hits only one bank does not cancel. The hardware section is the construction that keeps the two banks on the same page, the same burst, and the same turn, so the drift they see is common. Section 4 puts a scheduler stall on one rail and records three estimators of the same symbol.

### 2.3 Quaternary plateaus

Three synapses and a pedestal \(P = 4\), each active synapse adding a burst \(B = 4\) to one bank, give

\[
t_{\mathrm{pos}} = P + n_{+} B, \qquad t_{\mathrm{neg}} = P + (3 - n_{+}) B.
\]

The sum is fixed:

\[
t_{\mathrm{pos}} + t_{\mathrm{neg}} = 2P + 3B = 20.
\]

The signed excess is \((2n_{+} - 3)B\), so the four legal occupancies are

| \(n_{+}\) | counts \(t_{\mathrm{pos}}/t_{\mathrm{neg}}\) | excess | \(Y\) | magnitude |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 16/4 | +12 | +0.600 | +3 |
| 2 | 12/8 | +4 | +0.200 | +1 |
| 1 | 8/12 | −4 | −0.200 | −1 |
| 0 | 4/16 | −12 | −0.600 | −3 |

The ALU reads the magnitude off the excess in units of the burst. The sum bit of a full adder is 1 exactly on the magnitudes \(+3\) and \(-1\). The sign bit follows a positive excess. The XOR bit follows an excess whose absolute value exceeds one burst. AND, OR, XOR, and NOT are one contention job per byte. ADD and SUB are bit-serial over 16 bits. ADD starts with carry 0. SUB starts with carry 1 and inverts the subtrahend. The native carry is the unsigned condition \(a \ge b\). Compare uses the flags of SUB: \(A = B\) when the zero flag is set, \(A > B\) when carry is set and the zero flag is clear, \(A < B\) when carry is clear.

With the pedestal removed, three synapses of \(+1\) produce counts 12/0 and \(Y = +1\); three of \(-1\) produce 0/12 and \(Y = -1\); a cancelling pair produces 4/4 and \(Y = 0\). The release endurance log recorded those full-scale counts as `12/0, 0/12, 4/4`.

### 2.4 Paradox value

A circular liar ring \(X_1 = \neg X_2,\ X_2 = \neg X_3,\ X_3 = \neg X_1\) has no Boolean assignment. The Boolean recurrence \(x_i \leftarrow \neg x_{(i+1) \bmod n}\) on an odd cycle is a period-2 orbit. DCC encodes the three drives as \(\{+1,\ -1,\ 0\}\) with the pedestal off. The opposing bursts land 4 on each bank, the idle drive adds nothing, and

\[
Y = \frac{4 - 4}{4 + 4} = 0.
\]

That zero is the Kleene and Priest value BOTH / PARADOX / UNDECIDED: the sentence is not forced to a classical bit. A disconnected node, driven by \(\{0, 0, 0\}\) with the pedestal applied to both banks, also measures 4/4 and \(Y = 0\), which keeps an unforced spin from becoming a NaN.

## 3. Hardware realization

### 3.1 Page layout and the L2 spatial prefetcher

The ALU allocates one page with `VirtualAlloc` (`MEM_COMMIT | MEM_RESERVE`, `PAGE_READWRITE`) and frees it with `MEM_RELEASE`. The page is 4096 bytes and holds eight channels of 256 bytes. Inside a channel the positive `AtomicU32` is at offset 0 and the negative `AtomicU32` is at offset 64, so the two banks occupy different 64-byte lines. The next channel begins 256 bytes later, four cache lines away. A spatial prefetcher that pulls the adjacent line of a sequential stream does not see a contiguous run of counters: the live atomics are isolated by the 256-byte stride, and the whole working set stays inside a single committed page. Compile-time asserts lock `size_of::<Channel>() == 256`, `size_of::<Page>() == 4096`, and `offset_of(neg) == 64`.

The sensor thread and the three synapse threads are unpinned and run at `THREAD_PRIORITY_BELOW_NORMAL`. None of them is placed on logical CPU 0. Between jobs they wait on a condition variable. The sensor publishes the next turn token while holding that mutex, notifies the synapses, and drops the mutex before it touches a bank. Pedestal increments and synapse bursts are `fetch_add` (`AcqRel`, the `lock xadd`) executed only after the mutex is released. The sensor then waits until all three acknowledgements equal the token, and it re-checks that predicate under the mutex after every wakeup. A turn that is still open after 8 seconds returns an error. That bound is a condition-variable timeout, not a pause loop. `ThermalGovernor` inserts its 1 ms idle gap only after the caller has already stored the `Instant` sample for that operation, and each scenario resets the governor so the gap cadence does not leak from the previous test. A process-wide CPU lock still prevents the ALU and the sentinel from running together. Wrong-GPU, shader-compile, and `VirtualAlloc` failures abort. Pinning, `THREAD_PRIORITY_HIGHEST`, and the pause-spin timeout belong to the retired hostile campaign in Section 4.5.

### 3.2 Turn synchronization on the UHD

The GPU path creates a D3D11 device on the adapter whose name contains “Intel” and either “UHD” or “630”, requiring vendor `0x8086` and device `0x9B41`, feature level 11.1 and then 11.0. One process may hold one device. Shaders are `cs_5_0`.

Shared local memory is the turn barrier. `GroupMemoryBarrierWithGroupSync` publishes every `groupshared` store before the next phase, and `InterlockedAdd` is the atomic that accumulates a bank. Barriers sit outside thread-id and neighbor branches. Loop bounds that enclose a barrier are uniform. The vision and Ising shaders are compiled with `D3DCOMPILE_SKIP_OPTIMIZATION` so the compiler does not turn a flat switch into a nested `if` deep enough to trip the FXC nesting limit. The digit shader is compiled with flags 0, and its zero-weight test guards the atomic directly.

The vision shader uses a 65-thread group and `Dispatch(4, 1, 1)` over an 8×8 field, pedestal 256 and max burst 16. The Ising shader uses `numthreads(64, 1, 1)` and one `Dispatch(1, 1, 1)`, degree 8, 16 waves, push 4. GhostVault uses `numthreads(4, 1, 1)` and `Dispatch(16, 1, 1)`: sixteen lanes, eight bits, pedestal 4, burst 4.

### 3.3 Instructions that close on the ALU

The release storm executed three host programs on the quaternary ALU and checked them against a native integer oracle.

- Tax. Base 420 plus addons 135 is gross 555. Gross is above 500, so the discount is 55, the net is 500, five additions of 16 produce tax 80, and `(net + tax) XOR 42` is 622.
- Hybrid fold of the blocks `SENSOR-BLOCK-630`, `DCC-UHD-MASK-16b`, and `BIAS-LANE-QUATER`, threshold 1800. The accumulator was 136, the decision was 2, the stamp was 141, and the inverted stamp was 65394.
- Cipher round-trip on the ALU side contributed 0 extra bit errors.

## 4. Common-Mode vs. Differential-Mode Noise: The One-Sided Preemption Failure & Ablation Study

Section 2.2 divides out a scale \(\alpha(T, f)\) that multiplies both rails inside one measurement window. Thermal throttling, a Turbo Boost step, and a shared bus stretch are that kind of disturbance when they hit \(t_{\mathrm{pos}}\) and \(t_{\mathrm{neg}}\) together. A Windows NT preemption is a different disturbance. The scheduler suspends the sensor in the middle of one rail of one group. The sleep is added to that rail alone. Call it differential-mode impulse noise.

The command now runs 500 trials, 100 of each symbol \(-3, -1, 0, +1, +3\). It spawns no burn threads and pins nobody. Each trial is eight groups. A clean group is the symbol's ideal weights times 1,000. The weights are \((4,1)\), \((3,2)\), \((1,1)\), \((2,3)\), and \((1,4)\) for \(+3, +1, 0, -1, -3\). One synthetic stall of 5,000,000 ticks is added to the negative rail of group 0 on every trial. That stall is the injected \(\Delta_{\mathrm{sched}}\). Decision edges are fixed: \(Y > 0.4\) reads \(+3\), \(Y > 0.1\) reads \(+1\), \(Y \ge -0.1\) reads \(0\), \(Y \ge -0.4\) reads \(-1\), and any lower \(Y\) reads \(-3\). A rejected symbol counts as three bit errors. Symbol error rate is the decision metric. The whole ablation in the timed product process below took 0.966 ms.

### 4.1 A one-sided stall sends the cumulative sum to the rail

Let \(P = \sum_k t_{\mathrm{pos}}^{(k)}\) and \(N = \sum_k t_{\mathrm{neg}}^{(k)}\) be the sums of the unstalled group times, and let a scheduler delay \(\Delta_{\mathrm{sched}}\) land only on the negative rail of group \(k^*\). The naive estimator adds every first-attempt group, including that delay:

\[
Y_{\mathrm{naive}}
  = \frac{P - (N + \Delta_{\mathrm{sched}})}{P + N + \Delta_{\mathrm{sched}}}
  = \frac{(P - N)/\Delta_{\mathrm{sched}} - 1}{(P + N)/\Delta_{\mathrm{sched}} + 1}.
\]

As \(\Delta_{\mathrm{sched}} \to \infty\) the quotients vanish and \(Y_{\mathrm{naive}} \to -1\). The same delay on the positive rail sends \(Y_{\mathrm{naive}} \to +1\). Either limit sits past every quaternary threshold, so one stalled group moves the symbol.

The unit test `neg_stall_drives_the_sum_to_minus_one_and_the_gate_drops_it` plants 5,000,000 ticks on the last of eight otherwise-clean groups. The stall is on the negative rail for \(+3\), \(+1\), \(0\), and \(-1\), and on the positive rail for \(-3\): a further negative stall on a symbol that is already \(-3\) drives \(Y\) toward \(-1\) and stays on that symbol. All five planted stalls decode to the wrong symbol, and each naive \(Y\) has absolute value above 0.9. The limit case \(\Delta = 2^{62}\) on the negative rail of a \(+1\) pattern yields \(Y < -0.999\).

On this injection the stall hits the negative rail of every symbol, including \(-3\). A symbol that is already \(-3\) moves toward \(-1\) and still decodes as \(-3\). The other four symbols also move toward \(-1\) and decode as \(-3\), so they are wrong. That is 400 / 500 symbol errors. Because every trial is stalled, the median \(Y\) of those four symbols leaves the plateau. The \(-3\) median leaves \(-0.600\) as well and remains a correct decode.

A minority of live preemptions can leave the median on the plateau while the wrecked trials are wrong. An earlier sum of every micro-slice duration, in the separate `dcc-time-rail` harness (slice 80,000, 8 groups, hostile CPU 100.0%, median energy scale \(2.755\times\)), recorded 4,266 / 10,000 symbol errors (42.660% SER, 24.266667% BER) and still printed per-symbol median \(Y\) at \(-0.608,\ -0.201,\ -0.001,\ +0.205,\ +0.603\). The zero class in that run was wrong on 1,481 / 2,000 trials (74.050%) with a p10/p90 spread of \(-0.537\) to \(+0.525\). The middle sample stayed at \(-0.001\).

That 42.660% figure belongs to that earlier sum. It is not the naive cell of the cool table below, and the harness does not print it as a constant.

### 4.2 Remedy A — denominator gate and a per-group chopper median

Group \(k\) reports an energy \(E^{(k)} = t_{\mathrm{pos}}^{(k)} + t_{\mathrm{neg}}^{(k)}\). The baseline is the clean group energy of that symbol. A group whose energy exceeds \(\tau_{\mathrm{preempt}} = 3.5 \times E_{\mathrm{baseline}}\) is dropped. There is no resample. Each kept group is reduced to a ratio before the groups are combined:

\[
y^{(k)} = \frac{t_{\mathrm{pos}}^{(k)} - t_{\mathrm{neg}}^{(k)}}{t_{\mathrm{pos}}^{(k)} + t_{\mathrm{neg}}^{(k)}},
\qquad
Y_{\mathrm{chopper}} = \operatorname{median}_k y^{(k)}.
\]

A stall large enough to cross a threshold also inflates the denominator, so the gate removes group 0 from the median. The other seven groups are the clean weights. The naive row still adds every group, so the same stall remains visible there. A symbol whose eight groups all fail the gate would count as three bit errors. In the cool run, one group was dropped and seven were kept.

### 4.3 Remedy B — work-conserving atomic turns

The third estimator is the ALU of Section 3: pedestal 4, burst 4, `lock xadd` on the two banks. The counter advances when a cache line changes owners. A scheduler pause stops the thread and stops the counter with it. Sleep time is absent from both rails. The five symbols are the drives \(\{+1,+1,+1\}\), \(\{+1,+1,-1\}\), \(\{0,0,0\}\), \(\{-1,-1,+1\}\), and \(\{-1,-1,-1\}\), decoded with the same \(Y\) edges. The cool harness probes the ALU once per symbol, five probes in all, and repeats that \(Y\) across the 100 trials of the symbol. The synthetic stall is not an atomic transfer, so it is absent from this row. The workers sit at below-normal priority, unpinned, and wait on a condition variable between probes.

### 4.4 Cool ablation, 500 symbols, deterministic stall

Primary window: `cargo run --release -- --product-benchmark` on 2026-10-19 after the follow-up fixes. The command exited 0. Cargo reported `Finished` in 1.22 s because this binary was rebuilt. A stopwatch around the command, compile included, read 2.209 s. The process printed `product pillars bitnet PASS stress PASS ablation PASS`. The ablation log line was `elapsed 0.966 ms` and `CPU burn threads 0`. The busy field printed `0.0%`. The two `GetSystemTimes` stamps bracket that 0.966 ms window, from the start of the 500 trials through the drop of the ALU, and the system tick did not advance inside it. `cargo test --release -- --test-threads=1` finished 10 passed, 0 failed, in 0.61 s. Cargo reported `Finished` in 5.70 s on that rebuild. One of those tests plants a truncated shader-cache blob and a corrupt DXBC header and checks that both are rejected and recompiled. The ablation pillar passes when the turn estimator has 0 symbol errors, the chopper symbol-error rate is under 0.1%, and the naive error count is larger than the chopper error count. There is no CPU-busy gate.

| Estimator architecture | Median \(Y\) (\(-3, -1, 0, +1, +3\)) | Vulnerable to one-sided preemption? | Symbol error rate (500 trials, injected stall) |
| :--- | :---: | :---: | :---: |
| **1. Naive cumulative rail-sum** | \(-0.997,\ -0.994,\ -0.997,\ -0.990,\ -0.987\) | **YES** (the injected stall enters the rail sum) | **80.000000%** (400 / 500), BER 40.000000% |
| **2. Chopper + denominator gate** | \(-0.600,\ -0.200,\ +0.000,\ +0.200,\ +0.600\) | **NO** (group 0 dropped, then the per-group median) | **0.000000%** (0 / 500), BER 0.000000% |
| **3. Work-conserving turn pedestal** | \(-0.600,\ -0.200,\ +0.000,\ +0.200,\ +0.600\) | **IMMUNE** (counts completed atomic transfers) | **0.000000%** (0 / 500), BER 0.000000% |

The naive medians sit near \(-1\) because the stall is present on every trial. Chopper and turn stay on the plateaus.

### 4.5 Retired live windows, 10,000 symbols at 100.0% CPU busy

These two windows are the hostile campaign retired by the thermal patch. `cargo run --release -- --product-benchmark` does not print them. Eight unpinned threads thrashed a 16 MB buffer. The ablation sensor was pinned to logical CPU 2. The ALU in that campaign pinned its own sensor to logical CPU 0 with `SetThreadAffinityMask`, ran it at `THREAD_PRIORITY_HIGHEST`, and spun until a turn was acknowledged, giving up after 8 seconds. Each group was a Bresenham interleave of 80,000 dependent mul-add iterations, and the chopper of that binary resampled a rejected group. CPU busy was 100.0% in both windows.

Product-benchmark window of that binary. Quiet group energies, in TSC ticks, before the hostile threads started: \(-3\) 1,680,021; \(-1\) 1,599,954; \(0\) 637,913; \(+1\) 1,601,956; \(+3\) 1,598,228. The process printed `product pillars bitnet PASS stress PASS ablation PASS`.

| Estimator architecture | Median \(Y\) (\(-3, -1, 0, +1, +3\)) | Symbol error rate (10,000 trials at 100% load) |
| :--- | :---: | :---: |
| **1. Naive cumulative rail-sum** | \(-0.602,\ -0.201,\ -0.000,\ +0.201,\ +0.602\) | **12.390%** (1,239 / 10,000), BER 8.030000% |
| **2. Chopper + denominator gate** | \(-0.605,\ -0.205,\ +0.002,\ +0.207,\ +0.606\) | **0.000%** (0 / 10,000), BER 0.000000% |
| **3. Work-conserving turn pedestal** | \(-0.600,\ -0.200,\ +0.000,\ +0.200,\ +0.600\) | **0.000%** (0 / 10,000), BER 0.000000% |

Release-test window of that same binary, `cargo test --release -- --test-threads=1`. Quiet energies: \(-3\) 1,677,156; \(-1\) 1,697,117; \(0\) 643,987; \(+1\) 1,678,947; \(+3\) 1,682,612. The suite finished 9 passed, 0 failed, in 235.70 s.

| Estimator architecture | Median \(Y\) (\(-3, -1, 0, +1, +3\)) | Symbol error rate |
| :--- | :---: | :---: |
| **1. Naive cumulative rail-sum** | \(-0.641,\ -0.207,\ -0.000,\ +0.199,\ +0.600\) | **23.040%** (2,304 / 10,000), BER 12.826667% |
| **2. Chopper + denominator gate** | \(-0.599,\ -0.198,\ +0.001,\ +0.204,\ +0.603\) | **0.000%** (0 / 10,000), BER 0.000000% |
| **3. Work-conserving turn pedestal** | \(-0.600,\ -0.200,\ +0.000,\ +0.200,\ +0.600\) | **0.000%** (0 / 10,000), BER 0.000000% |

## 5. Beyond binary

### 5.1 Why the liar ring stops at zero

Boolean iteration on a 3-cycle or a 5-cycle of negations has no fixed point. The release log recorded, for both lengths, `livelock true` and `fixed point exists false` after 100 steps. The DCC pass does not iterate. Node drives \(\{+1,\ -1,\ 0\}\) at pedestal 0 produce counts 4/4. Every node of the 3-ring and every node of the 5-ring printed `Y +0.000`, finite. That is the paraconsistent reading: the ring is admitted as a single undecided value instead of being forced to oscillate. The same run checked a disconnected pedestal node (`counts 4/4`, `Y +0.000`, `NaN false`).

### 5.2 The 1.58-bit network and zero-weight skipping

The network is one dispatch. Layer 1 maps 64 pixels to 16 hidden neurons with weights in \(\{-1, 0, +1\}\). Layer 2 maps those 16 centered activations to 10 class scores, again in \(\{-1, 0, +1\}\).

Layer 1 is a tiled code. Hidden neuron \(t\) carries weight \(+1\) on one 2×2 tile and weight 0 on the other 60 pixels. Of the 1,184 weights in both layers, 960 are zero (81.08%). The shader loads `w1[h * 64 + tid]` and, when `w == 0`, does not call `InterlockedAdd`. That is the bus saving: a structural zero never becomes an atomic on the groupshared banks.

The hidden centered activation is

\[
h_t = (P_t - N_t) + \sum_i W^{(1)}_{t,i}.
\]

For a \(+1\)-only tile the weight sum is 4, so \(h_t = 2 \times (\text{ink count in the tile})\). An all-−1 canvas has \(P = 0\), \(N = 4\), and \(h_t = 0\) on every neuron. Every class score, being a sum of weights times those hidden values, is then 0. The argmax of a ten-way tie is class −1, reported as `BLANK / NO_SIGNAL`. The release log printed `empty canvas scores +0 +0 +0 +0 +0 +0 +0 +0 +0 +0`.

On a clean codeword the true class scores 64 and every rival scores 0. One flipped pixel moves any class score by at most 2, so twelve flips (18.75% of the canvas) move a score by at most 24. The margin from 64 down to 0 still leaves a gap of at least 16, which is why the guaranteed band is not a fitted threshold. Classification uses the integer score. The normalized \(Y\) of a clean true class is \(+1\) (positive mass 64, negative mass 0). A rival sits at \(Y = 0\) because its positive and negative mass are equal.

The four-class vision path is a separate cliff. Its empty-canvas raw counts on this run were 896/640, 1024/512, 832/704, and 768/768. Dividing those counts without removing the DC pedestal prefers the cross class (`+0.250 +0.500 +0.125 +0.000`). After DC centering the same frame is `+0.000 +0.000 +0.000 +0.000`.

### 5.3 Ising relaxation

Sixty-four spins, couplings in \(\{+1,\ -1\}\), degree at most 8. Energy is

\[
E = \sum_{\{i,j\} \in \text{edges}} s_i s_j,
\]

with \(s \in \{-1, 0, +1\}\). It is not the conventional \(-\!J_{ij} s_i s_j\). A zero spin contributes nothing, so two configurations can share an energy and differ in cut size. The acceptance rule is GPU energy at or below the binary greedy energy, together with an exact match against the host reference relaxation.

Wave 0 writes the bias as a burst of 4 on one bank. Later waves push from the current spin. A zero spin does not push. Antiferromagnetic coupling sends a positive spin onto the neighbor’s negative bank. If both banks are empty the previous spin is kept. Otherwise the heavier bank sets the spin, and equal nonzero banks reset the spin to 0 (the paradox value). The five graphs planted for the suite are a 56-edge star forest, a 90-edge dense Möbius ladder on 30 rungs, a 96-edge bipartite spin glass, an 8×8 grid with 112 edges, and 12 disjoint 5-cycles (60 edges).

## 6. Empirical results

Section 6.0 is the cool process this command prints. Sections 6.1 through 6.8 are one earlier `cargo run --release -- --product-benchmark` on the same machine, the long hostile campaign of 2026-10-18. That campaign held the CPU at 100.0% busy with eight burn threads. The current binary does not. Latencies in 6.1 through 6.8 are kept as that record. The harness prints a failure and exits nonzero. It does not relax an assertion to obtain a pass.

### 6.0 Cool product scorecard

`cargo run --release -- --product-benchmark` on 2026-10-19 after the follow-up fixes, exit 0. Cargo reported `Finished` in 1.22 s on the rebuild, and a stopwatch around the command, compile included, read 2.209 s. Workers are below normal priority and unpinned. Between jobs they wait. The sentinel pass spawns no burn threads. A class miss is allowed up to one; bit errors are not. This process classified 8/8 with 0 class misses and 0 bit errors, so the sentinel row shows 0 errors and status PASS. The governor hook runs strictly after the `Instant` latency is recorded and pushed into the sample vector. On the CPU path it sleeps 1 ms only when the per-scenario contention counter crosses a multiple of 32. GPU dispatches sleep 1 ms after `dispatch_us` is captured. Endurance p99 is 64.4 µs and the fuzzer p99 is 1,094.2 µs. The collision CPU-op line printed p50 432.8 µs and p99 878.8 µs. The sentinel p99 of 9,416.6 µs is the classify window: three RDTSC pairs on each of four channels.

| Scenario | Ops/Cycles | Latency p50 / p99 µs | Errors | Status |
| --- | ---: | ---: | ---: | --- |
| Cool sentinel pass | 189 | 302.9 / 9,416.6 | 0 | PASS |
| Endurance, 32 cycles | 32 | 39.3 / 64.4 | 0 | PASS |
| CPU + UHD collision | 8 | 616.0 / 5,582.7 | 0 | PASS |
| Adversarial fuzzer, 32 ops | 49 | 294.3 / 1,094.2 | 0 | PASS |
| Vision noise cliff | 10 | 1,092.9 / 5,177.8 | 0 | PASS |
| Paradox rings and Ising graphs | 221 | 96.5 / 2,502.5 | 0 | PASS |

Six of six passed. Endurance printed the four plateaus, full-scale counts 12/0, 0/12, and 4/4, 0 deadlocks, 0 overflows, and working-set growth 0 bytes. The fuzzer compared 1,367 bits at 0.000000% BER (GPU cipher 0 / 512 bits). Collision reported CPU bit errors 0, GPU cipher 0 / 2,048 bits, and vision 8/8. Its scorecard p99 of 5,582.7 µs mixes those CPU ops with the GPU dispatches. BitNet matched the host (`mismatches 0`), p50 434.4 µs, p99 5,434.3 µs, with one trial per noise level: 10/10 clean glyphs, 100% through 20 flips, and 0% at 24 and 28 flips. The printed SNR breakdown threshold is 24 flipped pixels (37.50%). Vision, also one trial per level, was 100% through 24 flips and 0% at 28. Its printed SNR breakdown threshold is 28 flipped pixels (43.75%). The odd-cycle web tied at \(E = -36\) (greedy cut 48, GPU cut 36). C ABI script results were 622, 0, and 136. Digit-3 inference returned class 3 in 2,179.5 µs. The GhostVault smoke block was 1,415.5 µs. The star solve returned status 0, energy −56, cut 56, in 1,986.7 µs. A separate `cargo test --release -- --test-threads=1` finished 10 passed, 0 failed, in 0.61 s once the Ising bytecode cache under `shader-cache/` was warm. Cargo reported `Finished` in 5.70 s on that rebuild. The process that wrote that cache finished in 40.55 s. With the studio window open and idle, an earlier sample of this message loop accumulated 0.000 ms of processor time across an 800 ms window.

### 6.1 Retired product scorecard

| Scenario | Ops/Cycles | Latency p50 / p99 µs | Errors | Status |
| --- | ---: | ---: | ---: | --- |
| Hostile OS storm | 1,181 | 5.5 / 2,028.7 | 0 | PASS |
| Endurance, 10,000 cycles | 10,000 | 2.4 / 23.3 | 0 | PASS |
| CPU + UHD ring-bus collision | 200 | 206.2 / 972.9 | 0 | PASS |
| Adversarial fuzzer | 50,513 | 5.1 / 22.2 | 0 | PASS |
| Vision noise cliff | 322 | 941.3 / 1,181.8 | 0 | PASS |
| Paradox rings and Ising graphs | 221 | 1.1 / 2,063.2 | 0 | PASS |

Six of six passed on that retired process. A separate `cargo test --release -- --test-threads=1 --nocapture` of that campaign passed these six plus the BitNet digit test: 7 passed, 0 failed, 47.09 s. The table above is that release-report process.

### 6.2 Hostile storm

Eight unpinned threads thrashed one shared 16 MB buffer. `GetSystemTimes` over the sample window reported CPU busy 100.0%. The sentinel classified 8 of 8 scripted regimes with 0 misses (authenticated idle, nominal quiet, thermal surge, surge plus light noise, memory pressure, memory pressure plus light noise, spoofed dual spike, spoofed partial authentication). The ALU mix inside the storm ran 9,598 jobs and 0 bit errors. Tax, cipher, and hybrid matched the oracle (622, 0 extra bit errors, accumulator 136). Mixed latency was p50 5.5 µs, p95 20.1 µs, p99 2,028.7 µs, max 10,709.0 µs, over 1,181 samples. The p99 is the tail of a machine already at full load, and the error count stayed 0.

### 6.3 Endurance

Ten thousand cycles, twenty thousand jobs, five buckets of two thousand. Every bucket reported 0 bit flips, a zero-magnitude \(|Y|\) of 0.000, and the four plateaus −0.600, −0.200, +0.200, +0.600. Deadlocks 0. Overflows 0. Working-set growth on this process was 12,288 bytes, which is inside the 32 MB fail ceiling. Latency p50 2.4 µs, p95 3.0 µs, p99 23.3 µs, max 3,966.5 µs, \(n = 10{,}000\).

### 6.4 Ring-bus collision

One D3D device. The CPU ALU on the main thread and the GPU thread overlapped for 200 iterations with no per-iteration barrier. CPU bit errors 0. GPU cipher bit errors 0 out of 51,200 bits. Vision misses 0 out of 200, and all 8 pattern kinds matched. CPU op latency p50 20.2 µs, p99 36.8 µs. The combined sample latency was p50 206.2 µs, p95 882.3 µs, p99 972.9 µs, max 1,376.0 µs, \(n = 800\).

### 6.5 Fuzzer, 1,014,247 bits

The CPU fuzzer drew 50,000 operations from xorshift64 (seed `0xDCC57E55A11C0001`) plus 13 corners. Progress at 10,000, 25,000, and 50,000 showed 0 bit errors. The GPU then issued 500 cipher dispatches. CPU bit errors 0. GPU bit errors 0 out of 64,000 bits. Compared bits 1,014,247. BER 0.000000%. The accounting is \(50{,}013 \times 19 + 500 \times 16 \times 8\). GPU block latency p50 162.7 µs, p99 354.4 µs. The CPU series was p50 5.1 µs, p95 17.9 µs, p99 22.2 µs, max 2,304.5 µs, \(n = 50{,}013\).

The C ABI smoke measured one cold GhostVault block, including device construction and both the XOR and the ADD dispatch, at 1,720.4 µs, status 0, ciphertext `5D D5 A4 F3 41 56 CE C7 E0 E7 8F D8 DF 9B AB 8D`. The steady-state block time is the fuzzer figure, 162.7 µs p50, not the cold-start figure.

### 6.6 Vision cliff

| Flipped pixels | Fraction | Accuracy | Mean confidence | Trials |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00% | 100% | 25.000 | 40/40 |
| 4 | 6.25% | 100% | 21.000 | 40/40 |
| 8 | 12.50% | 100% | 17.000 | 40/40 |
| 12 | 18.75% | 100% | 13.000 | 40/40 |
| 16 | 25.00% | 80% | 9.500 | 32/40 |
| 20 | 31.25% | 75% | 5.650 | 30/40 |
| 24 | 37.50% | 60% | 3.450 | 24/40 |
| 28 | 43.75% | 40% | 1.850 | 16/40 |

The guaranteed band is 0 through 12 flips. The first level that missed was 16 flips (25.00%). Latency p50 941.3 µs, p95 1,037.5 µs, p99 1,181.8 µs, max 1,961.7 µs, \(n = 322\).

### 6.7 Digit network

The same noise schedule, seed `0xC0FFEE`, one carried xorshift, digit = trial mod 10, 40 trials per level. GPU scores matched the host on every trial (`mismatches 0`).

| Flipped pixels | Fraction | Accuracy | Mean winner score | Trials |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00% | 100% | 64.00 | 40/40 |
| 4 | 6.25% | 100% | 56.00 | 40/40 |
| 8 | 12.50% | 100% | 48.00 | 40/40 |
| 12 | 18.75% | 100% | 40.00 | 40/40 |
| 16 | 25.00% | 100% | 32.00 | 40/40 |
| 20 | 31.25% | 100% | 24.00 | 40/40 |
| 24 | 37.50% | 82.50% | 16.50 | 33/40 |
| 28 | 43.75% | 10.00% | 12.40 | 4/40 |

Clean glyphs were 10/10. The asserted band is through 12 flips. On this draw the first miss was at 24 flips (37.50%). Dispatch latency after the device already existed: p50 164.9 µs, p99 411.1 µs. A cold ABI inference of glyph 3, which also creates the device, returned class 3, status 0, in 2,120.3 µs.

### 6.8 Ising

| Graph | Edges | Greedy cut / \(E\) | GPU cut / \(E\) | GPU µs |
| --- | ---: | ---: | ---: | ---: |
| Star forest | 56 | 56 / −56 | 56 / −56 | 2,063.2 |
| Dense Möbius ladder | 90 | 60 / −30 | 87 / −84 | 892.3 |
| Random spin glass | 96 | 76 / −56 | 96 / −96 | 814.6 |
| Bipartite grid | 112 | 103 / −94 | 112 / −112 | 789.1 |
| Odd-cycle web | 60 | 48 / −36 | 36 / −36 | 847.8 |

GPU energy is at or below the greedy energy on every graph. The odd-cycle web ties at \(E = -36\) with a lower cut because spins that settled at 0 are not cut edges. The C ABI solve of the star forest returned status 0, energy −56, cut 56, in 2,122.6 µs including shader compilation on that call.

## 7. Future outlook

The ratio in Section 2 does not require a `lock` prefix. It requires two accumulators that share a scale. A time-domain ASIC can replace the cache-line banks with a pair of delay lines, or a pair of pulse counters, clocked in picoseconds, and still emit \(Y\). The common-mode proof is unchanged: \(\alpha(T, f)\) cancels whenever it multiplies both rails. The quaternary plateaus become four calibrated separations rather than four atomic occupancies. The ternary network’s zero-skip becomes a wire that is simply absent. The paradox zero remains the balanced rail, the state a Boolean latch cannot hold. A stall that lands on only one rail stays in the ratio and drives that trial toward \(\pm 1\). Section 4 records that failure for a cumulative sum, and the two estimators that keep the symbol. The laptop results in Section 6 are the software measurement of the shared-scale ratio on parts that already ship. They are the reason a picosecond differential fabric is a scaling of this ALU, not a different algebra.

## References in this tree

The executable argument is the source of `aether-dcc-sdk` together with the cool scorecard in Section 6.0 and the ablation in Section 4. Shader text for the digit net and GhostVault is under `src/shaders/`. Vision and Ising HLSL are built in `src/gpu.rs`. The first process compiles them with `d3dcompiler_47` and writes the bytecode under `shader-cache/`. Later processes load those bytes.
