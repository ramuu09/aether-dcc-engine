# A Native Differential Contention Cell for a Fixed-Size ASIC

AetherDCC hardware note, 2026. This document is the Phase 2 specification that accompanies AetherDCC Studio v1.0. The software measurements it cites are from `aether-dcc-sdk` commit `faaae0b`, recorded in [WHITE_PAPER.md](WHITE_PAPER.md). No DCC-ASIC has been fabricated, taped out, or timed in silicon. Every picosecond figure below is a design budget under a stated gate delay. It is not a measurement.

The synthesizable cell is [rtl/dcc_arbiter.v](rtl/dcc_arbiter.v).

## Abstract

Software differential contention on the i5-10210U decides a quaternary symbol from two cache-line counts,

\[
Y =
\begin{cases}
0 & \text{if } t_{\mathrm{pos}} + t_{\mathrm{neg}} = 0, \\[4pt]
\dfrac{t_{\mathrm{pos}} - t_{\mathrm{neg}}}{t_{\mathrm{pos}} + t_{\mathrm{neg}}} & \text{otherwise.}
\end{cases}
\]

On the cool product run of 2026-10-19 that process kept bit errors at 0. The fuzzer compared 1,367 bits at 0.000000% BER. An earlier hostile campaign, retired by the thermal patch, compared 1,014,247 fuzzer bits at the same reported BER while the CPU was held at 100.0% busy. Those are the logged bit counts. They are not a claim about every future input. The same runs spent microseconds to milliseconds per sample because the race is a pair of `lock xadd` operations on a Windows process. The sentinel classify window alone printed a p99 of 9,416.6 µs.

This note moves that race into a dedicated cell. Two matched rails accumulate a pulse weight. A common positive scale \(\alpha(T, V_{dd})\) cancels in \(Y\). The quaternary code is selected by four comparisons that are algebraically identical to the thresholds \(Y > 0.4\), \(Y > 0.1\), \(Y \ge -0.1\), and \(Y \ge -0.4\), and that use only shifts and adds. A zero weight emits no pulse. The reference RTL is a clocked counter of those pulses. A full-custom delay line can replace the counter. Standard-cell synthesis of the RTL does not create picosecond delays.

A 64-cell array is specified as a fixed-size relaxer for the same 64-node, degree-8 graphs the SDK already runs. Its settling time is a constant set by the cell delay and the graph diameter. That constant is \(O(1)\) only because \(N = 64\) is hard-wired. The array is a local relaxer. One pass is not an exact algorithm for Max-Cut.

## 1. Introduction

### 1.1 What the software already decided

The SDK keeps one sensor and three synapse threads at below-normal priority, unpinned, waiting on a condition variable between jobs. A pedestal of 4 and a burst of 4 place three bipolar synapses on four plateaus:

| \(n_{+}\) | \(t_{\mathrm{pos}}/t_{\mathrm{neg}}\) | \(Y\) | code |
| ---: | ---: | ---: | ---: |
| 3 | 16/4 | +0.600 | +3 |
| 2 | 12/8 | +0.200 | +1 |
| 1 | 8/12 | −0.200 | −1 |
| 0 | 4/16 | −0.600 | −3 |

The sum of the two rails is \(2 \cdot 4 + 3 \cdot 4 = 20\). With the pedestal removed, three \(+1\) drives produce 12/0 and \(Y = +1\), three \(-1\) drives produce 0/12 and \(Y = -1\), and a cancelling pair produces 4/4 and \(Y = 0\). The endurance log recorded the full-scale counts `12/0, 0/12, 4/4`. A 0/0 window is defined as \(Y = 0\), not as NaN.

Section 2.2 of the software paper shows that a scale shared by both banks divides out, and that a stall on one bank does not. The cool ablation put a 5,000,000-tick stall on the negative rail of group 0 in each of 500 trials. The naive sum moved to about \(-1\) and made 400 / 500 symbol errors. The chopper and the work-conserving turn each made 0 / 500. That is the empirical content of common-mode rejection on this host. It is also the warning for silicon: mismatch between the rails is the same kind of fault as that stall.

The latency that remains is the operating system. Collision CPU-op p99 on the cool run was 878.8 µs. Digit-3 inference took 2,179.5 µs. The star Ising solve took 1,986.7 µs. None of those numbers is a gate delay. They are the reason to leave MESI atomics for a cell that has no scheduler inside the race.

### 1.2 The cell this note specifies

One Differential Time-to-Digital Arbiter (DTDA) has two rails, a set of synapse drivers, and a decoder. A driver with weight \(+1\) launches a pulse of \(B\) matched delays onto the positive rail. A driver with weight \(-1\) launches the same pulse onto the negative rail. A driver with weight \(0\) launches nothing. The rails are counted in units of one delay \(\tau\). The decoder emits the quaternary code, and it emits the sum, carry, and XOR bits the software ALU already uses for bit-serial addition.

Two realizations share that interface:

1. The file `rtl/dcc_arbiter.v` counts pulses with a clock. One clock is one bin. The file is Verilog-2001. It contains no `*` operator and no `/` operator. Synthesis maps it to a counter, an adder tree of width `SYNAPSES`, and a comparator.
2. A full-custom rail replaces the clocked increment with a chain of identical inverters, a vernier pair, or a ring-oscillator delay line. The decoder equations do not change. The delay \(\tau(T, V_{dd})\) is then a property of that chain. This note does not deliver that layout.

The name "picosecond" belongs only to realization 2, and only when the chosen library's inverter delay is a picosecond-class number. This repository does not contain a characterized cell library, so it does not assert a number for \(\tau\).

## 2. The native cell

### 2.1 Rails

Let synapse \(s \in \{1,\ldots,S\}\) present a weight \(w_s \in \{-1, 0, +1\}\). Let \(B\) be the burst in delay bins and \(P\) the pedestal in the same bins. The integrated counts are

\[
t_{\mathrm{pos}} = P + B \sum_s \mathbf{1}[w_s = +1],
\qquad
t_{\mathrm{neg}} = P + B \sum_s \mathbf{1}[w_s = -1].
\]

The software channel is the case \(S = 3\), \(P = 4\), \(B = 4\). The indicator form is the one the RTL implements: a zero weight adds to neither rail, and a cancelling pair does not require the third synapse to be bipolar.

A delay-line realization gives the same integers when each bin is one stage and both rails use the same stage. If the stage delay is \(\tau\), the physical arrival span of a rail is \(t \cdot \tau\). The decoder is specified on the integers \(t_{\mathrm{pos}}\) and \(t_{\mathrm{neg}}\), not on the product \(t \cdot \tau\).

### 2.2 Common-mode cancellation

Assume the two rails are matched: same transistor flavor, same drawn length, same load, same local temperature \(T\), and same local supply \(V_{dd}\). Then every bin on both rails equals one and the same \(\tau(T, V_{dd}) > 0\). The physical times are

\[
T_{\mathrm{pos}} = \alpha\, t_{\mathrm{pos}},
\qquad
T_{\mathrm{neg}} = \alpha\, t_{\mathrm{neg}},
\qquad
\alpha = \tau(T, V_{dd}) / \tau_0,
\]

with any fixed reference \(\tau_0 > 0\). Substitute the definition of \(Y\), for a positive denominator:

\begin{align*}
Y_{\mathrm{meas}}
  &= \frac{\alpha t_{\mathrm{pos}} - \alpha t_{\mathrm{neg}}}{\alpha t_{\mathrm{pos}} + \alpha t_{\mathrm{neg}}} \\
  &= \frac{\alpha\,(t_{\mathrm{pos}} - t_{\mathrm{neg}})}{\alpha\,(t_{\mathrm{pos}} + t_{\mathrm{neg}})} \\
  &= \frac{t_{\mathrm{pos}} - t_{\mathrm{neg}}}{t_{\mathrm{pos}} + t_{\mathrm{neg}}} \\
  &= Y.
\end{align*}

The hypothesis is the match, not the existence of a regulator. A supply loop is unnecessary for this identity. A supply loop does not repair a broken match.

The identity is the same statement as Section 2.2 of the software paper, with \(\alpha(T, f)\) rewritten as \(\alpha(T, V_{dd})\). Frequency dropped out because a dedicated rail has no core clock inside the delay. Temperature and supply remain, and they remain only as a common factor.

### 2.3 What a mismatch does

Let the positive rail stretch by an extra factor, \(\alpha_p = \alpha(1+\varepsilon)\) and \(\alpha_n = \alpha\), with \(\varepsilon > -1\) so both scales stay positive. Then

\[
Y_{\mathrm{meas}}
  = \frac{(1+\varepsilon)\, t_{\mathrm{pos}} - t_{\mathrm{neg}}}{(1+\varepsilon)\, t_{\mathrm{pos}} + t_{\mathrm{neg}}}.
\]

The difference from the matched ratio has numerator \(2 \varepsilon\, t_{\mathrm{pos}} t_{\mathrm{neg}}\) and denominator \(\bigl((1+\varepsilon) t_{\mathrm{pos}} + t_{\mathrm{neg}}\bigr)\bigl(t_{\mathrm{pos}} + t_{\mathrm{neg}}\bigr)\):

\[
Y_{\mathrm{meas}} - Y
  = \frac{2 \varepsilon\, t_{\mathrm{pos}} t_{\mathrm{neg}}}
         {\bigl((1+\varepsilon)\, t_{\mathrm{pos}} + t_{\mathrm{neg}}\bigr)
          \bigl(t_{\mathrm{pos}} + t_{\mathrm{neg}}\bigr)}.
\]

On the balanced pedestal counts 4/4, \(Y = 0\) and

\[
Y_{\mathrm{meas}} = \frac{\varepsilon}{2 + \varepsilon}.
\]

The code stays 0 while \(|Y_{\mathrm{meas}}| \le 0.1\). That holds for \(-2/11 \le \varepsilon \le 2/9\), about \(-18.2\%\) to \(+22.2\%\) of extra stretch on one rail. Past that bound the symbol moves, just as the one-sided stall moved the naive estimator.

On the \(+1\) plateau, counts 12/8, the exact stretched ratio is \((4 + 12\varepsilon) / (20 + 12\varepsilon)\). It remains inside \((0.1,\ 0.4]\) for \(-2/10.8 < \varepsilon \le 4/7.2\), that is about \(-18.5\% < \varepsilon \le +55.6\%\). The tight side is the side that shrinks the positive rail. The \(\pm 3\) plateaus sit farther from their inner threshold (\(0.600\) against \(0.4\)) and tolerate a wider \(\varepsilon\).

These bounds are algebra on the design counts. They are not a measured mismatch of a fabricated rail. Layout still has to earn a small \(\varepsilon\): common-centroid stages, equal wire load, and no supply gradient that hits one rail and misses the other. A gradient is differential-mode. Section 4 of the software paper is the experiment that shows differential-mode does not cancel.

### 2.4 Decoding \(Y\) without a multiplier or a divider

Let \(p = t_{\mathrm{pos}}\) and \(n = t_{\mathrm{neg}}\), with \(p + n > 0\). The four edges used by the ablation study are equivalent to integer comparisons:

\begin{align*}
Y > 0.4 = \tfrac{2}{5}
  &\iff 5(p - n) > 2(p + n)
  \iff 3p > 7n, \\
Y > 0.1 = \tfrac{1}{10}
  &\iff 10(p - n) > (p + n)
  \iff 9p > 11n, \\
Y \ge -0.1
  &\iff 10(p - n) \ge -(p + n)
  \iff 11p \ge 9n, \\
Y \ge -0.4
  &\iff 5(p - n) \ge -2(p + n)
  \iff 7p \ge 3n.
\end{align*}

Each coefficient is a shift-add:

\begin{align*}
3p &= (p \ll 1) + p, \\
7n &= (n \ll 3) - n, \\
9p &= (p \ll 3) + p, \\
11n &= (n \ll 3) + (n \ll 1) + n.
\end{align*}

The first true test in the order \(+3\), \(+1\), \(0\), \(-1\) wins. Otherwise the code is \(-3\). If \(p + n = 0\) the code is \(0\).

A common integer scale \(k > 0\) replaces \((p, n)\) by \((kp, kn)\) and multiplies both sides of every comparison by \(k\). The true branch does not change. That is the discrete form of the cancellation proof. The software excess decoder `delta > 2 * BURST` does not have this property: the threshold is a raw count, so a stretch that multiplies both rails can cross it. On the four legal plateaus, and on 12/0, 0/12, and 4/4, the two decoders agree. Between the plateaus they do not. A one-count excess is a software plateau of \(\pm 1\) and a ratio code of \(0\), because \(|Y|\) is below \(0.1\). The RTL therefore emits both readings. `y_state` is the ratio code. `sum_bit`, `carry_bit`, and `xor_bit` follow `decode_counts` in `src/cpu_alu.rs`, so the bit-serial adder does not change its ISA.

Under that ISA the sum bit is 1 on codes \(+3\) and \(-1\), the carry forwarded to the next bit is the predicate \(t_{\mathrm{pos}} > t_{\mathrm{neg}}\), and the XOR bit is 1 when the absolute excess exceeds one burst. Addition walks 16 bits. Subtraction inverts the subtrahend and starts with carry 1. Those conventions stay in software until a second cell implements the walk; this module is one bit plane's race.

### 2.5 Quantization

The cancellation identity applies to the real times before a bin boundary. After the times are integers, \(Y\) is exactly the ratio of those integers, and the comparisons above are exact on those integers. A later \(\pm 1\) bin error is a different question, and the \(+1\) plateau does not survive every such error.

Counts 12/8 give \(Y = 0.200\). Moving one count from the positive rail onto the negative rail produces 11/9 and \(Y = 0.100\), and \(0.100\) is not greater than \(0.1\), so the code becomes \(0\). The gap from this plateau to its inner edge is exactly that one transfer. Adding one bin to a single rail is milder: 13/8 gives \(Y = 5/21 \approx 0.238\) and 12/9 gives \(Y = 1/7 \approx 0.143\), and both remain code \(+1\).

The \(+3\) plateau has one transfer of room. Counts 16/4 give \(Y = 0.600\). Counts 15/5 give \(Y = 0.500\), still code \(+3\). Counts 14/6 give \(Y = 0.400\), which is not greater than \(0.4\), so the code becomes \(+1\). A decoder that must tolerate a full count moving from one rail to the other needs a wider pedestal gap than the software plateaus provide. The RTL does not add that hysteresis. It decodes the counts it has.

### 2.6 Design budget, not a measured delay

Let \(\tau\) be one matched stage. The race window is \((P + S B)\,\tau\) bins of span, which for the default cell is \(16\tau\) on a full rail (pedestal 4 plus three bursts of 4). The decoder is a handful of carry-lookahead depths on a bus of width \(\lceil \log_2 (P + S B + 1) \rceil + 4\). For the default parameters that width is 9 bits.

This repository does not pick a process node and does not cite a measured FO4. The budget that would put one race under a nanosecond is the inequality

\[
(P + S B)\,\tau + t_{\mathrm{decode}} < 1\,\mathrm{ns}.
\]

With \(P + S B = 16\), a stage slower than \(62.5\,\mathrm{ps}\) already spends the nanosecond on the rail alone, before the decoder. A stage of 10 ps would leave the rail at 160 ps. Both sentences are arithmetic. Neither has been timed on a DCC die, because there is no die.

The clocked RTL makes a different and simpler promise. Its race lasts `BURST` clocks after the clock that accepts `start`. At the default burst of 4, `done` rises on the fourth race clock. The wall time is then four periods of whatever clock the implementer constrains. An FPGA clock in the tens or hundreds of megahertz does not become a picosecond rail by being described with this equation.

## 3. Ternary weights in the wire

### 3.1 Three states, one pair of rails

A binary netlist pays a rail and a complement, or a rail and a clock, to carry one bit. This cell carries a balanced ternary symbol on the pair \((t_{\mathrm{pos}}, t_{\mathrm{neg}})\):

| weight | positive rail | negative rail | dynamic transitions on that driver |
| ---: | --- | --- | --- |
| \(+1\) | pulse of \(B\) bins | quiet | \(B\) bins, one rail |
| \(-1\) | quiet | pulse of \(B\) bins | \(B\) bins, one rail |
| \(0\) | quiet | quiet | none |

"Quiet" means the driver does not toggle. It does not mean the transistor leakage is zero. Dynamic energy of a node is the usual \(C V_{dd}^{2}\) times the number of transitions. A structural zero has a zero transition count, so that term is zero for that driver. Subthreshold and gate leakage on the same device remain, and so does any clock that the synthesizer still routes to a shared counter. Power proportionality applies to the synapse drivers that are actually gated. It does not apply to an ungated clock tree.

The SDK already skips the work. Of the 1,184 ternary weights in the 64→16→10 digit net, 960 are zero (81.08%). The shader does not issue `InterlockedAdd` for those zeros. The silicon statement is the same skip at the driver: 224 of 1,184 weight sites toggle, and each toggling site toggles one rail. The cool run's accuracy figures stay what they were: 10/10 clean glyphs, 100% through 20 flipped pixels, 0% at 24 and 28, printed SNR breakdown at 24 flipped pixels. Those accuracies were measured on the UHD shader, not on an ASIC weight array.

### 3.2 Where the zero code also comes from physics

Weight 0 is also the result of a race, not only an input. Equal rails give \(Y = 0\). The decoder returns that code for the balanced pedestal 4/4 and for the cancelling pair 4/4 with the pedestal removed. The blank digit canvas is the software form of the same fact: every class score was 0 and `argmax` returned −1, reported as `BLANK / NO_SIGNAL`. An empty physical rail pair is specified to decode the same way, including the defined 0/0 case.

## 4. Paradox junction and a 64-cell relaxer

### 4.1 Equilibrium is a code, not a float fight

A Boolean odd cycle of negations has no fixed point. The SDK ran the 3-cycle and the 5-cycle for 100 Boolean steps and printed `livelock true` and `fixed point exists false` for both. The contention pass does not iterate. Drives \(\{+1, -1, 0\}\) at pedestal 0 land 4 on each rail:

\[
Y = \frac{4 - 4}{4 + 4} = 0.
\]

Every node of both rings printed `Y +0.000`, finite. That is the value the junction is required to hold. The cell does not oscillate, and it does not tri-state the rails into an undriven node. Both rails are driven by the same burst, the counts match, and the decoder emits code 0. A disconnected node with the pedestal applied and no synapse pulse is the other software check: counts 4/4, \(Y = 0\), `NaN false`. The RTL covers it. With `PEDESTAL = 4` and every drive neutral, both counters stay at 4.

### 4.2 The 64-cell array

The SDK relaxes 64 spins, degree at most 8, couplings in \(\{+1, -1\}\). Its published energy, which the GPU must not exceed, is

\[
E = \sum_{\{i,j\}} s_i s_j,
\qquad s \in \{-1, 0, +1\}.
\]

The conventional Ising Hamiltonian is a different expression,

\[
H = -\sum_{\{i,j\}} J_{ij} s_i s_j.
\]

They describe the same search on an edge when \(J_{ij} = -1\) and the SDK coupling on that edge is the plain product. The white paper states the difference on purpose: a zero spin contributes nothing to \(E\), so two configurations can share an energy and differ in cut. The hardware array uses \(H\) as the Lyapunov function the coupling is built to descend, and it keeps the SDK's \(E\) as the number an off-chip checker compares with the greedy cut. The cool run's odd-cycle web tied at \(E = -36\) (greedy cut 48, GPU cut 36). That tie is a software result. A future array is not credited with it in advance.

Injection follows the software wave, translated onto the two rails. A neighbor spin of \(+1\) through an antiferromagnetic coupling pulses the local negative rail. A ferromagnetic coupling pulses the rail of the same sign. A zero spin pulses neither rail. Equal nonzero rails reset the local spin to 0, which is the paradox code, and that is the same rule the shader uses when the banks match. The array is 64 copies of the cell with nearest-neighbor wires of degree at most 8. It is not a new solver.

### 4.3 Settling time of a fixed array

Let the longest shortest path in the coupling graph be \(D\) hops, and let one hop take one race of duration \(t_{\mathrm{race}}\) from Section 2.6. A synchronous schedule that forwards each hop in one race finishes in \(D\) races. The planted graphs are not all small-diameter: the 8×8 grid has 112 edges and diameter 14, the star forest has diameter 2, and each 5-cycle has diameter 2. The grid term is \(14\, t_{\mathrm{race}}\). Under the unmeasured budget \(t_{\mathrm{race}} < 1\,\mathrm{ns}\), that schedule finishes in under 14 ns, not under 1 ns. A star-shaped instance is the case that fits in two races.

An analog reading, in which every cell slews continuously and the array is one dynamical system, has a different constant: a few local time constants, not a hop count. Calling that constant "under 1 ns" requires the local time constant itself to be a fraction of a nanosecond. Section 2.6 already shows what stage delay that demands. Until a library and a layout exist, the honest specification is

\[
t_{\mathrm{settle}} = c\, D\, (P + S B)\, \tau
\]

for a synchronous hop schedule, with \(c\) a small integer set by how many races a cell needs before its code is stable. For one race per hop, \(c = 1\).

\(N\), \(D\), and the pin list are constants of this array. Substituting constants into \(t_{\mathrm{settle}}\) produces a constant. That is the sense in which the 64-cell machine is \(O(1)\) in the length of a Turing input: the input cannot grow past 64 spins and degree 8. Growing the graph grows \(D\) or it grows a cut of the graph across chips. The construction does not decide the general Max-Cut problem, and it does not place NP in P.

A continuous descent of \(H\) stops at a stationary point. A stationary point of this coupling is a local compromise among the incident pulses, including the code 0 when the rails match. It is not, by itself, a global minimum. The software acceptance test remains the right check: compare \(E\) with a binary greedy energy, and compare the spins with a reference relaxation. The RTL of a single cell does not perform that check.

## 5. Reference RTL

### 5.1 Ports

`rtl/dcc_arbiter.v` is one cell. The defaults are the software channel: `SYNAPSES = 3`, `PEDESTAL = 4`, `BURST = 4`.

| port | direction | meaning |
| --- | --- | --- |
| `clk` | in | one TDC bin in the clocked realization |
| `rst_n` | in | synchronous active-low reset |
| `start` | in | one-cycle pulse, sampled in idle and hold |
| `drive` | in | `2*SYNAPSES` bits, synapse 0 in `drive[1:0]` |
| `y_state` | out | signed code in \(\{+3,+1,0,-1,-3\}\) from the ratio edges |
| `sum_bit` | out | software sum bit |
| `carry_bit` | out | software carry, `t_pos > t_neg` |
| `xor_bit` | out | software XOR bit, absolute excess above one burst |
| `done` | out | result registers are valid |
| `t_pos`, `t_neg` | out | integrated counts, zero-extended to 32 bits |

Drive codes are `01` for \(+1\), `10` for \(-1\), and `00` or `11` for 0. Drive is required to stay constant from the accepted `start` through the clock that raises `done`.

The race lasts `BURST` clocks. Each clock adds the population of \(+1\) drives to the positive counter and the population of \(-1\) drives to the negative counter, after the pedestal has been loaded. The registered counts are therefore \(P + n_{\pm} B\).

### 5.2 Synthesis boundary

A synthesis tool is entitled to map this module to flops and combinational adders. It is not required to keep a chain of buffers, and this file does not describe one. Attributes that ask a tool to keep buffers are tool-specific and are omitted so the RTL stays portable. The picosecond line, if it is built, is a hard macro with the same integers on its tap bus and the same decoder equations in Section 2.4. Replacing the counter with that macro is a new netlist, not a different reading of \(Y\).

Icarus Verilog 12 elaborated `rtl/dcc_arbiter.v` and ran `rtl/dcc_arbiter_tb.v`. All eight rows below passed. That run is a logic simulation, not a silicon measurement. Patterns and expected counts:

| drives | pedestal | counts | \(Y\) code | sum | carry | xor |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| \(+1,+1,+1\) | 4 | 16/4 | +3 | 1 | 1 | 1 |
| \(+1,+1,-1\) | 4 | 12/8 | +1 | 0 | 1 | 0 |
| \(+1,-1,-1\) | 4 | 8/12 | −1 | 1 | 0 | 0 |
| \(-1,-1,-1\) | 4 | 4/16 | −3 | 0 | 0 | 1 |
| \(0,0,0\) | 4 | 4/4 | 0 | 0 | 0 | 0 |
| \(+1,-1,0\) | 0 | 4/4 | 0 | 0 | 0 | 0 |
| \(+1,+1,+1\) | 0 | 12/0 | +3 | 1 | 1 | 1 |
| \(0,0,0\) | 0 | 0/0 | 0 | 0 | 0 | 0 |

The ratio on 12/0 is \(+1\), and the quaternary code is \(+3\) because \(Y = 1\) lies past \(0.4\). That matches both `contention_ratio` and `decode_counts` in the SDK. The 0/0 row is the defined zero.

### 5.3 What a tapeout would still have to add

A tapeout of a DTDA needs a matched rail, a place-and-route constraint that keeps \(\varepsilon\) inside the bound of Section 2.3, a reset and clock (or a delay-line start pulse) that do not themselves stretch one rail, and a scan or tap path that can read `t_pos` and `t_neg`. None of those artifacts is in this repository. The 64-cell coupling fabric of Section 4 is specified and not written as a second RTL file. One cell is the reference that has to be right before an array is worth describing in gates.

## 6. Status of the claims

Measured, on the i5-10210U and the UHD `0x9B41`, and already published in the software paper and the cool log of commit `faaae0b`:

- quaternary plateaus at \(Y \in \{-0.600, -0.200, +0.200, +0.600\}\)
- full-scale counts 12/0, 0/12, 4/4
- paradox rings at \(Y = 0\)
- digit-net zero skip, 960 / 1,184 weights
- fuzzer BER 0.000000% on 1,367 compared bits in the cool run, and on 1,014,247 bits in the retired hostile run
- ablation 400 / 500 naive symbol errors against 0 / 500 for the chopper and the turn, elapsed 0.966 ms, CPU busy 0.0%

Specified here, and not measured on a DCC die:

- the matched-rail identity and the first-order mismatch formula
- the shift-add decoder
- the clocked RTL
- the delay budget \((P + S B)\,\tau + t_{\mathrm{decode}}\)
- the 64-cell coupling rule and its hop-count settling time

The RTL is the piece that can be simulated and synthesized today. The delay line is the piece that needs a process. Until that process exists, a picosecond number in a slide is a value of \(\tau\) that has not been earned.

## References

1. AetherDCC Studio v1.0 software record, `WHITE_PAPER.md`, this repository, commit `faaae0b`.
2. `src/cpu_alu.rs`, `contention_ratio` and `decode_counts`, for the ratio definition and the sum, carry, and XOR map.
3. `rtl/dcc_arbiter.v`, the synthesizable cell specified in Section 5.
