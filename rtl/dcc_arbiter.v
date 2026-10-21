// dcc_arbiter.v — one differential contention cell.
//
// Verilog-2001, synthesizable to a clocked counter plus shift-add
// comparators. There is no multiply operator and no divider.
//
// This module is the digital twin of one AetherDCC channel
// (pedestal 4, burst 4, three synapses). Each active synapse adds
// one count per race cycle to one rail. After BURST cycles the rails
// hold the same integers the software ALU commits with lock xadd:
//
//   t_pos = PEDESTAL + n_plus  * BURST
//   t_neg = PEDESTAL + n_minus * BURST
//   Y     = 0                         if t_pos + t_neg = 0
//         = (t_pos - t_neg) / (t_pos + t_neg)   otherwise
//
// The quotient is never built. The quaternary symbol is the unique
// code in {+3,+1,0,-1,-3} selected by the cross-multiplied edges
// Y > 0.4, Y > 0.1, Y >= -0.1, Y >= -0.4. Those comparisons are
// invariant under a common positive scale of both rails.
//
// sum_bit, carry_bit, and xor_bit follow decode_counts in src/cpu_alu.rs
// so a bit-serial adder can use this cell as the software ISA does.
// carry_bit is the sign of the excess (t_pos > t_neg), which that ISA
// feeds forward as the next carry. sum_bit is 1 on plateaus +3 and -1.
//
// A full-custom delay line can replace the counter. Synthesis of this
// file does not emit picosecond delays. See HARDWARE_PAPER.md.
//
// Drive packing, two bits per synapse, low synapse in the low bits:
//   2'b01 = +1   pulse the positive rail
//   2'b10 = -1   pulse the negative rail
//   2'b00 =  0   drive nothing
//   2'b11 =  0   reserved, treated as neutral
//
// start is sampled only in IDLE and HOLD. A one-cycle pulse starts one
// race. done rises with the registered result and stays high until the
// next accepted start. Drive must be stable from the accepted start
// through the cycle on which done rises.

`timescale 1ns / 1ps

module dcc_arbiter #(
    parameter integer SYNAPSES = 3,
    parameter integer PEDESTAL = 4,
    parameter integer BURST    = 4
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [2*SYNAPSES-1:0]         drive,
    output reg  signed [3:0]             y_state,
    output reg                           sum_bit,
    output reg                           carry_bit,
    output reg                           xor_bit,
    output reg                           done,
    output reg  [31:0]                   t_pos,
    output reg  [31:0]                   t_neg
);

    localparam integer COUNT_MAX = PEDESTAL + (SYNAPSES * BURST);
    localparam integer COUNT_W   = (COUNT_MAX <= 1) ? 1 : $clog2(COUNT_MAX + 1);
    localparam integer ACC_W     = COUNT_W + 4;
    localparam integer DELTA_W   = COUNT_W + 1;
    localparam integer CYC_W     = (BURST <= 1) ? 1 : $clog2(BURST);

    localparam [1:0] S_IDLE = 2'd0;
    localparam [1:0] S_RACE = 2'd1;
    localparam [1:0] S_HOLD = 2'd2;

    reg [1:0]          state;
    reg [CYC_W-1:0]    cyc;
    reg [COUNT_W-1:0]  pos_r;
    reg [COUNT_W-1:0]  neg_r;

    integer s;
    reg [COUNT_W-1:0] add_pos;
    reg [COUNT_W-1:0] add_neg;

    // One count per active synapse. The loop bound is the parameter
    // SYNAPSES, so synthesis unrolls it into an adder tree.
    always @* begin
        add_pos = {COUNT_W{1'b0}};
        add_neg = {COUNT_W{1'b0}};
        for (s = 0; s < SYNAPSES; s = s + 1) begin
            if (drive[2*s +: 2] == 2'b01)
                add_pos = add_pos + {{(COUNT_W-1){1'b0}}, 1'b1};
            else if (drive[2*s +: 2] == 2'b10)
                add_neg = add_neg + {{(COUNT_W-1){1'b0}}, 1'b1};
        end
    end

    // Integer parameters are not vectors. Widen by assignment.
    wire [COUNT_W-1:0] pedestal_w = PEDESTAL;
    wire [COUNT_W-1:0] next_pos = pos_r + add_pos;
    wire [COUNT_W-1:0] next_neg = neg_r + add_neg;

    function signed [3:0] ratio_state;
        input [COUNT_W-1:0] p;
        input [COUNT_W-1:0] n;
        reg [ACC_W-1:0] px;
        reg [ACC_W-1:0] nx;
        reg [ACC_W-1:0] p3;
        reg [ACC_W-1:0] n7;
        reg [ACC_W-1:0] p9;
        reg [ACC_W-1:0] n11;
        reg [ACC_W-1:0] p11;
        reg [ACC_W-1:0] n9;
        reg [ACC_W-1:0] p7;
        reg [ACC_W-1:0] n3;
        begin
            // Defined 0/0. The software ratio returns 0.0 and does not
            // produce a NaN. Both-rail pedestal makes this case the
            // all-quiet window with PEDESTAL == 0.
            if (({1'b0, p} + {1'b0, n}) == 0) begin
                ratio_state = 4'sd0;
            end else begin
                px  = p;
                nx  = n;
                p3  = (px << 1) + px;
                n7  = (nx << 3) - nx;
                p9  = (px << 3) + px;
                n11 = (nx << 3) + (nx << 1) + nx;
                p11 = (px << 3) + (px << 1) + px;
                n9  = (nx << 3) + nx;
                p7  = (px << 3) - px;
                n3  = (nx << 1) + nx;
                // Y > 2/5, Y > 1/10, Y >= -1/10, Y >= -2/5.
                if (p3 > n7)
                    ratio_state = 4'sd3;
                else if (p9 > n11)
                    ratio_state = 4'sd1;
                else if (p11 >= n9)
                    ratio_state = 4'sd0;
                else if (p7 >= n3)
                    ratio_state = -4'sd1;
                else
                    ratio_state = -4'sd3;
            end
        end
    endfunction

    function signed [DELTA_W-1:0] excess;
        input [COUNT_W-1:0] p;
        input [COUNT_W-1:0] n;
        begin
            excess = $signed({1'b0, p}) - $signed({1'b0, n});
        end
    endfunction

    // Software plateau: |excess| above two bursts is a magnitude-3 code.
    // On the four legal pedestal plateaus this matches ratio_state.
    function sum_from_counts;
        input [COUNT_W-1:0] p;
        input [COUNT_W-1:0] n;
        reg signed [DELTA_W-1:0] d;
        reg signed [3:0] plateau;
        begin
            d = excess(p, n);
            if (d > (2 * BURST))
                plateau = 4'sd3;
            else if (d > 0)
                plateau = 4'sd1;
            else if (d < -(2 * BURST))
                plateau = -4'sd3;
            else if (d < 0)
                plateau = -4'sd1;
            else
                plateau = 4'sd0;
            sum_from_counts = (plateau == 4'sd3) || (plateau == -4'sd1);
        end
    endfunction

    function xor_from_counts;
        input [COUNT_W-1:0] p;
        input [COUNT_W-1:0] n;
        reg signed [DELTA_W-1:0] d;
        begin
            d = excess(p, n);
            xor_from_counts = (d > BURST) || (d < -BURST);
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            cyc      <= {CYC_W{1'b0}};
            pos_r    <= {COUNT_W{1'b0}};
            neg_r    <= {COUNT_W{1'b0}};
            y_state  <= 4'sd0;
            sum_bit  <= 1'b0;
            carry_bit <= 1'b0;
            xor_bit  <= 1'b0;
            done     <= 1'b0;
            t_pos    <= 32'd0;
            t_neg    <= 32'd0;
        end else begin
            case (state)
                S_IDLE, S_HOLD: begin
                    if (start) begin
                        pos_r     <= pedestal_w;
                        neg_r     <= pedestal_w;
                        cyc       <= {CYC_W{1'b0}};
                        y_state   <= 4'sd0;
                        sum_bit   <= 1'b0;
                        carry_bit <= 1'b0;
                        xor_bit   <= 1'b0;
                        done      <= 1'b0;
                        t_pos     <= 32'd0;
                        t_neg     <= 32'd0;
                        state     <= S_RACE;
                    end
                end
                S_RACE: begin
                    pos_r <= next_pos;
                    neg_r <= next_neg;
                    if (cyc == (BURST - 1)) begin
                        y_state   <= ratio_state(next_pos, next_neg);
                        sum_bit   <= sum_from_counts(next_pos, next_neg);
                        carry_bit <= (next_pos > next_neg);
                        xor_bit   <= xor_from_counts(next_pos, next_neg);
                        t_pos     <= next_pos;
                        t_neg     <= next_neg;
                        done      <= 1'b1;
                        state     <= S_HOLD;
                    end else begin
                        cyc <= cyc + {{(CYC_W-1){1'b0}}, 1'b1};
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
