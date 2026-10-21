`timescale 1ns / 1ps

module dcc_arbiter_tb;
    reg clk;
    reg rst_n;
    reg start;
    reg [5:0] drive;
    wire signed [3:0] y4, y0;
    wire sum4, car4, xor4, done4;
    wire sum0, car0, xor0, done0;
    wire [31:0] p4, n4, p0, n0;
    integer errors;

    dcc_arbiter #(.SYNAPSES(3), .PEDESTAL(4), .BURST(4)) dut4 (
        .clk(clk), .rst_n(rst_n), .start(start), .drive(drive),
        .y_state(y4), .sum_bit(sum4), .carry_bit(car4), .xor_bit(xor4),
        .done(done4), .t_pos(p4), .t_neg(n4)
    );

    dcc_arbiter #(.SYNAPSES(3), .PEDESTAL(0), .BURST(4)) dut0 (
        .clk(clk), .rst_n(rst_n), .start(start), .drive(drive),
        .y_state(y0), .sum_bit(sum0), .carry_bit(car0), .xor_bit(xor0),
        .done(done0), .t_pos(p0), .t_neg(n0)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    task run_case;
        input [5:0] drv;
        input integer which;
        input integer exp_p;
        input integer exp_n;
        input integer exp_y;
        input exp_sum;
        input exp_car;
        input exp_xor;
        reg signed [3:0] got_y;
        reg got_sum, got_car, got_xor, got_done;
        integer got_p, got_n;
        begin
            drive = drv;
            start = 1'b0;
            @(posedge clk);
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;
            // Four race clocks. done rises on the fourth.
            repeat (4) @(posedge clk);
            #1;
            if (which == 4) begin
                got_y = y4; got_sum = sum4; got_car = car4; got_xor = xor4;
                got_done = done4; got_p = p4; got_n = n4;
            end else begin
                got_y = y0; got_sum = sum0; got_car = car0; got_xor = xor0;
                got_done = done0; got_p = p0; got_n = n0;
            end
            if (!got_done || got_p !== exp_p || got_n !== exp_n ||
                got_y !== exp_y || got_sum !== exp_sum ||
                got_car !== exp_car || got_xor !== exp_xor) begin
                $display("FAIL drv=%b ped=%0d got %0d/%0d y=%0d sum=%b car=%b xor=%b done=%b expected %0d/%0d y=%0d",
                    drv, which, got_p, got_n, got_y, got_sum, got_car, got_xor, got_done,
                    exp_p, exp_n, exp_y);
                errors = errors + 1;
            end else begin
                $display("PASS drv=%b ped=%0d %0d/%0d y=%0d", drv, which, got_p, got_n, got_y);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst_n = 0;
        start = 0;
        drive = 6'b000000;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // drive[1:0] is synapse 0. 01=+1, 10=-1.
        run_case(6'b010101, 4, 16, 4,  3, 1, 1, 1);
        run_case(6'b100101, 4, 12, 8,  1, 0, 1, 0);
        run_case(6'b101001, 4,  8, 12, -1, 1, 0, 0);
        run_case(6'b101010, 4,  4, 16, -3, 0, 0, 1);
        run_case(6'b000000, 4,  4,  4,  0, 0, 0, 0);
        run_case(6'b001001, 0,  4,  4,  0, 0, 0, 0);
        run_case(6'b010101, 0, 12,  0,  3, 1, 1, 1);
        run_case(6'b000000, 0,  0,  0,  0, 0, 0, 0);

        if (errors == 0) begin
            $display("ALL PASS");
            $finish;
        end else begin
            $display("%0d FAILURES", errors);
            $finish(1);
        end
    end
endmodule
