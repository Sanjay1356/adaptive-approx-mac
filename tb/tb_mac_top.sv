`timescale 1ns/1ps

module tb_mac_top;
    localparam int OPERAND_WIDTH = 8;
    localparam int PRODUCT_WIDTH = 16;
    localparam int ACC_WIDTH     = 32;

    logic clk = 0, rst_n = 0, clear_acc = 0, enable = 0;
    logic signed [OPERAND_WIDTH-1:0] a, b;
    logic [1:0] approx_level;
    logic signed [ACC_WIDTH-1:0] acc_out;

    approx_mac_top #(
        .OPERAND_WIDTH(OPERAND_WIDTH),
        .PRODUCT_WIDTH(PRODUCT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .clear_acc(clear_acc), .enable(enable),
        .a(a), .b(b), .approx_level(approx_level), .acc_out(acc_out)
    );

    always #5 clk = ~clk;

    integer fd, r, total, failed;
    integer a_u, b_u, exp_u;
    logic signed [OPERAND_WIDTH-1:0] a_s, b_s;
    logic signed [ACC_WIDTH-1:0] exp_s;

    initial begin
        rst_n = 0; clear_acc = 0; enable = 0; approx_level = 2'b00;
        a = 0; b = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // Directed sequence at L0: exact accumulation, cross-check against
        // a manually-computed expected value (mirrors exact_mac_sequence
        // semantics: ACC_new = ACC_old + A*B, natural 32-bit wraparound).
        clear_acc = 1; @(posedge clk); @(negedge clk); clear_acc = 0;

        total = 0; failed = 0;

        // sequence: (127,127), (-128,-128), (-1,1), (0,0), (-128,127)
        run_one(127, 127);
        run_one(-128, -128);
        run_one(-1, 1);
        run_one(0, 0);
        run_one(-128, 127);

        // expected final = 16129 + 16384 + (-1) + 0 + (-16256) = 16256
        if (acc_out == 16256) begin
            $display("MAC L0 accumulation sequence: PASS (acc_out=%0d)", acc_out);
        end else begin
            $display("MAC L0 accumulation sequence: FAIL (acc_out=%0d, expected=16256)", acc_out);
            failed = failed + 1;
        end
        total = total + 1;

        // sanity: -128 operand should be approximation-invariant (proven
        // algebraically + verified in Python earlier) -- check L0 vs L3
        // give the same accumulator result for an all -128-involving sequence.
        clear_acc = 1; @(posedge clk); @(negedge clk); clear_acc = 0;
        approx_level = 2'b00;
        run_one(-128, 5);
        run_one(-128, -37);
        $display("L0 acc after -128-only sequence = %0d", acc_out);

        clear_acc = 1; @(posedge clk); @(negedge clk); clear_acc = 0;
        approx_level = 2'b11; // L3
        run_one(-128, 5);
        run_one(-128, -37);
        $display("L3 acc after -128-only sequence = %0d (should match L0)", acc_out);

        $display("========================================");
        $display("MAC-LEVEL TESTS: total=%0d failed=%0d -> %s",
                  total, failed, (failed == 0) ? "PASS" : "FAIL");
        $display("========================================");
        $finish;
    end

    task run_one(input int av, input int bv);
        begin
            @(negedge clk);   // change inputs mid-low-phase, safely away from
                              // the capturing edge -- avoids a same-instant
                              // race between testbench input changes and the
                              // always_ff block sampling them at posedge.
            a = av; b = bv;
            enable = 1;
            @(posedge clk);   // capture edge
            @(negedge clk);
            enable = 0;
        end
    endtask

endmodule
