`timescale 1ns/1ps
module tb_adaptive_mac_system;
    logic clk = 0, rst_n = 0, clear_acc = 0, enable = 0, layer_valid = 0;
    logic [4:0] layer_id = 0;
    logic signed [7:0] a, b;
    logic signed [31:0] acc_out;
    logic [1:0] approx_level_used;

    adaptive_mac_system dut (
        .clk(clk), .rst_n(rst_n), .layer_id(layer_id), .layer_valid(layer_valid),
        .clear_acc(clear_acc), .enable(enable), .a(a), .b(b),
        .acc_out(acc_out), .approx_level_used(approx_level_used)
    );

    always #5 clk = ~clk;
    integer errors = 0;

    task do_mac(input int av, input int bv);
        begin
            @(negedge clk); a = av; b = bv; enable = 1;
            @(posedge clk); @(negedge clk); enable = 0;
        end
    endtask

    initial begin
        @(negedge clk); @(negedge clk); rst_n = 1; @(negedge clk);

        // Layer 12 -> policy says L3 (K=6). Confirm approx_level_used==L3
        // one cycle after layer_id/layer_valid are set, then run a MAC
        // that exercises real approximation (127x127: exact=16129, L3
        // approx=15888, per project's own exhaustively-verified numbers).
        layer_id = 5'd12; layer_valid = 1;
        @(negedge clk); // one cycle for the registered controller output
        if (approx_level_used !== 2'b11) begin
            errors++; $display("FAIL: layer 12 expected approx_level=L3, got %b", approx_level_used);
        end

        clear_acc = 1; @(posedge clk); @(negedge clk); clear_acc = 0;
        do_mac(127, 127);
        if (acc_out !== 32'sd15888) begin
            errors++; $display("FAIL: layer 12 (L3) MAC expected 15888, got %0d", acc_out);
        end else begin
            $display("OK: layer 12 (L3) 127*127 -> acc_out=%0d (matches approx_multiplier's verified L3 result)", acc_out);
        end

        // Layer 3 -> policy says L0 (exact). Same operands must now give
        // the EXACT product, 16129, through the identical MAC hardware.
        layer_id = 5'd3; layer_valid = 1;
        @(negedge clk);
        if (approx_level_used !== 2'b00) begin
            errors++; $display("FAIL: layer 3 expected approx_level=L0, got %b", approx_level_used);
        end
        clear_acc = 1; @(posedge clk); @(negedge clk); clear_acc = 0;
        do_mac(127, 127);
        if (acc_out !== 32'sd16129) begin
            errors++; $display("FAIL: layer 3 (L0) MAC expected 16129, got %0d", acc_out);
        end else begin
            $display("OK: layer 3 (L0) 127*127 -> acc_out=%0d (exact)", acc_out);
        end

        $display("========================================");
        if (errors == 0) $display("ADAPTIVE MAC SYSTEM INTEGRATION: PASS");
        else $display("ADAPTIVE MAC SYSTEM INTEGRATION: FAIL (%0d errors)", errors);
        $display("========================================");
        $finish;
    end
endmodule
