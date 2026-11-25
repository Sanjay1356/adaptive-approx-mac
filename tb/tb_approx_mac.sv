// ============================================================================
// tb_approx_mac.sv
//
// Exhaustive verification: reads vectors/golden_vectors.txt (generated
// directly from the Python golden model, models/pp_truncation.py::
// level_multiply -- NOT re-derived in SystemVerilog) and checks
// approx_multiplier's combinational output against every entry.
//
// File format per line (hex, space-separated):
//     AA BB L PPPP
// where AA/BB are the 8-bit two's-complement encodings of A/B (matching
// Python's a & 0xFF), L is the level (0/1/2/3), and PPPP is the expected
// 16-bit product (two's-complement encoded, product & 0xFFFF).
//
// Runs the runtime-configurable instance (FIXED_K=-1) exhaustively over
// all 65,536 pairs x 4 levels = 262,144 vectors. Reports PASS/FAIL and,
// on failure, the first mismatching input/expected/actual.
// ============================================================================

`timescale 1ns/1ps

module tb_approx_mac;

    localparam int OPERAND_WIDTH = 8;
    localparam int PRODUCT_WIDTH = 16;

    logic signed [OPERAND_WIDTH-1:0] a, b;
    logic [1:0]                      approx_level;
    logic signed [PRODUCT_WIDTH-1:0] product;

    approx_multiplier #(
        .OPERAND_WIDTH(OPERAND_WIDTH),
        .PRODUCT_WIDTH(PRODUCT_WIDTH),
        .FIXED_K(-1)
    ) dut (
        .a(a),
        .b(b),
        .approx_level(approx_level),
        .product(product)
    );

    integer fd, r;
    integer total, passed, failed;
    integer a_u, b_u, lvl, expected_u;
    logic signed [OPERAND_WIDTH-1:0] a_s, b_s;
    logic signed [PRODUCT_WIDTH-1:0] expected_s;

    initial begin
        fd = $fopen("vectors/golden_vectors.txt", "r");
        if (fd == 0) begin
            $display("ERROR: could not open vectors/golden_vectors.txt");
            $finish;
        end

        total = 0; passed = 0; failed = 0;

        while (!$feof(fd)) begin
            r = $fscanf(fd, "%h %h %h %h\n", a_u, b_u, lvl, expected_u);
            if (r == 4) begin
                a_s = OPERAND_WIDTH'(a_u);   // reinterpret 8-bit pattern as signed
                b_s = OPERAND_WIDTH'(b_u);
                expected_s = PRODUCT_WIDTH'(expected_u);

                a = a_s;
                b = b_s;
                approx_level = lvl[1:0];
                #1; // allow combinational settle

                total = total + 1;
                if (product === expected_s) begin
                    passed = passed + 1;
                end else begin
                    failed = failed + 1;
                    if (failed == 1) begin
                        $display("FIRST MISMATCH:");
                        $display("  a=%0d b=%0d level=L%0d", a_s, b_s, lvl);
                        $display("  expected=%0d (0x%h)  actual=%0d (0x%h)",
                                  expected_s, expected_u, product, product);
                    end
                end
            end
        end

        $fclose(fd);

        $display("========================================");
        $display("TOTAL VECTORS : %0d", total);
        $display("PASSED        : %0d", passed);
        $display("FAILED        : %0d", failed);
        if (failed == 0 && total == 262144)
            $display("RESULT: PASS (%0d/%0d, full exhaustive coverage)", passed, total);
        else if (failed == 0)
            $display("RESULT: PASS on %0d vectors, but expected 262144 -- check vector file", total);
        else
            $display("RESULT: FAIL");
        $display("========================================");

        $finish;
    end

endmodule
