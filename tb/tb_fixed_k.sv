`timescale 1ns/1ps

module tb_fixed_k;
    localparam int OPERAND_WIDTH = 8;
    localparam int PRODUCT_WIDTH = 16;

    genvar gk;
    generate
        for (gk = 0; gk < 4; gk = gk + 1) begin : LVL
            localparam int K = (gk == 0) ? 0 : (gk == 1) ? 2 : (gk == 2) ? 4 : 6;
            logic signed [OPERAND_WIDTH-1:0] a, b;
            logic [1:0] dummy_level;
            logic signed [PRODUCT_WIDTH-1:0] product;

            approx_multiplier #(
                .OPERAND_WIDTH(OPERAND_WIDTH),
                .PRODUCT_WIDTH(PRODUCT_WIDTH),
                .FIXED_K(K)
            ) dut (
                .a(a), .b(b), .approx_level(dummy_level), .product(product)
            );
        end
    endgenerate

    integer fd, r, total, failed;
    integer a_u, b_u, lvl, expected_u;
    logic signed [OPERAND_WIDTH-1:0] a_s, b_s;
    logic signed [PRODUCT_WIDTH-1:0] expected_s;

    initial begin
        fd = $fopen("vectors/golden_vectors.txt", "r");
        total = 0; failed = 0;
        while (!$feof(fd)) begin
            r = $fscanf(fd, "%h %h %h %h\n", a_u, b_u, lvl, expected_u);
            if (r == 4) begin
                a_s = OPERAND_WIDTH'(a_u);
                b_s = OPERAND_WIDTH'(b_u);
                expected_s = PRODUCT_WIDTH'(expected_u);
                total = total + 1;

                LVL[0].a = a_s; LVL[0].b = b_s;
                LVL[1].a = a_s; LVL[1].b = b_s;
                LVL[2].a = a_s; LVL[2].b = b_s;
                LVL[3].a = a_s; LVL[3].b = b_s;
                #1;

                case (lvl)
                    0: if (LVL[0].product !== expected_s) failed = failed + 1;
                    1: if (LVL[1].product !== expected_s) failed = failed + 1;
                    2: if (LVL[2].product !== expected_s) failed = failed + 1;
                    3: if (LVL[3].product !== expected_s) failed = failed + 1;
                endcase
            end
        end
        $fclose(fd);
        $display("FIXED-K INSTANCES: total=%0d failed=%0d -> %s",
                  total, failed, (failed == 0) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
