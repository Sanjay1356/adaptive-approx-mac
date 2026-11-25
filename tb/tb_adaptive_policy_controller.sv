// ============================================================================
// tb_adaptive_policy_controller.sv
//
// Exhaustively tests adaptive_policy_controller:
//   - all layer IDs 0..19 (valid range) against an INDEPENDENTLY WRITTEN
//     expected-policy array (not copy-pasted from the DUT's internal
//     table, so this catches a transcription bug in either copy)
//   - all invalid layer IDs 20..31 -> must output L0
//   - layer_valid = 0 for every layer_id (including valid ones) -> L0
//   - async reset behavior -> L0 immediately, regardless of inputs
//
// Reports PASS/FAIL and, on failure, the first mismatching case.
// ============================================================================

`timescale 1ns/1ps

module tb_adaptive_policy_controller;

    localparam int NUM_LAYERS = 20;
    localparam logic [1:0] L0 = 2'b00, L1 = 2'b01, L2 = 2'b10, L3 = 2'b11;

    logic       clk = 0;
    logic       rst_n;
    logic [4:0] layer_id;
    logic       layer_valid;
    logic [1:0] approx_level;

    adaptive_policy_controller #(.NUM_LAYERS(NUM_LAYERS)) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .layer_id    (layer_id),
        .layer_valid (layer_valid),
        .approx_level(approx_level)
    );

    always #5 clk = ~clk;

    // Independently-written expected policy (see header note) -- must
    // match POLICY_TABLE inside the DUT, but is typed out separately here.
    logic [1:0] expected_policy [0:NUM_LAYERS-1];

    initial begin
        expected_policy[0]  = L0; // conv1
        expected_policy[1]  = L0; // layer1.0.conv1
        expected_policy[2]  = L0; // layer1.0.conv2
        expected_policy[3]  = L0; // layer1.1.conv1
        expected_policy[4]  = L0; // layer1.1.conv2
        expected_policy[5]  = L0; // layer2.0.conv1
        expected_policy[6]  = L0; // layer2.0.conv2
        expected_policy[7]  = L2; // layer2.0.downsample.0
        expected_policy[8]  = L0; // layer2.1.conv1
        expected_policy[9]  = L0; // layer2.1.conv2
        expected_policy[10] = L0; // layer3.0.conv1
        expected_policy[11] = L0; // layer3.0.conv2
        expected_policy[12] = L3; // layer3.0.downsample.0
        expected_policy[13] = L0; // layer3.1.conv1
        expected_policy[14] = L0; // layer3.1.conv2
        expected_policy[15] = L0; // layer4.0.conv1
        expected_policy[16] = L0; // layer4.0.conv2
        expected_policy[17] = L1; // layer4.0.downsample.0
        expected_policy[18] = L2; // layer4.1.conv1
        expected_policy[19] = L1; // layer4.1.conv2
    end

    int total, passed, failed;

    task check(input logic [4:0] id, input logic valid, input logic [1:0] expected, input string case_name);
        begin
            @(negedge clk);
            layer_id    = id;
            layer_valid = valid;
            @(posedge clk); // DUT registers level_next here
            @(negedge clk); // settle, read registered output safely

            total = total + 1;
            if (approx_level === expected) begin
                passed = passed + 1;
            end else begin
                failed = failed + 1;
                if (failed <= 5) begin
                    $display("MISMATCH [%s]: layer_id=%0d valid=%0b expected=%b actual=%b",
                              case_name, id, valid, expected, approx_level);
                end
            end
        end
    endtask

    integer i;
    initial begin
        rst_n = 0; layer_id = 0; layer_valid = 0;
        total = 0; passed = 0; failed = 0;

        @(negedge clk); @(negedge clk);

        // --- Reset behavior: while rst_n=0, approx_level must be L0
        //     regardless of inputs (async reset, check without waiting
        //     for a clock edge).
        layer_id = 12; layer_valid = 1; // would be L3 if not in reset
        #1;
        total = total + 1;
        if (approx_level === L0) passed = passed + 1;
        else begin
            failed = failed + 1;
            $display("MISMATCH [async reset]: expected=L0 actual=%b", approx_level);
        end

        rst_n = 1;
        @(negedge clk);

        // --- All valid layer IDs 0..19, layer_valid=1: must match the
        //     independently-written expected_policy exactly.
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            check(i[4:0], 1'b1, expected_policy[i], "valid layer");
        end

        // --- All invalid layer IDs 20..31: must be L0 regardless of
        //     layer_valid.
        for (i = NUM_LAYERS; i < 32; i = i + 1) begin
            check(i[4:0], 1'b1, L0, "invalid id, valid=1");
            check(i[4:0], 1'b0, L0, "invalid id, valid=0");
        end

        // --- layer_valid=0 for every VALID layer id too: must be L0,
        //     not the layer's policy entry.
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            check(i[4:0], 1'b0, L0, "valid id, valid=0");
        end

        // --- Specific spot checks called out explicitly in the spec.
        check(5'd7,  1'b1, L2, "spec check: layer 7 -> L2");
        check(5'd12, 1'b1, L3, "spec check: layer 12 -> L3");
        check(5'd17, 1'b1, L1, "spec check: layer 17 -> L1");
        check(5'd18, 1'b1, L2, "spec check: layer 18 -> L2");
        check(5'd19, 1'b1, L1, "spec check: layer 19 -> L1");

        // --- Reset re-asserted mid-operation must immediately force L0.
        layer_id = 12; layer_valid = 1;
        @(posedge clk); @(negedge clk); // approx_level now L3
        if (approx_level !== L3) begin
            $display("SETUP CHECK FAILED: expected L3 before reset-reassert test, got %b", approx_level);
        end
        rst_n = 0;
        #1;
        total = total + 1;
        if (approx_level === L0) passed = passed + 1;
        else begin
            failed = failed + 1;
            $display("MISMATCH [reset reasserted mid-op]: expected=L0 actual=%b", approx_level);
        end
        rst_n = 1;

        $display("========================================");
        $display("ADAPTIVE POLICY CONTROLLER TESTS");
        $display("TOTAL : %0d", total);
        $display("PASSED: %0d", passed);
        $display("FAILED: %0d", failed);
        if (failed == 0)
            $display("RESULT: PASS");
        else
            $display("RESULT: FAIL");
        $display("========================================");
        $finish;
    end

endmodule
