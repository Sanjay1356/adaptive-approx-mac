// ============================================================================
// adaptive_policy_controller.sv
//
// Converts layer_id -> approx_level by looking up an OFFLINE-DERIVED,
// FIXED policy table (from Python layer-isolation sensitivity analysis on
// ResNet-18, 90% reference-agreement threshold). This module performs NO
// sensitivity estimation of its own -- it is a runtime lookup of a
// decision that was already made offline. See "NAMING" section below for
// why this distinction matters for how the design should be described.
//
// ENCODING (reused verbatim from approx_multiplier.sv -- NOT redefined
// here, to avoid the exact class of encoding-mismatch bug found and fixed
// earlier in this project's DNN evaluation script):
//     2'b00 = L0 (exact)      2'b01 = L1 (K=2)
//     2'b10 = L2 (K=4)        2'b11 = L3 (K=6)
//
// ARCHITECTURE DECISION -- combinational LUT + registered output:
//   The layer -> level decode itself is a simple 20-entry constant lookup
//   (elaborates to a small mux/ROM, not a case-statement chain, so it is
//   trivially replaceable by a programmable register file or small policy
//   RAM later -- same read interface, see NUM_LAYERS/POLICY_TABLE below).
//   The OUTPUT is registered (clk/rst_n given in the required interface
//   make this the natural choice) for two reasons:
//     1. Isolates this decode's combinational path from whatever logic
//        drives layer_id, and from approx_multiplier's own (fairly deep)
//        partial-product adder tree -- registering here means the LUT's
//        propagation delay is never added onto the multiplier's own
//        critical path in the same cycle.
//     2. Gives a clean, synchronous, glitch-free approx_level to the MAC:
//        layer_id is expected to change once per convolution layer (i.e.
//        far less often than once per MAC cycle), so the resulting
//        1-cycle registration latency between a layer_id change and the
//        corresponding approx_level update is negligible relative to the
//        number of MAC operations performed per layer, and is a standard,
//        cheap price for a clean synchronous interface.
//   Net effect: layer_id must be stable one cycle before the MAC is meant
//   to use the new approx_level -- document this at the integration site
//   (see adaptive_mac_system.sv).
//
// FAULT-SAFE DEFAULT: L0 (exact) is the output whenever layer_valid=0,
// layer_id is out of the valid 0..NUM_LAYERS-1 range, or immediately after
// reset. L0 is always safe because it performs no approximation.
// ============================================================================

module adaptive_policy_controller #(
    parameter int NUM_LAYERS = 20
)(
    input  logic       clk,
    input  logic       rst_n,        // async active-low reset -> approx_level = L0
    input  logic [4:0] layer_id,     // 0 .. NUM_LAYERS-1 valid (5 bits covers 0..31)
    input  logic       layer_valid,
    output logic [1:0] approx_level  // registered; see architecture note above
);

    // Same encoding as approx_multiplier.sv -- reused, not redefined.
    localparam logic [1:0] L0 = 2'b00;
    localparam logic [1:0] L1 = 2'b01;
    localparam logic [1:0] L2 = 2'b10;
    localparam logic [1:0] L3 = 2'b11;

    // ------------------------------------------------------------
    // OFFLINE-DERIVED POLICY TABLE (Python layer-isolation sensitivity
    // analysis, 90% reference-agreement threshold -- see project results).
    //
    // Implemented as a function with a case statement rather than a
    // localparam array literal: functionally identical (synthesizes to
    // the same small mux/ROM; a case statement over a dense integer
    // range is exactly what a ROM lookup is), and just as trivially
    // replaceable later by a programmable register file or small policy
    // RAM -- only this function's body would change (from a case
    // statement to a memory read), the read interface (index in, 2-bit
    // level out) and everything that calls it stay identical.
    // (Icarus Verilog does not support localparam/parameter unpacked-
    // array '{...} aggregate literals -- this form avoids that tool
    // limitation entirely rather than working around it.)
    // ------------------------------------------------------------
    function automatic logic [1:0] policy_lookup(input logic [4:0] id);
        case (id)
            5'd0  : policy_lookup = L0; // conv1
            5'd1  : policy_lookup = L0; // layer1.0.conv1
            5'd2  : policy_lookup = L0; // layer1.0.conv2
            5'd3  : policy_lookup = L0; // layer1.1.conv1
            5'd4  : policy_lookup = L0; // layer1.1.conv2
            5'd5  : policy_lookup = L0; // layer2.0.conv1
            5'd6  : policy_lookup = L0; // layer2.0.conv2
            5'd7  : policy_lookup = L2; // layer2.0.downsample.0
            5'd8  : policy_lookup = L0; // layer2.1.conv1
            5'd9  : policy_lookup = L0; // layer2.1.conv2
            5'd10 : policy_lookup = L0; // layer3.0.conv1
            5'd11 : policy_lookup = L0; // layer3.0.conv2
            5'd12 : policy_lookup = L3; // layer3.0.downsample.0
            5'd13 : policy_lookup = L0; // layer3.1.conv1
            5'd14 : policy_lookup = L0; // layer3.1.conv2
            5'd15 : policy_lookup = L0; // layer4.0.conv1
            5'd16 : policy_lookup = L0; // layer4.0.conv2
            5'd17 : policy_lookup = L1; // layer4.0.downsample.0
            5'd18 : policy_lookup = L2; // layer4.1.conv1
            5'd19 : policy_lookup = L1; // layer4.1.conv2
            default: policy_lookup = L0; // unreachable when id_in_range
                                          // is checked first, kept for
                                          // defensive completeness
        endcase
    endfunction

    // ------------------------------------------------------------
    // Guarded lookup -- the function is never even called with an
    // out-of-range id in the intended path (id_in_range gates level_next
    // below), but safe_id additionally clamps to 0 so no path ever
    // depends on the function's own default arm.
    // ------------------------------------------------------------
    logic             id_in_range;
    logic [4:0]       safe_id;
    logic [1:0]       looked_up_level;
    logic [1:0]       level_next;

    assign id_in_range     = (layer_id < 5'(NUM_LAYERS));
    assign safe_id         = id_in_range ? layer_id : 5'd0;
    assign looked_up_level = policy_lookup(safe_id);
    assign level_next      = (layer_valid && id_in_range) ? looked_up_level : L0;

    // ------------------------------------------------------------
    // Registered output -- see architecture note above.
    // ------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            approx_level <= L0;
        else
            approx_level <= level_next;
    end

endmodule
