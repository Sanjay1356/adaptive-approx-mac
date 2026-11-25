// ============================================================================
// adaptive_mac_system.sv
//
// Integration example ONLY: wires adaptive_policy_controller's approx_level
// output directly into the EXISTING, UNMODIFIED approx_mac_top /
// approx_multiplier. Neither of those two files is changed by this module
// -- this is pure top-level wiring.
//
// TIMING NOTE (see adaptive_policy_controller.sv header): approx_level is
// registered inside the controller, so it updates one cycle after
// layer_id/layer_valid are presented. Drive layer_id for the layer you are
// about to feed operands for at least one cycle before the corresponding
// (a,b,enable) pair reaches approx_mac_top, or hold layer_id constant for
// the whole duration of a layer's MAC sequence (the normal case -- layer_id
// changes once per convolution layer, not once per MAC operation).
// ============================================================================

module adaptive_mac_system #(
    parameter int OPERAND_WIDTH = 8,
    parameter int PRODUCT_WIDTH = 16,
    parameter int ACC_WIDTH     = 32,
    parameter int NUM_LAYERS    = 20
)(
    input  logic                            clk,
    input  logic                            rst_n,

    // Policy control (per-layer)
    input  logic [4:0]                      layer_id,
    input  logic                            layer_valid,

    // MAC control/data (per-operation)
    input  logic                            clear_acc,
    input  logic                            enable,
    input  logic signed [OPERAND_WIDTH-1:0] a,
    input  logic signed [OPERAND_WIDTH-1:0] b,

    output logic signed [ACC_WIDTH-1:0]     acc_out,
    output logic [1:0]                      approx_level_used // observability only
);

    logic [1:0] approx_level;

    adaptive_policy_controller #(
        .NUM_LAYERS (NUM_LAYERS)
    ) u_policy (
        .clk         (clk),
        .rst_n       (rst_n),
        .layer_id    (layer_id),
        .layer_valid (layer_valid),
        .approx_level(approx_level)
    );

    // approx_mac_top and approx_multiplier are used exactly as already
    // verified -- FIXED_K=-1 (runtime-configurable) internally, unchanged.
    approx_mac_top #(
        .OPERAND_WIDTH (OPERAND_WIDTH),
        .PRODUCT_WIDTH (PRODUCT_WIDTH),
        .ACC_WIDTH     (ACC_WIDTH)
    ) u_mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .clear_acc   (clear_acc),
        .enable      (enable),
        .a           (a),
        .b           (b),
        .approx_level(approx_level),
        .acc_out     (acc_out)
    );

    assign approx_level_used = approx_level;

endmodule
