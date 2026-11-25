// ============================================================================
// approx_mac_top.sv
//
// Runtime-configurable approximate MAC: ACC <= ACC + approx_multiply(A,B).
// Uses approx_multiplier with FIXED_K=-1 (runtime approx_level input) --
// this is the system-integration variant (per-layer programmable level),
// NOT the fixed-K PPA-comparison variant (see approx_multiplier.sv header).
//
// Accumulator: 32-bit signed, natural two's-complement wraparound on
// overflow (no saturation logic) -- matches config.ACC_OVERFLOW_POLICY
// == "wrap" in the Python model exactly: a plain register add with no
// saturation IS two's-complement wraparound.
// ============================================================================

module approx_mac_top #(
    parameter int OPERAND_WIDTH = 8,
    parameter int PRODUCT_WIDTH = 16,
    parameter int ACC_WIDTH     = 32
)(
    input  logic                            clk,
    input  logic                            rst_n,     // async active-low reset
    input  logic                            clear_acc, // sync clear, priority over enable
    input  logic                            enable,    // perform ACC <= ACC + A*B this cycle
    input  logic signed [OPERAND_WIDTH-1:0] a,
    input  logic signed [OPERAND_WIDTH-1:0] b,
    input  logic [1:0]                      approx_level, // 00=L0 01=L1 10=L2 11=L3
    output logic signed [ACC_WIDTH-1:0]     acc_out
);

    logic signed [PRODUCT_WIDTH-1:0] product;

    approx_multiplier #(
        .OPERAND_WIDTH (OPERAND_WIDTH),
        .PRODUCT_WIDTH (PRODUCT_WIDTH),
        .FIXED_K       (-1)              // runtime-configurable
    ) u_mult (
        .a            (a),
        .b            (b),
        .approx_level (approx_level),
        .product      (product)
    );

    logic signed [ACC_WIDTH-1:0] acc_q;
    logic signed [ACC_WIDTH-1:0] product_ext;

    assign product_ext = {{(ACC_WIDTH-PRODUCT_WIDTH){product[PRODUCT_WIDTH-1]}}, product};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_q <= '0;
        else if (clear_acc)
            acc_q <= '0;
        else if (enable)
            acc_q <= acc_q + product_ext; // natural wraparound, no saturation
    end

    assign acc_out = acc_q;

endmodule
