// ============================================================================
// approx_multiplier.sv
//
// 8x8 signed approximate multiplier implementing PARTIAL-PRODUCT COLUMN
// TRUNCATION, bit-accurate against the Python golden model in
// models/pp_truncation.py (level_multiply / pp_truncated_multiply / pp_terms).
//
// ARCHITECTURE (Option A -- direct two's-complement PP decomposition):
//   The golden model builds each PP-array cell directly from the RAW
//   two's-complement bits of A and B (NOT a sign+magnitude decomposition --
//   see project audit notes). Cell (i,j) contributes:
//
//       raw_pp[i][j] = A[i] & B[j]
//       weight       = 2^(i+j)
//       sign         = negative if EXACTLY ONE of (i==7, j==7) is true
//                      (Baugh-Wooley row/column-7 sign correction),
//                      positive otherwise (including the i=j=7 corner).
//
//   Column c = i+j. Dropping "the k LSB PP columns" means: for the fixed-K
//   instance, cells with c < K are never generated at all (no AND gate, no
//   adder input -- true gate-level elimination, verifiable at synthesis).
//   For the runtime-configurable instance, all 64 AND gates must exist (to
//   serve L0), and cells with c < k_eff are gated to zero before entering
//   the adder network instead.
//
// TWO USES OF THIS SAME MODULE (see project audit -- do not conflate them):
//   FIXED_K = -1 (default): runtime-configurable via approx_level input.
//                Use this instance inside approx_mac_top for the adaptive/
//                programmable system-level design.
//   FIXED_K = 0,2,4,6: compile-time fixed K. Use FOUR SEPARATE synthesis
//                runs of this configuration (one per K) for the Section-29
//                PPA comparison. Do NOT synthesize the runtime-configurable
//                instance four times with approx_level tied to a constant
//                and call that a PPA comparison -- all four would show
//                nearly identical area, since every AND gate must still
//                physically exist to support L0 in that variant.
//
// COMPENSATION:
//   Applied AFTER the signed term-sum, in the signed product domain --
//   there is no separate magnitude/sign-restoration step in Option A, so
//   there is no other point in the pipeline where "compensation before
//   sign restoration" could even apply. Gated exactly as in
//   pp_truncation.py::pp_truncated_multiply: gate = OR(A[0+:K]) & OR(B[0+:K])
//   on the RAW two's-complement low bits of A and B (matches the Python
//   `a & low_mask` / `b & low_mask` bitwise-AND semantics exactly, since
//   Python's bitwise AND on a negative int extracts the same low bits as
//   the two's-complement hardware representation for any width >= K bits).
//
// PRODUCT WIDTH:
//   16-bit signed. Verified sufficient: exact_mac.py asserts
//   PRODUCT_MIN <= a*b <= PRODUCT_MAX for signed INT8xINT8 (worst case
//   (-128)*(-128)=16384, fits in signed 16 bits with 1 bit to spare), and
//   the approximate product's magnitude never exceeds the exact product's
//   plausible range (verified exhaustively against the golden model, see
//   generate_golden_vectors.py / tb_approx_mac.sv).
// ============================================================================

module approx_multiplier #(
    parameter int OPERAND_WIDTH = 8,
    parameter int PRODUCT_WIDTH = 16,
    parameter int FIXED_K       = -1   // -1 = runtime-configurable; else 0/2/4/6
)(
    input  logic signed [OPERAND_WIDTH-1:0] a,
    input  logic signed [OPERAND_WIDTH-1:0] b,
    input  logic [1:0]                      approx_level, // 00=L0 01=L1 10=L2 11=L3
                                                            // ignored when FIXED_K>=0
    output logic signed [PRODUCT_WIDTH-1:0] product
);

    localparam int W    = OPERAND_WIDTH;
    localparam int SUMW = PRODUCT_WIDTH + 2; // headroom for internal term-sum

    // Raw two's-complement bit views (bit0 = LSB), matching
    // multiplier.py::to_bits() / pp_truncation.py::to_bits() exactly.
    wire [W-1:0] a_bits = a;
    wire [W-1:0] b_bits = b;

    // ------------------------------------------------------------
    // Effective K and compensation constant.
    // ------------------------------------------------------------
    logic [3:0]                      k_eff;
    logic signed [PRODUCT_WIDTH-1:0] comp_eff;

    generate
        if (FIXED_K >= 0) begin : g_fixed_sel
            assign k_eff    = FIXED_K[3:0];
            assign comp_eff = (FIXED_K == 0)  ? PRODUCT_WIDTH'(0)  :
                               (FIXED_K == 2)  ? PRODUCT_WIDTH'(1)  :
                               (FIXED_K == 4)  ? PRODUCT_WIDTH'(12) :
                               (FIXED_K == 6)  ? PRODUCT_WIDTH'(80) :
                                                  PRODUCT_WIDTH'(0);
        end else begin : g_runtime_sel
            always_comb begin
                unique case (approx_level)
                    2'b00: begin k_eff = 4'd0; comp_eff = PRODUCT_WIDTH'(0);  end
                    2'b01: begin k_eff = 4'd2; comp_eff = PRODUCT_WIDTH'(1);  end
                    2'b10: begin k_eff = 4'd4; comp_eff = PRODUCT_WIDTH'(12); end
                    2'b11: begin k_eff = 4'd6; comp_eff = PRODUCT_WIDTH'(80); end
                endcase
            end
        end
    endgenerate

    // ------------------------------------------------------------
    // Partial-product generation, Baugh-Wooley sign handling, column gating.
    // ------------------------------------------------------------
    logic signed [SUMW-1:0] term_val [0:W-1][0:W-1];
    logic                   valid    [0:W-1][0:W-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < W; gi = gi + 1) begin : ROW
            for (gj = 0; gj < W; gj = gj + 1) begin : COL

                if (FIXED_K >= 0 && (gi + gj) < FIXED_K) begin : g_omit
                    // TRUE elimination: this cell's AND gate and adder
                    // input do not exist in this netlist at all.
                    assign valid[gi][gj]    = 1'b0;
                    assign term_val[gi][gj] = '0;

                end else begin : g_present
                    wire raw_pp   = a_bits[gi] & b_bits[gj];
                    wire sign_neg = (gi == (W-1)) ^ (gj == (W-1));

                    logic signed [SUMW-1:0] magnitude;
                    assign magnitude = raw_pp ? (SUMW'(1) <<< (gi + gj)) : '0;
                    assign term_val[gi][gj] = sign_neg ? -magnitude : magnitude;

                    assign valid[gi][gj] = (FIXED_K >= 0) ? 1'b1
                                                           : ((gi + gj) >= k_eff);
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------
    // Reduction: sum all gated terms. Fully unrolled combinational adder
    // (64, or fewer for FIXED_K>0, terms) -- correctness/clarity-first per
    // project priority; synthesis will compress this into an efficient
    // tree. A hand-built Wallace/Dadda tree is a later optimization.
    // ------------------------------------------------------------
    logic signed [SUMW-1:0] gated_term [0:W-1][0:W-1];
    generate
        for (gi = 0; gi < W; gi = gi + 1) begin : GROW
            for (gj = 0; gj < W; gj = gj + 1) begin : GCOL
                assign gated_term[gi][gj] = valid[gi][gj] ? term_val[gi][gj] : '0;
            end
        end
    endgenerate

    logic signed [SUMW-1:0] sum_terms;
    integer si, sj;
    always_comb begin
        sum_terms = '0;
        for (si = 0; si < W; si = si + 1)
            for (sj = 0; sj < W; sj = sj + 1)
                sum_terms = sum_terms + gated_term[si][sj];
    end

    // ------------------------------------------------------------
    // Compensation gating: gate = OR(A[0+:K]) & OR(B[0+:K]).
    // Precompute OR-reductions for K=2,4,6 (only 3 possible nonzero K
    // values) and select -- avoids a variable-width/barrel-shifted
    // reduction for a 2-bit-selected K.
    // ------------------------------------------------------------
    wire a_or2 = |a_bits[1:0];
    wire a_or4 = |a_bits[3:0];
    wire a_or6 = |a_bits[5:0];
    wire b_or2 = |b_bits[1:0];
    wire b_or4 = |b_bits[3:0];
    wire b_or6 = |b_bits[5:0];

    logic gate;
    generate
        if (FIXED_K == 0) begin : g_gate_k0
            assign gate = 1'b0;
        end else if (FIXED_K == 2) begin : g_gate_k2
            assign gate = a_or2 & b_or2;
        end else if (FIXED_K == 4) begin : g_gate_k4
            assign gate = a_or4 & b_or4;
        end else if (FIXED_K == 6) begin : g_gate_k6
            assign gate = a_or6 & b_or6;
        end else begin : g_gate_runtime
            always_comb begin
                unique case (approx_level)
                    2'b00: gate = 1'b0;
                    2'b01: gate = a_or2 & b_or2;
                    2'b10: gate = a_or4 & b_or4;
                    2'b11: gate = a_or6 & b_or6;
                endcase
            end
        end
    endgenerate

    // ------------------------------------------------------------
    // Final: sum_terms + (gate ? comp_eff : 0), truncated to PRODUCT_WIDTH.
    // Truncation is safe -- see module header re: proven range.
    // ------------------------------------------------------------
    logic signed [SUMW-1:0] comp_ext;
    assign comp_ext = {{(SUMW-PRODUCT_WIDTH){comp_eff[PRODUCT_WIDTH-1]}}, comp_eff};

    logic signed [SUMW-1:0] product_wide;
    assign product_wide = sum_terms + (gate ? comp_ext : '0);

    assign product = product_wide[PRODUCT_WIDTH-1:0];

endmodule
