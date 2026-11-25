// Standalone calibrated INT8 LayerNorm, one channel vector buffered at a time.
// Synchronous active-high reset. Length 1..576 is sampled on first accepted beat.
// Paired x/gamma/beta are signed INT8; output is signed INT8, ties-to-even.
// Numeric contract and scale provenance: layernorm/example/manifest.json.
module layernorm_row #(
    parameter EPS_FILE = "layernorm/example/epsilon.hex",
    parameter EXTERNAL_COEFF = 0
) (
    input wire clk, reset,
    input wire in_valid,
    output wire in_ready,
    input wire signed [7:0] in_data, in_gamma, in_beta,
    input wire [9:0] row_length,
    output reg out_valid,
    input wire out_ready,
    output reg signed [7:0] out_data,
    output reg out_last,
    // In external mode, read gamma/beta here and return them on
    // in_gamma/in_beta one clock later. The standalone mode buffers inputs.
    output wire coeff_rd_en,
    output wire [9:0] coeff_rd_index
);
    localparam RECEIVE=0, VARIANCE=1, ROOT_INIT=2, ROOT_STEP=3,
               READ=4, WAIT_READ=5, NORM_MULT=6, RECIP_STEP=7,
               NORM_ROUND=8, Z_MULT=9, AFFINE=10, QUANTIZE=11, SEND=12,
               SCALE_MULT=13, SCALE_JOIN=14, QUANT_MAG=15,
               STATS_DRAIN=16, VAR_PRODUCTS=17, RECIP_INIT=18,
               NORM_ESTIMATE=19, NORM_PRODUCT=20, NORM_REMAINDER=21;
    reg [4:0] state;
    reg signed [7:0] data_mem [0:575];
    reg [7:0] epsilon_rom [0:1023];
    reg signed [7:0] data_value;
    wire signed [7:0] gamma_value, beta_value;
    // Separate M10K read timing from coefficient multiplication. Capture during
    // the existing read-wait slot, then hold throughout normalization/division.
    reg signed [7:0] gamma_channel, beta_channel;
    reg [7:0] epsilon_value;
    reg [9:0] count, length, index;
    reg signed [17:0] total;
    reg [23:0] squares;
    reg signed [7:0] square_input;
    reg [15:0] input_square;
    reg [1:0] square_valid;
    reg [32:0] n_squares, total_square;
    reg [32:0] variance_numerator;
    wire [32:0] variance_epsilon = variance_numerator + {25'b0,epsilon_value};

    // Digit-by-digit floor square root: 25 bit-pairs of a padded 49-bit argument.
    reg [49:0] radicand;
    reg [26:0] root_remainder;
    reg [24:0] root_result, denominator;
    reg [4:0] root_step;
    wire [26:0] root_trial = {root_remainder[24:0],radicand[49:48]};
    wire [26:0] root_subtrahend = {root_result,2'b01};
    wire root_bit = root_trial >= root_subtrahend;
    wire [24:0] next_root = {root_result[23:0],root_bit};

    // |N*q-S| <= 575*255 < 2^18. The scaled numerator is below 2^40.
    wire signed [18:0] centered = $signed({1'b0,length})*$signed(data_value)-$signed(total);
    reg signed [18:0] centered_reg;
    wire [17:0] centered_magnitude = centered_reg[18] ? -centered_reg : centered_reg;
    // Compute floor(2^40 / denominator) once per token. E >= 1 makes
    // denominator >= 256, so the reciprocal needs at most 33 bits.
    reg [32:0] reciprocal;
    reg [24:0] recip_remainder;
    reg [5:0] recip_step;
    wire [25:0] recip_trial = {recip_remainder, 1'b0};
    wire [26:0] recip_difference = {1'b0,recip_trial} - {2'b0,denominator};
    wire recip_bit = !recip_difference[26];
    reg [26:0] triple_denominator;
    reg [34:0] recip_product_low;
    reg [33:0] recip_product_high;
    wire [20:0] estimate_sum = recip_product_high[20:0] +
                               {3'b0,recip_product_low[34:17]};
    reg [19:0] estimate;
    reg [25:0] estimate_product, remainder;
    // The reciprocal truncation can lower the floor quotient by at most one.
    // Hence the residual is in [0, 2*denominator) and fits in 26 bits.
    // Subtracting the low 26 bits is sufficient to recover this residual.
    wire [26:0] twice_remainder = {remainder,1'b0};
    wire add_one = (twice_remainder > {2'b0,denominator}) ||
                   ((twice_remainder == {2'b0,denominator}) && estimate[0]);
    wire add_two = (twice_remainder > triple_denominator) ||
                   ((twice_remainder == triple_denominator) && !estimate[0]);
    wire [20:0] rounded_magnitude = {1'b0,estimate} + (add_two ? 21'd2 : {20'b0,add_one});
    // For N<=576, |z| <= 2*(floor(sqrt(575*2^28))+1) < 2^20,
    // including floor-root error and normalization rounding. Signed 21 bits
    // therefore preserve the exact integer result; 21x8 needs 29 signed bits.
    reg signed [20:0] normalized;
    reg signed [28:0] norm_gamma;
    // Q16 affine ratios from the calibrated config: Mg=45764, Mb=55988.
    // Split signed 29-bit norm_gamma into signed high 13 and unsigned low 16.
    // Independent <=18x18 products terminate at registers, avoiding chained DSPs.
    reg signed [29:0] scale_high;
    reg [31:0] scale_low;
    reg signed [24:0] scale_beta;
    wire signed [45:0] high_extended = {{16{scale_high[29]}},scale_high};
    wire signed [45:0] low_extended = $signed({14'b0,scale_low});
    wire signed [45:0] beta_extended = {{21{scale_beta[24]}},scale_beta};
    reg signed [45:0] gamma_scaled, affine;
    // Separate absolute value from final rounding/saturation for timing.
    reg [45:0] output_magnitude;
    reg output_negative;
    wire [45:0] affine_quotient = output_magnitude >> 30;
    wire round_affine = (output_magnitude[29:0] > 30'd536870912) ||
                        ((output_magnitude[29:0] == 30'd536870912) && affine_quotient[0]);
    wire [45:0] affine_rounded = affine_quotient + {{45{1'b0}},round_affine};
    wire signed [45:0] output_code = output_negative ? -$signed(affine_rounded) : $signed(affine_rounded);

    assign in_ready = !reset && state == RECEIVE &&
                      (count != 0 || (row_length >= 1 && row_length <= 576));
    assign coeff_rd_en = EXTERNAL_COEFF && !reset && state == READ;
    assign coeff_rd_index = EXTERNAL_COEFF ? index : 10'd0;
    initial $readmemh(EPS_FILE, epsilon_rom);
    // No memory resets; explicit states allow registered/block-memory reads.
    always @(posedge clk) begin
        data_value <= data_mem[index];
        epsilon_value <= epsilon_rom[length];
    end
    generate
        if (EXTERNAL_COEFF) begin : g_external_coeff
            assign gamma_value = in_gamma;
            assign beta_value = in_beta;
        end else begin : g_buffer_coeff
            reg [15:0] affine_mem [0:575];
            reg [15:0] affine_value;
            always @(posedge clk) begin
                if (in_valid && in_ready)
                    affine_mem[count] <= {in_beta, in_gamma};
                affine_value <= affine_mem[index];
            end
            assign {beta_value, gamma_value} = affine_value;
        end
    endgenerate

    always @(posedge clk) begin
        if (reset) begin
            state <= RECEIVE;
            count <= 0; length <= 0; index <= 0;
            total <= 0; squares <= 0; variance_numerator <= 0;
            square_valid <= 0; square_input <= 0; input_square <= 0;
            n_squares <= 0; total_square <= 0;
            radicand <= 0; root_remainder <= 0; root_result <= 0;
            denominator <= 0; root_step <= 0;
            centered_reg <= 0; reciprocal <= 0; recip_remainder <= 0; recip_step <= 0;
            recip_product_low <= 0; recip_product_high <= 0;
            estimate <= 0; estimate_product <= 0; remainder <= 0;
            triple_denominator <= 0;
            normalized <= 0; norm_gamma <= 0; affine <= 0;
            gamma_channel <= 0; beta_channel <= 0;
            scale_high <= 0; scale_low <= 0; scale_beta <= 0;
            gamma_scaled <= 0; output_magnitude <= 0; output_negative <= 0;
            out_valid <= 0; out_data <= 0; out_last <= 0;
        end else begin
            // Each accepted sample contributes once, including through stalls.
            square_valid <= {square_valid[0], in_valid && in_ready};
            if (square_valid[0])
                input_square <= $signed(square_input) * $signed(square_input);
            if (square_valid[1])
                squares <= squares + {8'b0,input_square};
            case (state)
                RECEIVE: if (in_valid && in_ready) begin
                    data_mem[count] <= in_data;
                    square_input <= in_data;
                    if (count == 0) begin
                        length <= row_length;
                        total <= {{10{in_data[7]}},in_data};
                        squares <= 0;
                    end else begin
                        total <= total + {{10{in_data[7]}},in_data};
                    end
                    if ((count == 0 && row_length == 1) ||
                        (count != 0 && count == length-1)) begin
                        count <= 0; index <= 0;
                        state <= STATS_DRAIN;
                    end else count <= count+1'b1;
                end
                STATS_DRAIN: if (!square_valid[0] && !square_valid[1])
                    state <= VAR_PRODUCTS;
                VAR_PRODUCTS: begin
                    n_squares <= length * squares;
                    total_square <= $signed(total) * $signed(total);
                    state <= VARIANCE;
                end
                VARIANCE: begin
                    variance_numerator <= n_squares-total_square;
                    state <= ROOT_INIT;
                end
                ROOT_INIT: begin
                    radicand <= {1'b0,variance_epsilon,16'b0};
                    root_remainder <= 0; root_result <= 0; root_step <= 0;
                    state <= ROOT_STEP;
                end
                ROOT_STEP: begin
                    radicand <= {radicand[47:0],2'b0};
                    root_remainder <= root_bit ? root_trial-root_subtrahend : root_trial;
                    root_result <= next_root;
                    if (root_step == 24) begin
                        denominator <= next_root;
                        state <= RECIP_INIT;
                    end else root_step <= root_step+1'b1;
                end
                RECIP_INIT: begin
                    reciprocal <= 0;
                    recip_remainder <= 1;
                    recip_step <= 0;
                    triple_denominator <= {1'b0, denominator, 1'b0} + {2'b0, denominator};
                    state <= RECIP_STEP;
                end
                RECIP_STEP: begin
                    reciprocal <= {reciprocal[31:0],recip_bit};
                    recip_remainder <= recip_bit ? recip_difference[24:0] : recip_trial[24:0];
                    if (recip_step == 39) state <= READ;
                    else recip_step <= recip_step+1'b1;
                end
                READ: state <= WAIT_READ;
                WAIT_READ: begin
                    gamma_channel <= gamma_value;
                    beta_channel <= beta_value;
                    centered_reg <= centered;
                    state <= NORM_MULT;
                end
                NORM_MULT: begin
                    recip_product_low <= centered_magnitude * reciprocal[16:0];
                    recip_product_high <= centered_magnitude * reciprocal[32:17];
                    state <= NORM_ESTIMATE;
                end
                NORM_ESTIMATE: begin
                    estimate <= estimate_sum[20:1];
                    state <= NORM_PRODUCT;
                end
                NORM_PRODUCT: begin
                    estimate_product <= estimate * denominator;
                    state <= NORM_REMAINDER;
                end
                NORM_REMAINDER: begin
                    remainder <= {centered_magnitude[3:0],22'b0} - estimate_product;
                    state <= NORM_ROUND;
                end
                NORM_ROUND: begin
                    normalized <= centered_reg[18] ? -$signed(rounded_magnitude) : $signed(rounded_magnitude);
                    state <= Z_MULT;
                end
                Z_MULT: begin
                    norm_gamma <= $signed(normalized)*$signed(gamma_channel);
                    state <= SCALE_MULT;
                end
                SCALE_MULT: begin
                    scale_high <= $signed(norm_gamma[28:16])*17'sd45764;
                    scale_low <= norm_gamma[15:0]*16'd45764;
                    scale_beta <= $signed(beta_channel)*17'sd55988;
                    state <= SCALE_JOIN;
                end
                SCALE_JOIN: begin
                    gamma_scaled <= (high_extended <<< 16)+low_extended;
                    state <= AFFINE;
                end
                AFFINE: begin
                    affine <= gamma_scaled + (beta_extended <<< 14);
                    state <= QUANT_MAG;
                end
                QUANT_MAG: begin
                    output_magnitude <= affine[45] ? -affine : affine;
                    output_negative <= affine[45];
                    state <= QUANTIZE;
                end
                QUANTIZE: begin
                    if (output_code > 46'sd127) out_data <= 8'sd127;
                    else if (output_code < -46'sd128) out_data <= 8'sh80;
                    else out_data <= output_code[7:0];
                    out_last <= index == length-1;
                    out_valid <= 1;
                    state <= SEND;
                end
                SEND: if (out_valid && out_ready) begin
                    out_valid <= 0; out_last <= 0;
                    if (index == length-1) state <= RECEIVE;
                    else begin
                        index <= index+1'b1;
                        state <= READ;
                    end
                end
                default: begin
                    state <= RECEIVE; count <= 0; out_valid <= 0; out_last <= 0;
                end
            endcase
        end
    end
endmodule
