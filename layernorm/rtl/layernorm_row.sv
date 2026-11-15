// Standalone calibrated INT8 LayerNorm, one channel vector buffered at a time.
// Synchronous active-high reset. Length 1..576 is sampled on first accepted beat.
// Paired x/gamma/beta are signed INT8; output is signed INT8, ties-to-even.
// Numeric contract and scale provenance: layernorm/example/manifest.json.
module layernorm_row #(
    parameter EPS_FILE = "layernorm/example/epsilon.hex"
) (
    input wire clk, reset,
    input wire in_valid,
    output wire in_ready,
    input wire signed [7:0] in_data, in_gamma, in_beta,
    input wire [9:0] row_length,
    output reg out_valid,
    input wire out_ready,
    output reg signed [7:0] out_data,
    output reg out_last
);
    localparam RECEIVE=0, VARIANCE=1, ROOT_INIT=2, ROOT_STEP=3,
               READ=4, WAIT_READ=5, NORM_INIT=6, DIVIDE=7,
               NORM_ROUND=8, Z_MULT=9, AFFINE=10, QUANTIZE=11, SEND=12,
               SCALE_MULT=13, SCALE_JOIN=14, QUANT_MAG=15;
    reg [3:0] state;
    reg signed [7:0] data_mem [0:575];
    reg signed [7:0] gamma_mem [0:575];
    reg signed [7:0] beta_mem [0:575];
    reg [7:0] epsilon_rom [0:1023];
    reg signed [7:0] data_value, gamma_value, beta_value;
    // Separate M10K read timing from coefficient multiplication. Capture during
    // the existing read-wait slot, then hold throughout normalization/division.
    reg signed [7:0] gamma_channel, beta_channel;
    reg [7:0] epsilon_value;
    reg [9:0] count, length, index;
    reg signed [17:0] total;
    reg [23:0] squares;
    wire signed [15:0] input_square = $signed(in_data)*$signed(in_data);
    wire [32:0] n_squares = {23'b0,length} * {9'b0,squares};
    wire signed [35:0] total_square = $signed(total)*$signed(total);
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

    // (N*q-S) has 19 signed bits; shifting by R+K=22 requires 41 bits.
    wire signed [18:0] centered = $signed({1'b0,length})*$signed(data_value)-$signed(total);
    wire signed [40:0] normalize_numerator = $signed({{22{centered[18]}},centered}) <<< 22;
    wire [40:0] normalize_magnitude = normalize_numerator[40] ? -normalize_numerator : normalize_numerator;
    // Restoring unsigned division, 41 cycles; round quotient separately.
    reg [40:0] dividend;
    reg [25:0] remainder;
    reg [5:0] divide_step;
    reg negative;
    wire [25:0] divide_trial = {remainder[24:0],dividend[40]};
    wire divide_bit = divide_trial >= {1'b0,denominator};
    wire [40:0] next_dividend = {dividend[39:0],divide_bit};
    wire [26:0] twice_remainder = {remainder,1'b0};
    wire round_normalize = (twice_remainder > {2'b0,denominator}) ||
                         ((twice_remainder == {2'b0,denominator}) && dividend[0]);
    wire [41:0] rounded_magnitude = {1'b0,dividend} + {{41{1'b0}},round_normalize};
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
    initial $readmemh(EPS_FILE, epsilon_rom);
    // No memory resets; explicit states allow registered/block-memory reads.
    always @(posedge clk) begin
        data_value <= data_mem[index];
        gamma_value <= gamma_mem[index];
        beta_value <= beta_mem[index];
        epsilon_value <= epsilon_rom[length];
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= RECEIVE;
            count <= 0; length <= 0; index <= 0;
            total <= 0; squares <= 0; variance_numerator <= 0;
            radicand <= 0; root_remainder <= 0; root_result <= 0;
            denominator <= 0; root_step <= 0;
            dividend <= 0; remainder <= 0; divide_step <= 0; negative <= 0;
            normalized <= 0; norm_gamma <= 0; affine <= 0;
            gamma_channel <= 0; beta_channel <= 0;
            scale_high <= 0; scale_low <= 0; scale_beta <= 0;
            gamma_scaled <= 0; output_magnitude <= 0; output_negative <= 0;
            out_valid <= 0; out_data <= 0; out_last <= 0;
        end else begin
            case (state)
                RECEIVE: if (in_valid && in_ready) begin
                    data_mem[count] <= in_data;
                    gamma_mem[count] <= in_gamma;
                    beta_mem[count] <= in_beta;
                    if (count == 0) begin
                        length <= row_length;
                        total <= {{10{in_data[7]}},in_data};
                        squares <= {8'b0,input_square};
                    end else begin
                        total <= total + {{10{in_data[7]}},in_data};
                        squares <= squares + {8'b0,input_square};
                    end
                    if ((count == 0 && row_length == 1) ||
                        (count != 0 && count == length-1)) begin
                        count <= 0; index <= 0;
                        state <= VARIANCE;
                    end else count <= count+1'b1;
                end
                VARIANCE: begin
                    variance_numerator <= n_squares-total_square[32:0];
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
                        state <= READ;
                    end else root_step <= root_step+1'b1;
                end
                READ: state <= WAIT_READ;
                WAIT_READ: begin
                    gamma_channel <= gamma_value;
                    beta_channel <= beta_value;
                    state <= NORM_INIT;
                end
                NORM_INIT: begin
                    dividend <= normalize_magnitude; remainder <= 0; divide_step <= 0;
                    negative <= normalize_numerator[40];
                    state <= DIVIDE;
                end
                DIVIDE: begin
                    dividend <= next_dividend;
                    remainder <= divide_bit ? divide_trial-{1'b0,denominator} : divide_trial;
                    if (divide_step == 40) state <= NORM_ROUND;
                    else divide_step <= divide_step+1'b1;
                end
                NORM_ROUND: begin
                    normalized <= negative ? -$signed(rounded_magnitude[20:0]) : $signed(rounded_magnitude[20:0]);
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
