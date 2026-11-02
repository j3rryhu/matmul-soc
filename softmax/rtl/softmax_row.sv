// Integer baseline: signed score /256, exp ROM /65535, probability /65535.
// Synchronous active-high reset. One row buffered at a time, length 1..196.
module softmax_row #(
    parameter LUT_FILE = "softmax/example/exp_lut.hex"
) (
    input wire clk, reset,
    input wire in_valid,
    output wire in_ready,
    input wire signed [15:0] in_score,
    input wire [7:0] row_length, // sampled on the first accepted input only
    output reg out_valid,
    input wire out_ready,
    output reg [15:0] out_probability,
    output reg out_last
);
    localparam RECEIVE=0, SCORE_READ=1, EXP_READ=2, EXP_STORE=3,
               NORMALIZE=4, DIVIDE=5, SEND=6;
    reg [2:0] state;
    reg signed [15:0] scores [0:195];
    reg [15:0] exponentials [0:195];
    reg [15:0] rom [0:4095]; // 3073 useful entries, zero-padded for block ROM
    reg [7:0] count, length, index;
    reg signed [15:0] maximum;
    reg [15:0] exp_value;
    reg signed [15:0] score_value;
    reg [23:0] total;
    wire signed [16:0] difference = $signed({maximum[15], maximum}) -
                                    $signed({score_value[15], score_value});
    wire [11:0] rom_address = difference <= 17'sd3072 ? difference[11:0] : 12'd3073;
    wire [31:0] product = {16'b0, exponentials[index]} * 32'd65535;
    wire [32:0] numerator = {1'b0, product} + {9'b0, (total >> 1)};
    // Restoring unsigned division: one numerator bit per cycle, 33 cycles.
    reg [32:0] dividend;
    reg [24:0] remainder;
    reg [5:0] divide_step;
    wire [24:0] trial = {remainder[23:0], dividend[32]};
    wire subtract = trial >= {1'b0, total};
    wire [32:0] next_dividend = {dividend[31:0], subtract};
    assign in_ready = !reset && state == RECEIVE &&
                      (count != 0 || (row_length >= 1 && row_length <= 196));
    initial $readmemh(LUT_FILE, rom);

    // Unconditional registered reads permit block RAM/ROM inference.
    // Their contents/output registers need no reset: the FSM schedules valid reads.
    always @(posedge clk) begin
        score_value <= scores[index];
        exp_value <= rom[rom_address];
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= RECEIVE;
            count <= 0;
            length <= 0;
            index <= 0;
            maximum <= 16'sh8000;
            total <= 0;
            dividend <= 0;
            remainder <= 0;
            divide_step <= 0;
            out_valid <= 0;
            out_probability <= 0;
            out_last <= 0;
        end else begin
            case (state)
                RECEIVE: if (in_valid && in_ready) begin
                    scores[count] <= in_score;
                    if (count == 0) begin
                        length <= row_length;
                        maximum <= in_score;
                    end else if (in_score > maximum) maximum <= in_score;
                    if ((count == 0 && row_length == 1) ||
                        (count != 0 && count == length-1)) begin
                        count <= 0;
                        index <= 0;
                        total <= 0;
                        state <= SCORE_READ;
                    end else count <= count+1'b1;
                end
                SCORE_READ: state <= EXP_READ;
                EXP_READ: state <= EXP_STORE;
                EXP_STORE: begin
                    exponentials[index] <= exp_value;
                    total <= total + {8'b0, exp_value};
                    if (index == length-1) begin
                        index <= 0;
                        state <= NORMALIZE;
                    end else begin
                        index <= index+1'b1;
                        state <= SCORE_READ;
                    end
                end
                NORMALIZE: begin
                    dividend <= numerator;
                    remainder <= 0;
                    divide_step <= 0;
                    state <= DIVIDE;
                end
                DIVIDE: begin
                    dividend <= next_dividend;
                    if (subtract) remainder <= trial - {1'b0, total};
                    else remainder <= trial;
                    if (divide_step == 32) begin
                        out_probability <= next_dividend[15:0];
                        out_last <= index == length-1;
                        out_valid <= 1;
                        state <= SEND;
                    end else divide_step <= divide_step+1'b1;
                end
                SEND: if (out_valid && out_ready) begin
                    out_valid <= 0;
                    out_last <= 0;
                    if (index == length-1) state <= RECEIVE;
                    else begin
                        index <= index+1'b1;
                        state <= NORMALIZE;
                    end
                end
                default: state <= RECEIVE;
            endcase
        end
    end
endmodule
