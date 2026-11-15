// INT8 boundary: calibrated signed scores; signed output codes 0..127, /127.
// Internal exponential ROM remains unsigned 16-bit; division is ties-to-even.
// Synchronous active-high reset. One row buffered at a time, length 1..196.
module softmax_row #(
    parameter LUT_FILE = "softmax/example/exp_lut.hex"
) (
    input wire clk, reset,
    input wire in_valid,
    output wire in_ready,
    input wire signed [7:0] in_score,
    input wire [7:0] row_length, // sampled on the first accepted input only
    output reg out_valid,
    input wire out_ready,
    output reg signed [7:0] out_probability,
    output reg out_last
);
    localparam RECEIVE=0, SCORE_READ=1, EXP_READ=2, EXP_STORE=3,
               NORMALIZE=4, DIVIDE=5, SEND=6, WEIGHT_READ=7, ROUND=8;
    reg [3:0] state;
    reg signed [7:0] scores [0:195];
    reg [15:0] exponentials [0:195];
    reg [15:0] rom [0:255]; // Every possible signed-INT8 score difference.
    reg [7:0] count, length, index;
    reg signed [7:0] maximum;
    reg [15:0] exp_value, weight_value;
    reg signed [7:0] score_value;
    reg [23:0] total;
    wire signed [8:0] difference = $signed({maximum[7], maximum}) -
                                   $signed({score_value[7], score_value});
    wire [7:0] rom_address = difference[7:0];
    // e*127 = (e<<7)-e, unsigned 23-bit exact product.
    wire [22:0] numerator = ({7'b0,weight_value} << 7)-{7'b0,weight_value};
    // Restoring unsigned division, one numerator bit per cycle, 23 cycles.
    reg [22:0] dividend;
    reg [24:0] remainder;
    reg [5:0] divide_step;
    wire [24:0] trial = {remainder[23:0], dividend[22]};
    wire subtract = trial >= {1'b0, total};
    wire [22:0] next_dividend = {dividend[21:0], subtract};
    wire [25:0] twice_remainder = {remainder,1'b0};
    wire round_up = (twice_remainder > {2'b0,total}) ||
                    ((twice_remainder == {2'b0,total}) && dividend[0]);
    wire [23:0] rounded_code = {1'b0,dividend}+{{23{1'b0}},round_up};
    assign in_ready = !reset && state == RECEIVE &&
                      (count != 0 || (row_length >= 1 && row_length <= 196));
    initial $readmemh(LUT_FILE, rom);

    // Unconditional registered reads permit block RAM/ROM inference.
    // Their contents/output registers need no reset: the FSM schedules valid reads.
    always @(posedge clk) begin
        score_value <= scores[index];
        exp_value <= rom[rom_address];
        weight_value <= exponentials[index];
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= RECEIVE;
            count <= 0;
            length <= 0;
            index <= 0;
            maximum <= 8'sh80;
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
                        state <= WEIGHT_READ;
                    end else begin
                        index <= index+1'b1;
                        state <= SCORE_READ;
                    end
                end
                WEIGHT_READ: state <= NORMALIZE;
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
                    if (divide_step == 22) state <= ROUND;
                    else divide_step <= divide_step+1'b1;
                end
                ROUND: begin
                    out_probability <= $signed({1'b0,rounded_code[6:0]});
                    out_last <= index == length-1;
                    out_valid <= 1;
                    state <= SEND;
                end
                SEND: if (out_valid && out_ready) begin
                    out_valid <= 0;
                    out_last <= 0;
                    if (index == length-1) state <= RECEIVE;
                    else begin
                        index <= index+1'b1;
                        state <= WEIGHT_READ;
                    end
                end
                default: state <= RECEIVE;
            endcase
        end
    end
endmodule
