// Registered virtual I/O isolates core timing from board/package I/O delays.
module softmax_benchmark (
    input wire clk, reset, in_valid, out_ready,
    input wire signed [15:0] in_score,
    input wire [7:0] row_length,
    output reg in_ready, out_valid, out_last,
    output reg [15:0] out_probability
);
    reg reset_r, in_valid_r, out_ready_r;
    reg signed [15:0] in_score_r;
    reg [7:0] row_length_r;
    wire core_in_ready, core_out_valid, core_out_last;
    wire [15:0] core_probability;
    always @(posedge clk) begin
        reset_r <= reset;
        in_valid_r <= in_valid;
        out_ready_r <= out_ready;
        in_score_r <= in_score;
        row_length_r <= row_length;
        in_ready <= core_in_ready;
        out_valid <= core_out_valid;
        out_last <= core_out_last;
        out_probability <= core_probability;
    end
    softmax_row #(.LUT_FILE("exp_lut.hex")) core (
        .clk(clk), .reset(reset_r), .in_valid(in_valid_r), .in_ready(core_in_ready),
        .in_score(in_score_r), .row_length(row_length_r), .out_valid(core_out_valid),
        .out_ready(out_ready_r), .out_probability(core_probability), .out_last(core_out_last)
    );
endmodule
