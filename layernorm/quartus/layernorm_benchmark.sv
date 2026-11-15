// Registered virtual I/O for standalone resource/register-path timing checks.
// This is a measurement harness, not a protocol-preserving SoC wrapper.
module layernorm_benchmark (
    input wire clk, reset, in_valid, out_ready,
    input wire signed [7:0] in_data, in_gamma, in_beta,
    input wire [9:0] row_length,
    output reg in_ready, out_valid, out_last,
    output reg signed [7:0] out_data
);
    reg reset_r, in_valid_r, out_ready_r;
    reg signed [7:0] data_r, gamma_r, beta_r;
    reg [9:0] length_r;
    wire core_in_ready, core_out_valid, core_out_last;
    wire signed [7:0] core_out_data;
    always @(posedge clk) begin
        reset_r <= reset;
        in_valid_r <= in_valid;
        out_ready_r <= out_ready;
        data_r <= in_data; gamma_r <= in_gamma; beta_r <= in_beta;
        length_r <= row_length;
        in_ready <= core_in_ready;
        out_valid <= core_out_valid;
        out_last <= core_out_last;
        out_data <= core_out_data;
    end
    layernorm_row #(.EPS_FILE("epsilon.hex")) core (
        .clk(clk), .reset(reset_r), .in_valid(in_valid_r), .in_ready(core_in_ready),
        .in_data(data_r), .in_gamma(gamma_r), .in_beta(beta_r), .row_length(length_r),
        .out_valid(core_out_valid), .out_ready(out_ready_r),
        .out_data(core_out_data), .out_last(core_out_last)
    );
endmodule
