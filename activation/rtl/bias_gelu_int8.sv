// Contract: matmul_data and bias_data share the GELU LUT input scale.
// bias_data is paired with matmul_data on every accepted input transfer.
module bias_gelu_int8 #(
    parameter LUT_FILE = "activation/example/gelu_int8.hex"
) (
    input wire clk,
    input wire reset_n,
    input wire signed [7:0] matmul_data,
    input wire signed [7:0] bias_data,
    input wire in_valid,
    output wire in_ready,
    output wire signed [7:0] out_data,
    output wire out_valid,
    input wire out_ready
);
    wire signed [8:0] wide_sum;
    wire signed [7:0] clipped_sum;
    assign wide_sum = $signed({matmul_data[7], matmul_data})
                    + $signed({bias_data[7], bias_data});
    assign clipped_sum = (wide_sum > 9'sd127) ? 8'sd127 :
                         (wide_sum < -9'sd128) ? 8'sh80 : wide_sum[7:0];
    gelu_int8 #(.LUT_FILE(LUT_FILE)) activation (
        .clk(clk), .reset_n(reset_n),
        .in_data(clipped_sum), .in_valid(in_valid), .in_ready(in_ready),
        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready)
    );
endmodule
