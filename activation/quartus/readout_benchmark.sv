// Timing harness only: registered virtual I/O, not an integration wrapper.
module readout_benchmark (
    input wire clock, reset_n, config_write,
    input wire [15:0] config_address, config_byteenable, columns,
    input wire [127:0] config_data, raw_data,
    input wire [6:0] read_word_address,
    output reg [127:0] result
);
    reg reset_r, write_r;
    reg [15:0] address_r, byteenable_r, columns_r;
    reg [127:0] data_r, raw_r;
    reg [6:0] read_address_r;
    wire [127:0] core_result;
    always @(posedge clock) begin
        reset_r <= reset_n;
        write_r <= config_write;
        address_r <= config_address;
        byteenable_r <= config_byteenable;
        columns_r <= columns;
        data_r <= config_data;
        raw_r <= raw_data;
        read_address_r <= read_word_address;
        result <= core_result;
    end
    matmul_gelu_readout #(.LUT_FILE("gelu_int8.hex")) core (
        .clock(clock), .reset_n(reset_r), .config_write(write_r),
        .config_address(address_r), .config_byteenable(byteenable_r),
        .columns(columns_r), .config_data(data_r), .raw_data(raw_r),
        .read_word_address(read_address_r), .result(core_result)
    );
endmodule
