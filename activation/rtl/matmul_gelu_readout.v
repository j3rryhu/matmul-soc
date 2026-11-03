// Optional postprocessing of FINAL output-buffer reads. Never changes the
// partial sums stored in RAM. Operands and LUT input share a real-value scale.
module matmul_gelu_readout #(
    parameter DATA_WIDTH = 128,
    parameter BANK_ADDR_WIDTH = 7,
    parameter BANKS = 16,
    parameter LUT_FILE = "activation/example/gelu_int8.hex"
) (
    input clock,
    input reset_n,
    input config_write,
    input [15:0] config_address,
    input [DATA_WIDTH-1:0] config_data,
    input [DATA_WIDTH/8-1:0] config_byteenable,
    input [15:0] columns,
    input [$clog2(BANKS)+BANK_ADDR_WIDTH-$clog2(DATA_WIDTH/8)-1:0] read_word_address,
    input [DATA_WIDTH-1:0] raw_data,
    output reg [DATA_WIDTH-1:0] result
);
    localparam BYTES = DATA_WIDTH/8;
    localparam WORD_SHIFT = $clog2(BYTES);
    localparam BANK_WORD_ADDR_WIDTH = BANK_ADDR_WIDTH-WORD_SHIFT;
    localparam WORDS_PER_BANK = (1 << BANK_WORD_ADDR_WIDTH);
    reg [7:0] bias [0:255];
    reg [7:0] rom [0:255];
    reg enabled;
    reg [15:0] row_index;
    integer i, lane;
    // All live row segments must be wide-word aligned. One bias per word.
    wire geometry_ok = columns != 0 && (columns % BYTES) == 0;
    wire [BANK_WORD_ADDR_WIDTH-1:0] word_offset = read_word_address[BANK_WORD_ADDR_WIDTH-1:0];
    wire [15:0] row_words = columns >> WORD_SHIFT;
    // Count crossed row boundaries instead of dividing by runtime columns.
    // Each threshold is a CONSTANT multiple of row_words; comparisons run in
    // parallel. Full-width products avoid overflow for large column counts.
    wire [WORDS_PER_BANK-1:1] crossed_boundary;
    genvar boundary;
    generate
        for (boundary=1; boundary<WORDS_PER_BANK; boundary=boundary+1) begin : ROW_BOUNDARY
            assign crossed_boundary[boundary] = word_offset >= boundary * row_words;
        end
    endgenerate
    reg [15:0] segment_next;
    integer segment_number;
    always @* begin
        segment_next = 0;
        for (segment_number=1; segment_number<WORDS_PER_BANK; segment_number=segment_number+1)
            if (crossed_boundary[segment_number]) segment_next = segment_number;
    end
    wire [15:0] row_index_next = segment_next * BANKS +
        (read_word_address >> BANK_WORD_ADDR_WIDTH);
    wire signed [7:0] row_bias = bias[row_index[7:0]];
    reg signed [8:0] sum;
    reg [7:0] clipped;
    initial $readmemh(LUT_FILE, rom);
    always @(posedge clock) begin
        if (!reset_n) begin
            enabled <= 0;
            row_index <= 0;
            for (i=0; i<256; i=i+1) bias[i] <= 0;
        end else begin
            // Calculate row selection BEFORE the RAM read edge, then register
            // it alongside the RAM's one-clock response. No extra read cycle.
            row_index <= row_index_next;
            if (config_write) begin
                for (i=0; i<BYTES; i=i+1) begin
                    if (config_byteenable[i]) begin
                        if (config_address[15:8] == 8'h80)
                            bias[(config_address[7:0] & (256-BYTES))+i] <= config_data[i*8 +: 8];
                        if ((config_address & ~(BYTES-1))+i == 16'h8100)
                            enabled <= config_data[i*8];
                    end
                end
            end
        end
    end
    always @* begin
        result = raw_data;
        sum = 0; clipped = 0;
        for (lane=0; lane<BYTES; lane=lane+1) begin
            if (enabled && geometry_ok && row_index < 256) begin
                sum = $signed({raw_data[lane*8+7], raw_data[lane*8 +: 8]}) +
                      $signed({row_bias[7], row_bias});
                clipped = sum > 9'sd127 ? 8'd127 :
                          (sum < -9'sd128 ? 8'h80 : sum[7:0]);
                result[lane*8 +: 8] = rom[clipped];
            end
        end
    end
endmodule
