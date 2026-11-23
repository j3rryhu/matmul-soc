// Normalize one token (one canvas column) at a time across output channels.
// The parent grants exclusive access to the canvas for the duration of busy.
`timescale 1ps/1ps
module layernorm_canvas #(
    parameter BANKS = 16,
    parameter ADDR_WIDTH = 12,
    parameter WORD_BYTES = 16,
    parameter EPS_FILE = "layernorm/example/epsilon.hex",
    parameter BANK_BITS = $clog2(BANKS),
    parameter WORD_SHIFT = $clog2(WORD_BYTES)
) (
    input wire clk, reset_n,
    input wire start, canvas_ready, gelu_scale,
    input wire [9:0] channels,
    input wire [ADDR_WIDTH:0] tokens, row_pitch,
    input wire [10-WORD_SHIFT:0] coeff_waddr,
    input wire [WORD_BYTES*8-1:0] coeff_wdata,
    input wire [WORD_BYTES-1:0] coeff_wren,
    output wire busy,
    output reg done, error,
    output wire rd_en,
    output wire [BANK_BITS+ADDR_WIDTH-WORD_SHIFT-1:0] rd_addr,
    input wire [WORD_BYTES*8-1:0] rd_data,
    output wire wr_en,
    output wire [BANK_BITS-1:0] wr_bank,
    output wire [ADDR_WIDTH-1:0] wr_addr,
    output wire signed [7:0] wr_data
);
    localparam MAX_CHANNELS = 576;
    localparam BANK_DEPTH = 1 << ADDR_WIDTH;
    localparam [2:0] IDLE=0, CHECK=1, FETCH=2, CAPTURE=3, FEED=4, WRITE=5;
    reg [2:0] state;
    reg [9:0] length, channel;
    reg [ADDR_WIDTH:0] token_count, pitch;
    reg [ADDR_WIDTH-1:0] token, byte_addr;
    reg rescale;

    // Consecutive bytes contain gamma/beta pairs in channel order.
    wire [WORD_BYTES*8-1:0] coeff_word;
    wire [10:0] coeff_offset = {channel, 1'b0};
    // Independent byte lanes infer RAM on Quartus versions that map
    // byte-select writes to a single packed array into flip-flops.
    genvar lane;
    generate
        for (lane = 0; lane < WORD_BYTES; lane = lane + 1) begin : g_coeff_lane
            reg [7:0] mem [0:MAX_CHANNELS*2/WORD_BYTES-1];
            reg [7:0] q;
            always @(posedge clk) begin
                if (coeff_wren[lane])
                    mem[coeff_waddr] <= coeff_wdata[lane*8 +: 8];
                if (state == FETCH)
                    q <= mem[coeff_offset[10:WORD_SHIFT]];
            end
            assign coeff_word[lane*8 +: 8] = q;
        end
    endgenerate

    // This bound covers the actual bank packing, including a partial last
    // group of channels. The padded pitch comes from the matmul canvas.
    wire [10:0] groups = ({1'b0, length} + BANKS - 1) >> BANK_BITS;
    wire [ADDR_WIDTH+11:0] bytes_per_bank = groups * pitch;
    wire [ADDR_WIDTH-1:0] next_byte_addr = byte_addr + pitch[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] next_token = token + 1'b1;
    wire end_of_group = &channel[BANK_BITS-1:0];

    assign busy = state != IDLE;
    assign rd_en = state == FETCH;
    assign rd_addr = {channel[BANK_BITS-1:0], byte_addr[ADDR_WIDTH-1:WORD_SHIFT]};
    assign wr_bank = channel[BANK_BITS-1:0];
    assign wr_addr = byte_addr;

    reg signed [7:0] input_code;
    wire [15:0] coeff_pair = coeff_word[coeff_offset[WORD_SHIFT-1:0]*8 +: 16];
    // Round q*0.03125/0.14372939763106699 to the nearest INT8 code.
    // Q0.16 factor 14249 gives the same rounded result for all 256 codes.
    wire [7:0] magnitude = input_code[7] ? -input_code : input_code;
    wire [22:0] product = magnitude * 14'd14249;
    wire [7:0] rounded = (product + 23'd32768) >> 16;
    wire signed [7:0] scaled = input_code[7] ? -rounded : rounded;
    wire in_ready, out_valid, out_last;
    wire out_ready = state == WRITE;
    assign wr_en = out_valid && out_ready;

    layernorm_row #(.EPS_FILE(EPS_FILE)) u_row (
        .clk(clk), .reset(!reset_n),
        .in_valid(state == FEED), .in_ready(in_ready),
        .in_data(rescale ? scaled : input_code),
        .in_gamma(coeff_pair[7:0]), .in_beta(coeff_pair[15:8]),
        .row_length(length),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_data(wr_data), .out_last(out_last)
    );

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= IDLE;
            done <= 0;
            error <= 0;
            length <= 0;
            token_count <= 0;
            pitch <= 0;
            rescale <= 0;
            channel <= 0;
            token <= 0;
            byte_addr <= 0;
            input_code <= 0;
        end else begin
            case (state)
                IDLE: if (start) begin
                    done <= 0;
                    error <= !canvas_ready;
                    if (canvas_ready) begin
                        length <= channels;
                        token_count <= tokens;
                        pitch <= row_pitch;
                        rescale <= gelu_scale;
                        channel <= 0;
                        token <= 0;
                        byte_addr <= 0;
                        state <= CHECK;
                    end
                end
                CHECK: begin
                    if (length == 0 || length > MAX_CHANNELS ||
                        token_count == 0 || token_count > pitch ||
                        bytes_per_bank > BANK_DEPTH) begin
                        error <= 1;
                        state <= IDLE;
                    end else state <= FETCH;
                end
                FETCH: state <= CAPTURE;
                CAPTURE: begin
                    input_code <= rd_data[byte_addr[WORD_SHIFT-1:0]*8 +: 8];
                    state <= FEED;
                end
                FEED: if (in_ready) begin
                    if (channel == length - 1'b1) begin
                        channel <= 0;
                        byte_addr <= token;
                        state <= WRITE;
                    end else begin
                        channel <= channel + 1'b1;
                        if (end_of_group) byte_addr <= next_byte_addr;
                        state <= FETCH;
                    end
                end
                WRITE: if (out_valid) begin
                    if (out_last) begin
                        if ({1'b0, token} + 1'b1 == token_count) begin
                            done <= 1;
                            state <= IDLE;
                        end else begin
                            channel <= 0;
                            token <= next_token;
                            byte_addr <= next_token;
                            state <= FETCH;
                        end
                    end else begin
                        channel <= channel + 1'b1;
                        if (end_of_group) byte_addr <= next_byte_addr;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
