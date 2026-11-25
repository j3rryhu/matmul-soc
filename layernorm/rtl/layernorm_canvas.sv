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
    localparam [1:0] IDLE=0, CHECK=1, FETCH=2, WRITE=3;
    reg [1:0] state;
    reg [9:0] length, channel;
    reg [ADDR_WIDTH:0] token_count, pitch;
    reg [ADDR_WIDTH-1:0] token, byte_addr;
    reg rescale;

    // Read the host's coefficient RAM directly while emitting each row.
    wire [WORD_BYTES*8-1:0] coeff_word;
    wire coeff_rd_en;
    wire [9:0] coeff_rd_index;
    wire [10:0] coeff_offset = {coeff_rd_index, 1'b0};
    reg [WORD_SHIFT-1:0] coeff_lane;
    wire [15:0] coeff_pair = coeff_word[coeff_lane*8 +: 16];
    always @(posedge clk)
        if (coeff_rd_en) coeff_lane <= coeff_offset[WORD_SHIFT-1:0];
    // Match the other accelerator memories' synchronous read latency.
    // Host writes are blocked while this port is in use.
    altsyncram #(
        .operation_mode("DUAL_PORT"), .intended_device_family("Cyclone V"),
        .width_a(WORD_BYTES*8), .width_b(WORD_BYTES*8),
        .widthad_a(11-WORD_SHIFT), .widthad_b(11-WORD_SHIFT),
        .numwords_a(MAX_CHANNELS*2/WORD_BYTES), .numwords_b(MAX_CHANNELS*2/WORD_BYTES),
        .width_byteena_a(WORD_BYTES), .byte_size(8),
        .address_reg_b("CLOCK0"), .rdcontrol_reg_b("CLOCK0"),
        .outdata_reg_b("UNREGISTERED"), .read_during_write_mode_mixed_ports("DONT_CARE"),
        .ram_block_type("M10K")
    ) u_coefficients (
        .clock0(clk), .clock1(1'b1), .aclr0(1'b0), .aclr1(1'b0),
        .clocken0(1'b1), .clocken1(1'b1), .clocken2(1'b1), .clocken3(1'b1),
        .address_a(coeff_waddr), .data_a(coeff_wdata),
        .byteena_a(coeff_wren), .wren_a(|coeff_wren), .rden_a(1'b0),
        .address_b(coeff_offset[10:WORD_SHIFT]), .rden_b(coeff_rd_en),
        .byteena_b(1'b1), .wren_b(1'b0), .data_b({WORD_BYTES*8{1'b0}}),
        .addressstall_a(1'b0), .addressstall_b(1'b0),
        .q_a(), .q_b(coeff_word), .eccstatus()
    );

    // This bound covers the actual bank packing, including a partial last
    // group of channels. The padded pitch comes from the matmul canvas.
    wire [10:0] groups = ({1'b0, length} + BANKS - 1) >> BANK_BITS;
    wire [ADDR_WIDTH+11:0] bytes_per_bank = groups * pitch;
    wire [ADDR_WIDTH-1:0] next_byte_addr = byte_addr + pitch[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] next_token = token + 1'b1;
    wire end_of_group = &channel[BANK_BITS-1:0];

    assign busy = state != IDLE;
    reg issued_all, read_pending, read_last, input_valid, input_last;
    wire in_ready, out_valid, out_last;
    wire advance_input = !input_valid || in_ready;
    assign rd_en = state == FETCH && advance_input && !issued_all;
    assign rd_addr = {channel[BANK_BITS-1:0], byte_addr[ADDR_WIDTH-1:WORD_SHIFT]};
    assign wr_bank = channel[BANK_BITS-1:0];
    assign wr_addr = byte_addr;

    reg signed [7:0] input_code;
    // Round q*0.03125/0.14372939763106699 to the nearest INT8 code.
    // Q0.16 factor 14249 gives the same rounded result for all 256 codes.
    wire [7:0] magnitude = input_code[7] ? -input_code : input_code;
    wire [22:0] product = magnitude * 14'd14249;
    wire [7:0] rounded = (product + 23'd32768) >> 16;
    wire signed [7:0] scaled = input_code[7] ? -rounded : rounded;
    wire out_ready = state == WRITE;
    assign wr_en = out_valid && out_ready;

    layernorm_row #(.EPS_FILE(EPS_FILE), .EXTERNAL_COEFF(1)) u_row (
        .clk(clk), .reset(!reset_n),
        .in_valid(input_valid), .in_ready(in_ready),
        .in_data(rescale ? scaled : input_code),
        .in_gamma(coeff_pair[7:0]), .in_beta(coeff_pair[15:8]),
        .row_length(length),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_data(wr_data), .out_last(out_last),
        .coeff_rd_en(coeff_rd_en), .coeff_rd_index(coeff_rd_index)
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
            issued_all <= 0;
            read_pending <= 0;
            read_last <= 0;
            input_valid <= 0;
            input_last <= 0;
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
                FETCH: if (advance_input) begin
                    // Within one token every read uses the same byte lane,
                    // since row pitch is padded to the configured bus width.
                    if (read_pending)
                        input_code <= rd_data[token[WORD_SHIFT-1:0]*8 +: 8];
                    input_valid <= read_pending;
                    input_last <= read_last;
                    read_pending <= rd_en;
                    if (rd_en) begin
                        read_last <= channel == length - 1'b1;
                        if (channel == length - 1'b1) issued_all <= 1;
                        else begin
                            channel <= channel + 1'b1;
                            if (end_of_group) byte_addr <= next_byte_addr;
                        end
                    end
                    if (input_valid && input_last) begin
                        channel <= 0;
                        byte_addr <= token;
                        issued_all <= 0;
                        state <= WRITE;
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
