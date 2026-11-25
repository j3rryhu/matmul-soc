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
               OUTPUT_ROW=4, WEIGHT_READ=5;
    reg [2:0] state;
    // One M10K holds the row scratch space in its lower half and the
    // exponential lookup table in its upper half.
    (* ramstyle = "no_rw_check" *) reg [15:0] table_mem [0:511];
    reg [7:0] count, length, index;
    reg signed [7:0] maximum;
    reg [15:0] row_value;
    wire signed [7:0] score_value = row_value[7:0];
    reg [23:0] total;
    wire signed [8:0] difference = $signed({maximum[7], maximum}) -
                                   $signed({score_value[7], score_value});
    wire [7:0] rom_address = difference[7:0];
    // e*127 = (e<<7)-e, unsigned 23-bit exact product.
    wire [22:0] numerator = ({7'b0,row_value} << 7)-{7'b0,row_value};
    // One registered stage per radix-4 digit. The entire pipeline, including
    // the synchronous RAM read, holds when a valid output is backpressured.
    reg [32:0] division_pipe [0:4];
    reg [4:0] valid_pipe, last_pipe;
    reg weight_valid, weight_last, issued_all;
    reg [25:0] triple_total;
    wire advance_output = state == OUTPUT_ROW && (!out_valid || out_ready);
    wire [25:0] twice_remainder = {division_pipe[4][32:8],1'b0};
    wire round_up = (twice_remainder > {2'b0,total}) ||
                    ((twice_remainder == {2'b0,total}) && division_pipe[4][0]);
    wire [8:0] rounded_code = {1'b0, division_pipe[4][7:0]}+{{8{1'b0}},round_up};

    function automatic [32:0] divide_digit(
        input [32:0] partial,
        input [23:0] divisor,
        input [25:0] triple
    );
        reg [25:0] trial;
        reg [26:0] minus_one, minus_two, minus_three;
        reg [1:0] digit;
        reg [24:0] remainder;
        begin
            trial = partial[31:6];
            minus_one = {1'b0, trial} - {3'b0, divisor};
            minus_two = {1'b0, trial} - {2'b0, divisor, 1'b0};
            minus_three = {1'b0, trial} - {1'b0, triple};
            digit = !minus_two[26] ? {1'b1, !minus_three[26]} :
                                     {1'b0, !minus_one[26]};
            remainder = !minus_three[26] ? minus_three[24:0] :
                        !minus_two[26]   ? minus_two[24:0] :
                        !minus_one[26]   ? minus_one[24:0] : trial[24:0];
            divide_digit = {remainder, partial[5:0], digit};
        end
    endfunction
    assign in_ready = !reset && state == RECEIVE &&
                      (count != 0 || (row_length >= 1 && row_length <= 196));
    initial $readmemh(LUT_FILE, table_mem, 256, 511);

    wire receive_score = in_valid && in_ready;
    wire write_row = receive_score || (!reset && state == EXP_STORE);
    wire [7:0] write_addr = receive_score ? count : index;
    wire [15:0] write_data = receive_score ? {8'b0, in_score} : row_value;
    wire more = index != length-1;
    wire prefetch = more && state == EXP_STORE;
    wire [7:0] row_read_index = prefetch ? index+1'b1 : index;
    wire [8:0] read_addr = state == EXP_READ ? {1'b1, rom_address} :
                                               {1'b0, row_read_index};
    // While writing an exponential, read the next score from a different
    // address. In the output phase, read one exponential per enabled cycle.
    always @(posedge clk) begin
        if (write_row)
            table_mem[{1'b0, write_addr}] <= write_data;
        if (state != OUTPUT_ROW || advance_output)
            row_value <= table_mem[read_addr];
    end

    integer stage;
    always @(posedge clk) begin
        if (reset) begin
            state <= RECEIVE;
            count <= 0;
            length <= 0;
            index <= 0;
            maximum <= 8'sh80;
            total <= 0;
            valid_pipe <= 0;
            last_pipe <= 0;
            weight_valid <= 0;
            weight_last <= 0;
            issued_all <= 0;
            triple_total <= 0;
            out_valid <= 0;
            out_probability <= 0;
            out_last <= 0;
        end else begin
            case (state)
                RECEIVE: if (in_valid && in_ready) begin
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
                    total <= total + {8'b0, row_value};
                    if (!more) begin
                        index <= 0;
                        state <= WEIGHT_READ;
                    end else begin
                        index <= index+1'b1;
                        state <= EXP_READ;
                    end
                end
                WEIGHT_READ: begin
                    triple_total <= {1'b0, total, 1'b0} + {2'b0, total};
                    weight_valid <= 1;
                    weight_last <= !more;
                    issued_all <= !more;
                    if (more) index <= index+1'b1;
                    valid_pipe <= 0;
                    state <= OUTPUT_ROW;
                end
                OUTPUT_ROW: if (advance_output) begin
                    weight_valid <= !issued_all;
                    weight_last <= !more;
                    if (!issued_all) begin
                        if (more) index <= index+1'b1;
                        else issued_all <= 1;
                    end
                    valid_pipe <= {valid_pipe[3:0], weight_valid};
                    last_pipe <= {last_pipe[3:0], weight_last};
                    // numerator <= 127*total: the prefix is the initial remainder.
                    division_pipe[0] <= {10'b0, numerator};
                    for (stage=1; stage<=4; stage=stage+1)
                        division_pipe[stage] <= divide_digit(division_pipe[stage-1], total, triple_total);
                    out_valid <= valid_pipe[4];
                    out_last <= valid_pipe[4] && last_pipe[4];
                    out_probability <= $signed({1'b0,rounded_code[6:0]});
                    if (out_valid && out_last) state <= RECEIVE;
                end
                default: state <= RECEIVE;
            endcase
        end
    end
endmodule
