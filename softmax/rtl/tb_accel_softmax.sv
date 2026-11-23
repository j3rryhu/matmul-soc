`timescale 1ns/1ps
module tb_accel_softmax;
    localparam KEYS = 49;
    reg clk = 0;
    always #6.667 clk = ~clk;
    reg reset_n = 0;
    reg [31:0] avs_address = 0;
    reg avs_read = 0, avs_write = 0;
    reg [127:0] avs_writedata = 0;
    reg [15:0] avs_byteenable = '1;
    reg [10:0] avs_burstcount = 1;
    wire [127:0] avs_readdata;
    wire avs_readdatavalid, avs_waitrequest;
    reg passed = 0;
    accel_top dut(.*);

    reg [15:0] exp_lut [0:255];
    reg [7:0] gelu_lut [0:255];
    reg [127:0] word_data;
    reg [7:0] scores [0:KEYS-1];
    reg [7:0] probabilities [0:KEYS-1];
    integer score_codes [0:KEYS-1];
    integer maximum;
    integer word_index, key, lane, result_count;
    integer total, numerator, quotient, remainder, context_sum;

    task write_word(input [31:0] address, input [127:0] data,
                    input [15:0] enable = '1);
        @(negedge clk);
        avs_address = address;
        avs_writedata = data;
        avs_byteenable = enable;
        avs_write = 1;
        @(posedge clk);
        while (avs_waitrequest) @(posedge clk);
        @(negedge clk);
        avs_write = 0;
        avs_byteenable = '1;
    endtask

    task start_read(input [31:0] address, input [10:0] beats);
        @(negedge clk);
        avs_address = address;
        avs_burstcount = beats;
        avs_read = 1;
        @(posedge clk);
        while (avs_waitrequest) @(posedge clk);
    endtask

    task read_word(input [31:0] address, output [127:0] data);
        start_read(address, 1);
        @(negedge clk);
        avs_read = 0;
        while (!avs_readdatavalid) @(negedge clk);
        data = avs_readdata;
    endtask

    task start_matmul(input [31:0] activation);
        write_word('h90, activation);
        write_word('h00, 1);
        do read_word('h50, word_data); while (!word_data[1]);
    endtask

    task run_matmul(input bit gelu);
        start_matmul(gelu ? 2 : 0);
        start_read('h9000, 4);
        for (word_index = 0; word_index < 4; word_index++) begin
            @(negedge clk);
            avs_read = 0;
            while (!avs_readdatavalid) @(negedge clk);
            for (lane = 0; lane < 16 && word_index*16+lane < KEYS; lane++) begin
                key = word_index*16+lane;
                if (avs_readdata[lane*8 +: 8] !==
                    (gelu ? gelu_lut[scores[key]] : scores[key]))
                    $fatal(1, "Matmul/GELU mismatch at key %0d (GELU %0d): %0h expected %0h",
                           key, gelu, avs_readdata[lane*8 +: 8],
                           gelu ? gelu_lut[scores[key]] : scores[key]);
            end
        end
    endtask

    task check_probabilities(input bit gelu, input bit reversed, input bit biased = 0);
        maximum = -128;
        for (key = 0; key < KEYS; key++) begin
            score_codes[key] = gelu ? $signed(gelu_lut[scores[key]]) : $signed(scores[key]);
            if (reversed) score_codes[key] = -score_codes[key];
            if (biased) score_codes[key] += (key % 5) - 2;
            if (score_codes[key] > maximum) maximum = score_codes[key];
        end
        total = 0;
        for (key = 0; key < KEYS; key++) total += exp_lut[maximum-score_codes[key]];
        result_count = 0;
        while (result_count < KEYS) begin
            read_word('h120, word_data);
            if (word_data[12]) $fatal(1, "OBUF softmax command rejected");
            if (word_data[8]) begin
                numerator = 127 * exp_lut[maximum-score_codes[result_count]];
                quotient = numerator / total;
                remainder = numerator % total;
                if (2*remainder > total || (2*remainder == total && quotient % 2))
                    quotient++;
                if (word_data[7:0] !== quotient[7:0] || word_data[9] !== (result_count == KEYS-1))
                    $fatal(1, "Softmax mismatch at key %0d: got %0d expected %0d",
                           result_count, word_data[7:0], quotient);
                probabilities[result_count] = word_data[7:0];
                result_count++;
            end
        end
        read_word('h120, word_data);
        if (word_data[12:8] !== 5'b00100) $fatal(1, "Row did not drain");
    endtask

    initial begin
        $readmemh("softmax/example/exp_lut.hex", exp_lut);
        $readmemh("activation/example/gelu_int8.hex", gelu_lut);
        repeat (4) @(negedge clk);
        reset_n = 1;
        write_word('h130, KEYS << 16);
        read_word('h120, word_data);
        if (!word_data[12]) $fatal(1, "Direct launch accepted before matmul");
        write_word('h100, 0);
        write_word('h10, 2);
        write_word('h20, 16);
        write_word('h30, KEYS);
        write_word('h40, KEYS);
        write_word('ha0, 2);
        write_word('hb0, KEYS);
        write_word('h60, 16384);
        // W[0,0]=4, W[1,0]=-4 and X[0,k]=k-24. X[1,k] holds the
        // synthetic INT8 relative-position bias for the later test.
        // The 1/4 requantization leaves the real matmul output codes at k-24.
        write_word('h1000, 4);
        write_word('h1010, 'hfc);
        for (word_index = 0; word_index < 64; word_index++) begin
            word_data = 0;
            for (lane = 0; lane < 16 && word_index*16+lane < KEYS; lane++) begin
                key = word_index*16+lane;
                scores[key] = key - 24;
                word_data[lane*8 +: 8] = scores[key];
            end
            for (lane = 0; lane < 16 && word_index >= 4 && word_index*16+lane < 64+KEYS; lane++) begin
                key = word_index*16 + lane - 64;
                word_data[lane*8 +: 8] = (key % 5) - 2;
            end
            write_word('h5000 + word_index*16, word_data);
        end
        run_matmul(0);

        write_word('h100, 197);
        read_word('h120, word_data);
        if (word_data[10:8] !== 0) $fatal(1, "Invalid length accepted");
        write_word('h100, 2);
        write_word('h110, 127);
        write_word('h100, 0); // Discard the partial row.
        write_word('h100, KEYS);
        write_word('h110, 127, 16'h0002); // Lane zero disabled.
        for (key = 0; key < KEYS; key++) write_word('h110, {120'b0, scores[key]});
        check_probabilities(0, 0);

        write_word('h130, (KEYS << 16) | 1);
        read_word('h120, word_data);
        if (!word_data[12]) $fatal(1, "Misaligned OBUF launch accepted");
        write_word('h130, (KEYS << 16) | 64);
        read_word('h120, word_data);
        if (word_data[12:10] !== 3'b010) $fatal(1, "Direct ingest status mismatch");
        write_word('h100, 0);

        // The second padded canvas row starts 64 bytes from OBUF base.
        // Ingest it directly, with a competing host OBUF read held off.
        write_word('h130, (KEYS << 16) | 64);
        read_word('h9000, word_data);
        if (word_data[7:0] !== scores[0])
            $fatal(1, "Host OBUF read was corrupted during direct ingest");
        check_probabilities(0, 1);

        // Confirm the same datapath can consume the final GELU writeback.
        run_matmul(1);
        write_word('h130, KEYS << 16);
        check_probabilities(1, 0);

        // Seed the canvas with the relative-position bias, then add it to
        // scaled QK through the existing prev_int8 final-pass adder.
        write_word('h1000, 'h0400);
        start_matmul(0);
        write_word('h1000, 4);
        write_word('h70, 32'h4000_0000);
        start_matmul(4);
        write_word('h130, KEYS << 16);
        check_probabilities(0, 0, 1);

        // Feed the probability row back into WBUF for P @ V. One value
        // column is sufficient to check the INT8 contraction boundary.
        write_word('h10, 1);
        write_word('h20, KEYS);
        write_word('h30, 1);
        write_word('h40, 1);
        write_word('ha0, 1);
        write_word('hb0, 1);
        write_word('h60, 32768);
        write_word('h70, 0);
        write_word('h90, 0);
        for (word_index = 0; word_index < 4; word_index++) begin
            word_data = 0;
            for (lane = 0; lane < 16 && word_index*16+lane < KEYS; lane++)
                word_data[lane*8 +: 8] = probabilities[word_index*16+lane];
            write_word('h1000 + word_index*16, word_data);
        end
        context_sum = 0;
        for (key = 0; key < KEYS; key++) begin
            write_word('h5000 + key*16, key % 3);
            context_sum += probabilities[key] * (key % 3);
        end
        write_word('h00, 1);
        do read_word('h50, word_data); while (!word_data[1]);
        read_word('h9000, word_data);
        if (word_data[7:0] !== 8'(context_sum >>> 1))
            $fatal(1, "P @ V mismatch: %0d expected %0d", word_data[7:0], context_sum >>> 1);

        $display("PASS: direct OBUF -> softmax for raw QK, biased QK and GELU, host arbitration, and P @ V");
        passed = 1;
        $finish;
    end

    initial begin
        repeat (15000) @(posedge clk);
        $fatal(1, "Integrated accelerator watchdog expired");
    end
endmodule
