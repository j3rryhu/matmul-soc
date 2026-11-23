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
    reg [7:0] qk_from_obuf [0:KEYS-1];
    reg [7:0] probabilities [0:KEYS-1];
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

    task run_matmul(input bit gelu);
        write_word('h90, gelu ? 2 : 0);
        write_word('h00, 1);
        do read_word('h50, word_data); while (!word_data[1]);
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
                if (!gelu) qk_from_obuf[key] = avs_readdata[lane*8 +: 8];
            end
        end
    endtask

    initial begin
        $readmemh("softmax/example/exp_lut.hex", exp_lut);
        $readmemh("activation/example/gelu_int8.hex", gelu_lut);
        repeat (4) @(negedge clk);
        reset_n = 1;
        write_word('h10, 1);
        write_word('h20, 16);
        write_word('h30, KEYS);
        write_word('h40, KEYS);
        write_word('ha0, 1);
        write_word('hb0, KEYS);
        write_word('h60, 16384);
        // W[0,0]=4 and X[0,k]=k-24. Other contraction lanes are zero.
        // The 1/4 requantization leaves the real matmul output codes at k-24.
        write_word('h1000, 4);
        for (word_index = 0; word_index < 64; word_index++) begin
            word_data = 0;
            for (lane = 0; lane < 16 && word_index*16+lane < KEYS; lane++) begin
                key = word_index*16+lane;
                scores[key] = key - 24;
                word_data[lane*8 +: 8] = scores[key];
            end
            write_word('h5000 + word_index*16, word_data);
        end
        run_matmul(0);
        run_matmul(1);

        write_word('h100, 197);
        read_word('h120, word_data);
        if (word_data[10:8] !== 0) $fatal(1, "Invalid length accepted");
        write_word('h100, 2);
        write_word('h110, 127);
        write_word('h100, 0); // Discard the partial row.
        write_word('h100, KEYS);
        write_word('h110, 127, 16'h0002); // Lane zero disabled.
        for (key = 0; key < KEYS; key++) write_word('h110, {120'b0, qk_from_obuf[key]});
        run_matmul(0);

        total = 0;
        for (key = 0; key < KEYS; key++) total += exp_lut[KEYS-1-key];
        result_count = 0;
        while (result_count < KEYS) begin
            read_word('h120, word_data);
            if (word_data[8]) begin
                numerator = 127 * exp_lut[KEYS-1-result_count];
                quotient = numerator / total;
                remainder = numerator % total;
                if (2*remainder > total || (2*remainder == total && quotient % 2))
                    quotient++;
                if (word_data[7:0] !== quotient[7:0] || word_data[9] !== (result_count == KEYS-1))
                    $fatal(1, "Softmax mismatch at key %0d", result_count);
                probabilities[result_count] = word_data[7:0];
                result_count++;
            end
        end
        read_word('h120, word_data);
        if (word_data[10:8] !== 3'b100) $fatal(1, "Row did not drain");

        // Feed the probability row back into WBUF for P @ V. One value
        // column is sufficient to check the INT8 contraction boundary.
        write_word('h20, KEYS);
        write_word('h30, 1);
        write_word('h40, 1);
        write_word('hb0, 1);
        write_word('h60, 32768);
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

        $display("PASS: 49-key INT8 QK -> softmax -> P @ V, GELU, Avalon byte enable and abort");
        passed = 1;
        $finish;
    end

    initial begin
        repeat (15000) @(posedge clk);
        $fatal(1, "Integrated accelerator watchdog expired");
    end
endmodule
