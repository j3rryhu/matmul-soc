`timescale 1ns/1ps
module tb_layernorm_canvas;
    localparam K = 17, N = 16, M = 17, WORDS = K*2;
    localparam LN_CTRL = 32'h0c00, OBUF = 32'h9000;
    reg clk = 0, reset_n = 0;
    always #5 clk = !clk;
    reg [31:0] address = 0;
    reg read = 0, write = 0;
    reg [127:0] writedata = 0;
    reg [15:0] byteenable = 0;
    reg [10:0] burstcount = 1;
    wire [127:0] readdata;
    wire readdatavalid, waitrequest;
    reg passed = 0;

    accel_top dut (
        .clk(clk), .reset_n(reset_n), .avs_address(address),
        .avs_read(read), .avs_write(write), .avs_writedata(writedata),
        .avs_byteenable(byteenable), .avs_burstcount(burstcount),
        .avs_readdata(readdata), .avs_readdatavalid(readdatavalid),
        .avs_waitrequest(waitrequest)
    );

    reg [127:0] weights [0:K-1], inputs [0:N*2-1], bias [0:4], coeff [0:2];
    reg [127:0] raw [0:WORDS-1], gelu [0:WORDS-1];
    reg [127:0] raw_norm [0:WORDS-1], gelu_norm [0:WORDS-1];

    task automatic put(input [31:0] addr, input [127:0] data, input [15:0] mask = 16'h000f);
        @(negedge clk);
        address = addr;
        writedata = data;
        byteenable = mask;
        write = 1;
        @(posedge clk);
        while (waitrequest) @(posedge clk);
        @(negedge clk);
        write = 0;
    endtask

    task automatic issue_read(input [31:0] addr, input [10:0] beats, output integer stalls);
        @(negedge clk);
        address = addr;
        burstcount = beats;
        read = 1;
        stalls = 0;
        @(posedge clk);
        while (waitrequest) begin
            stalls++;
            @(posedge clk);
        end
        #1 read = 0;
    endtask

    task automatic next_beat(output logic [127:0] data);
        @(negedge clk);
        while (!readdatavalid) @(negedge clk);
        data = readdata;
    endtask

    task automatic status(input [31:0] addr, output logic [127:0] data);
        integer stalls;
        issue_read(addr, 1, stalls);
        next_beat(data);
    endtask

    task automatic check_canvas(input bit activated, normalized, expect_stall);
        integer stalls;
        logic [127:0] actual, expected;
        issue_read(OBUF, WORDS, stalls);
        if (expect_stall && stalls < 2)
            $fatal(1, "Canvas read did not wait for LayerNorm");
        for (int i = 0; i < WORDS; i++) begin
            next_beat(actual);
            if (normalized) expected = activated ? gelu_norm[i] : raw_norm[i];
            else            expected = activated ? gelu[i]      : raw[i];
            if (actual !== expected)
                $fatal(1, "GELU=%0b LN=%0b word %0d expected %h got %h",
                       activated, normalized, i, expected, actual);
        end
    endtask

    task automatic run_pipeline(input bit activated);
        logic [127:0] data;
        put(32'h90, activated ? 3 : 1); // bias, optional GELU
        put(0, 1);                     // matmul_start
        do status(32'h50, data); while (data[1] !== 1'b1);
        check_canvas(activated, 0, 0);

        put(LN_CTRL, 32'h8000_0000 | (M+1)<<10 | K);
        status(LN_CTRL, data);
        if (data !== 4) $fatal(1, "Out-of-canvas command status %h", data);

        put(LN_CTRL, 32'h8000_0000 | (activated ? 32'h4000_0000 : 0) | M<<10 | K);
        status(LN_CTRL, data);
        if (data !== 1) $fatal(1, "LayerNorm busy status %h", data);
        check_canvas(activated, 1, 1);
        status(LN_CTRL, data);
        if (data !== 2) $fatal(1, "LayerNorm done status %h", data);
    endtask

    initial begin
        logic [127:0] data;
        $readmemh("layernorm/rtl/canvas_vectors/canvas_weights.hex", weights);
        $readmemh("layernorm/rtl/canvas_vectors/canvas_input.hex", inputs);
        $readmemh("layernorm/rtl/canvas_vectors/canvas_bias.hex", bias);
        $readmemh("layernorm/rtl/canvas_vectors/canvas_coeff.hex", coeff);
        $readmemh("layernorm/rtl/canvas_vectors/canvas_raw.hex", raw);
        $readmemh("layernorm/rtl/canvas_vectors/canvas_gelu.hex", gelu);
        $readmemh("layernorm/rtl/canvas_vectors/canvas_raw_norm.hex", raw_norm);
        $readmemh("layernorm/rtl/canvas_vectors/canvas_gelu_norm.hex", gelu_norm);
        repeat (4) @(negedge clk);
        reset_n = 1;
        put(LN_CTRL, 32'h8000_0000 | M<<10 | K);
        status(LN_CTRL, data);
        if (data !== 4) $fatal(1, "Command before matmul status %h", data);
        put(32'h10, K);
        put(32'h20, N);
        put(32'h30, M);
        put(32'h40, M);
        put(32'h60, 16384);
        for (int i = 0; i < K; i++) put(32'h1000 + 16*i, weights[i], '1);
        for (int i = 0; i < N*2; i++) put(32'h5000 + 16*i, inputs[i], '1);
        for (int i = 0; i < 5; i++) put(32'h800 + 16*i, bias[i], '1);
        for (int i = 0; i < 3; i++) begin
            put(32'h100 + 16*i, coeff[i], 16'h00ff);
            put(32'h100 + 16*i, {coeff[i][127:64], ~coeff[i][63:0]}, 16'hff00);
        end
        run_pipeline(0);
        run_pipeline(1);
        passed = 1;
        $display("PASS: matmul/bias with and without GELU -> LayerNorm -> Avalon canvas");
        $finish;
    end

    initial begin
        repeat (50000) @(posedge clk);
        $fatal(1, "Timed out");
    end
endmodule
