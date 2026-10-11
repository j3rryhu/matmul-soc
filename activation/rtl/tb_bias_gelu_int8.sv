`timescale 1ns/1ps
module tb_bias_gelu_int8;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;
    reg signed [7:0] matmul_data = 0, bias_data = 0;
    reg in_valid = 0, out_ready = 0;
    wire in_ready, out_valid;
    wire signed [7:0] out_data;
    reg [7:0] lut [0:255];
    reg [7:0] queue [0:65535];
    integer sent = 0, received = 0, sum, clipped;
    integer a, b, ticks = 0;
    reg stalled = 0;
    reg passed = 0;
    reg [7:0] held;
    bias_gelu_int8 dut(.*);
    initial $readmemh("activation/example/gelu_int8.hex", lut);
    always @(posedge clk) begin
        if (reset_n) begin
            if (stalled && (!out_valid || out_data !== held))
                $fatal(1, "Output changed while stalled");
            if (out_valid && out_ready) begin
                if (received >= sent || out_data !== queue[received])
                    $fatal(1, "Bias/GELU mismatch at transfer %0d", received);
                received = received + 1;
            end
            if (in_valid && in_ready) begin
                // Integer arithmetic independently models widened addition.
                sum = $signed(matmul_data);
                sum = sum + $signed(bias_data);
                clipped = sum > 127 ? 127 : (sum < -128 ? -128 : sum);
                if ($signed(dut.wide_sum) !== sum ||
                    $signed(dut.clipped_sum) !== clipped)
                    $fatal(1, "Adder/saturation mismatch for %0d + %0d",
                           $signed(matmul_data), $signed(bias_data));
                queue[sent] = lut[clipped & 255];
                sent = sent + 1;
            end
            stalled = out_valid && !out_ready;
            held = out_data;
        end
    end
    initial begin
        repeat (3) @(negedge clk);
        reset_n = 1;
        for (a = -128; a <= 127; a = a + 1)
            for (b = -128; b <= 127; b = b + 1) begin
                @(negedge clk);
                matmul_data = a;
                bias_data = b;
                in_valid = 0;
                // Include bubbles and hold the paired operands until accepted.
                ticks = ticks + 1;
                out_ready = ticks % 5 != 0;
                if (ticks % 7 == 0) @(negedge clk);
                in_valid = 1;
                @(posedge clk);
                while (!in_ready) begin
                    @(negedge clk);
                    out_ready = 1;
                    @(posedge clk);
                end
            end
        @(negedge clk);
        in_valid = 0;
        out_ready = 1;
        while (received < sent) @(negedge clk);
        if (received != 65536) $fatal(1, "Wrong transfer count");
        $display("PASS: all 65536 int8 bias pairs, saturation and backpressure");
        passed = 1;
        $finish;
    end
    initial begin
        #3000000;
        $fatal(1, "Simulation timeout");
    end
endmodule
