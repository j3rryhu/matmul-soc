`timescale 1ns/1ps
module tb_softmax_directed;
    reg clk=0;
    always #5 clk=~clk;
    reg reset=1, in_valid=0, out_ready=1;
    wire in_ready, out_valid, out_last;
    reg signed [7:0] in_score=0;
    reg [7:0] row_length=0;
    wire signed [7:0] out_probability;
    softmax_row dut(.*);
    reg passed=0;
    integer cycles=0, last_input, previous_output, i, received;
    always @(posedge clk) begin
        cycles <= cycles+1;
        if(cycles>30000) $fatal(1,"Directed timeout");
    end
    task equal_row(input integer n, input integer expected);
        begin
            for(i=0;i<n;i=i+1) begin
                @(negedge clk); in_score=0; row_length=i==0?n:0; in_valid=1;
                @(posedge clk); if(!in_ready) $fatal(1,"Not ready");
                #1; last_input=cycles;
            end
            @(negedge clk); in_valid=0;
            wait(out_valid); #1;
            if(cycles-last_input!=3*n+26) $fatal(1,"Incorrect first-valid latency");
            received=0;
            while(received<n) begin
                @(posedge clk);
                if(out_valid && out_ready) begin
                    if($signed(out_probability)!=expected) $fatal(1,"Directed code mismatch");
                    if(out_last !== (received==n-1)) $fatal(1,"Incorrect last");
                    if(received!=0 && cycles-previous_output!=27) $fatal(1,"Incorrect interval");
                    previous_output=cycles; received=received+1;
                end
            end
        end
    endtask
    initial begin
        repeat(2) @(posedge clk);
        @(negedge clk); reset=0;
        equal_row(1,127);
        equal_row(2,64); // 63.5 rounds to even 64, rather than 63.
        equal_row(49,3);
        equal_row(196,1);
        passed=1;
        $display("PASS: signed INT8 range, ties-to-even, equal-row mass examples; first valid 3*N+26 clocks, interval 27 clocks");
        $finish;
    end
endmodule
