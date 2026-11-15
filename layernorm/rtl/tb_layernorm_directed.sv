`timescale 1ns/1ps
module tb_layernorm_directed;
    reg clk=0;
    always #5 clk=~clk;
    reg reset=1, in_valid=0, out_ready=1;
    wire in_ready, out_valid, out_last;
    reg signed [7:0] in_data=0, in_gamma=0, in_beta=0;
    reg [9:0] row_length=0;
    wire signed [7:0] out_data;
    layernorm_row dut(.*);
    reg passed=0;
    integer cycles=0, last_input_cycle, previous_output_cycle, received;
    always @(posedge clk) begin
        cycles <= cycles+1;
        if(cycles>1000) $fatal(1,"Directed test timeout");
    end
    // With N=2, q=(-128,127), E=1, the floor root is exactly 65280.
    // z=(-16384,16384). Effective g=62, b=-22 gives exactly +24.5 output
    // codes; reversed signs give -24.5. Ties-to-even must yield +/-24.
    task check_tie_row(input integer sign_value);
        begin
            @(negedge clk);
            in_data=-128; in_gamma=-62*sign_value; in_beta=-22*sign_value;
            row_length=2; in_valid=1;
            @(posedge clk); if(!in_ready) $fatal(1,"Not ready for directed row");
            @(negedge clk);
            in_data=127; in_gamma=62*sign_value; row_length=0;
            @(posedge clk);
            #1; last_input_cycle=cycles;
            @(negedge clk); in_valid=0;
            wait(out_valid);
            #1;
            if(cycles-last_input_cycle!=78) $fatal(1,"First valid latency %0d, expected 78",cycles-last_input_cycle);
            received=0;
            while(received<2) begin
                @(posedge clk);
                if(out_valid && out_ready) begin
                    if($signed(out_data)!=24*sign_value) $fatal(1,"Ties-to-even mismatch");
                    if(out_last !== (received==1)) $fatal(1,"Directed last mismatch");
                    if(received==1 && cycles-previous_output_cycle!=52)
                        $fatal(1,"Output interval %0d, expected 52",cycles-previous_output_cycle);
                    previous_output_cycle=cycles;
                    received=received+1;
                end
            end
        end
    endtask
    initial begin
        repeat(2) @(posedge clk);
        @(negedge clk); reset=0;
        check_tie_row(1);
        check_tie_row(-1);
        passed=1;
        $display("PASS: signed half ties round to even; first-valid latency 78 clocks; output interval 52 clocks");
        $finish;
    end
endmodule
