`timescale 1ns/1ps
module tb_softmax_row;
    reg clk=0;
    always #5 clk=~clk;
    reg reset=1, in_valid=0, out_ready=0;
    wire in_ready, out_valid, out_last;
    reg signed [15:0] in_score=0;
    reg [7:0] row_length=0;
    wire [15:0] out_probability;
    softmax_row dut(.*);
    reg passed=0;
    integer scores[0:195], expected[0:195];
    integer fd, rc, n, i, row=0, received=0, cycles=0;
    reg [31:0] rng=32'h15cafe;
    reg stalled=0, held_last;
    reg [15:0] held_probability;
    function [31:0] next_rng(input [31:0] x);
        next_rng={x[30:0],x[31]^x[21]^x[1]^x[0]};
    endfunction
    always @(posedge clk) begin
        cycles=cycles+1;
        if (cycles>1000000) $fatal(1,"Timeout");
        if (!reset) begin
            if (stalled && (!out_valid || out_probability !== held_probability || out_last !== held_last))
                $fatal(1,"Output changed while stalled");
            if (out_valid && in_ready) $fatal(1,"Input accepted while output row active");
        end
        stalled <= !reset && out_valid && !out_ready;
        held_probability <= out_probability;
        held_last <= out_last;
    end
    task send_score(input integer value, input integer length_value);
        begin
            @(negedge clk);
            rng=next_rng(rng);
            repeat (rng[1:0]) @(negedge clk);
            in_score=value; row_length=length_value; in_valid=1;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk); in_valid=0;
        end
    endtask
    task do_reset;
        begin
            @(negedge clk); reset=1; in_valid=0; out_ready=0;
            repeat (2) @(posedge clk);
            @(negedge clk); reset=0;
        end
    endtask
    initial begin
        do_reset();
        // Invalid lengths must not start rows.
        @(negedge clk); row_length=0; in_valid=1;
        @(posedge clk); if(in_ready) $fatal(1,"Accepted zero length");
        @(negedge clk); row_length=197;
        @(posedge clk); if(in_ready) $fatal(1,"Accepted length >196");
        @(negedge clk); in_valid=0;
        // Abort a partial input row.
        send_score(123,49); send_score(-456,49); do_reset();
        // Abort active iterative division, not just reception/output.
        send_score(0,1);
        wait(dut.state == 5 && dut.divide_step == 10);
        do_reset();
        // Abort a stalled output, ensuring no stale valid survives reset.
        send_score(0,1);
        wait(out_valid);
        repeat(3) @(negedge clk);
        do_reset();
        @(posedge clk); if(out_valid) $fatal(1,"Stale output after reset");
        fd=$fopen("softmax/example/rows.txt","r");
        if(!fd) $fatal(1,"Missing vectors");
        rc=$fscanf(fd,"%d",n);
        while(rc==1) begin
            if(n<1 || n>196) $fatal(1,"Invalid vector length");
            for(i=0;i<n;i=i+1) begin
                rc=$fscanf(fd,"%d %d",scores[i],expected[i]);
                if(rc!=2) $fatal(1,"Truncated vector");
            end
            for(i=0;i<n;i=i+1) send_score(scores[i],i==0?n:0);
            received=0;
            while(received<n) begin
                @(negedge clk);
                rng=next_rng(rng);
                out_ready=(rng[2:0]!=0 && rng[4:3]!=0);
                @(posedge clk);
                if(out_valid && out_ready) begin
                    if(out_probability !== expected[received])
                        $fatal(1,"Mismatch row %0d index %0d got %0d expected %0d",row,received,out_probability,expected[received]);
                    if(out_last !== (received==n-1)) $fatal(1,"Incorrect out_last");
                    received=received+1;
                end
            end
            @(negedge clk); out_ready=0;
            row=row+1;
            rc=$fscanf(fd,"%d",n);
        end
        $fclose(fd);
        repeat(5) begin
            @(posedge clk);
            if(out_valid) $fatal(1,"Unexpected extra output");
        end
        if(row!=147) $fatal(1,"Wrong row count: %0d",row);
        passed=1;
        $display("PASS: %0d rows, integer outputs, stalls, invalid lengths, reset recovery; %0d cycles",row,cycles);
        $finish;
    end
endmodule
