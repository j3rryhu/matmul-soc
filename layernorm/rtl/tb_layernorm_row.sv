`timescale 1ns/1ps
module tb_layernorm_row;
    reg clk=0;
    always #5 clk=~clk;
    reg reset=1, in_valid=0, out_ready=0;
    wire in_ready, out_valid, out_last;
    reg signed [7:0] in_data=0, in_gamma=0, in_beta=0;
    reg [9:0] row_length=0;
    wire signed [7:0] out_data;
    layernorm_row dut(.*);
    reg passed=0;
    integer values[0:575], gammas[0:575], betas[0:575], expected[0:575];
    integer fd, rc, n, i, row=0, received=0, outputs=0, cycles=0;
    integer expected_rows, expected_outputs;
    reg [31:0] rng=32'h261114ab;
    reg stalled=0, held_last;
    reg signed [7:0] held_data;
    function [31:0] next_rng(input [31:0] x);
        next_rng={x[30:0],x[31]^x[21]^x[1]^x[0]};
    endfunction

    always @(posedge clk) begin
        cycles=cycles+1;
        if(cycles>20000000) $fatal(1,"Timeout");
        if(!reset) begin
            if(stalled && (!out_valid || out_data !== held_data || out_last !== held_last))
                $fatal(1,"Output changed while stalled");
            if(out_valid && in_ready) $fatal(1,"Input ready while output row active");
            if(out_last && !out_valid) $fatal(1,"out_last without out_valid");
            if(dut.state==3 && dut.root_step==24 && dut.next_root==0)
                $fatal(1,"Zero denominator");
        end
        stalled <= !reset && out_valid && !out_ready;
        held_data <= out_data; held_last <= out_last;
    end

    task send_value(input integer value, input integer gamma_value,
                    input integer beta_value, input integer length_value);
        begin
            @(negedge clk);
            rng=next_rng(rng);
            repeat(rng[1:0]) @(negedge clk);
            in_data=value; in_gamma=gamma_value; in_beta=beta_value;
            row_length=length_value; in_valid=1;
            @(posedge clk);
            while(!in_ready) @(posedge clk);
            @(negedge clk); in_valid=0;
        end
    endtask
    task do_reset;
        begin
            @(negedge clk); reset=1; in_valid=0; out_ready=0;
            repeat(2) @(posedge clk);
            @(negedge clk); reset=0;
        end
    endtask

    initial begin
        do_reset();
        @(negedge clk); row_length=0; in_valid=1;
        @(posedge clk); if(in_ready) $fatal(1,"Accepted zero length");
        @(negedge clk); row_length=577;
        @(posedge clk); if(in_ready) $fatal(1,"Accepted length 577");
        @(negedge clk); row_length=1023;
        @(posedge clk); if(in_ready) $fatal(1,"Accepted length 1023");
        @(negedge clk); in_valid=0;
        // Reset partial reception, active sqrt, active division, and stalled output.
        send_value(-128,96,0,576); send_value(127,-128,127,0); do_reset();
        send_value(127,96,0,1);
        wait(dut.state==3 && dut.root_step==10); do_reset();
        send_value(-128,96,0,1);
        wait(dut.state==7 && dut.divide_step==10); do_reset();
        send_value(127,-128,127,1);
        wait(dut.state==13); do_reset();
        send_value(-128,127,-128,1);
        wait(dut.state==15); do_reset();
        send_value(127,96,0,1);
        wait(out_valid);
        repeat(5) @(negedge clk);
        do_reset();
        @(posedge clk); if(out_valid) $fatal(1,"Stale output after reset");

        fd=$fopen("layernorm/example/rows.txt","r");
        if(!fd) $fatal(1,"Missing vectors");
        rc=$fscanf(fd,"%d %d",expected_rows,expected_outputs);
        if(rc!=2 || expected_rows<=0 || expected_outputs<=0) $fatal(1,"Bad vector header");
        while(row<expected_rows) begin
            rc=$fscanf(fd,"%d",n);
            if(rc!=1 || n<1 || n>576) $fatal(1,"Invalid vector length");
            for(i=0;i<n;i=i+1) begin
                rc=$fscanf(fd,"%d %d %d %d",values[i],gammas[i],betas[i],expected[i]);
                if(rc!=4) $fatal(1,"Truncated vector");
            end
            // Change row_length after first beat: the core must use its latch.
            for(i=0;i<n;i=i+1) send_value(values[i],gammas[i],betas[i],i==0?n:0);
            received=0;
            while(received<n) begin
                @(negedge clk);
                rng=next_rng(rng);
                out_ready=(rng[2:0]!=0 && rng[4:3]!=0);
                // Busy input pressure with an invalid new length must not enter.
                in_valid=1; row_length=0; in_data=-128; in_gamma=127; in_beta=-128;
                if(in_ready) $fatal(1,"Ready while busy");
                @(posedge clk);
                if(out_valid && out_ready) begin
                    if($signed(out_data) !== expected[received])
                        $fatal(1,"Mismatch row %0d n=%0d index=%0d got=%0d expected=%0d",
                               row,n,received,$signed(out_data),expected[received]);
                    if(out_last !== (received==n-1)) $fatal(1,"Incorrect out_last");
                    received=received+1; outputs=outputs+1;
                end
            end
            @(negedge clk); out_ready=0; in_valid=0;
            row=row+1;
            if(row%128==0) $display("Checked %0d rows / %0d outputs",row,outputs);
        end
        rc=$fscanf(fd,"%d",n);
        if(rc==1) $fatal(1,"Unexpected extra vector data");
        $fclose(fd);
        if(outputs!=expected_outputs) $fatal(1,"Wrong output count");
        repeat(5) begin
            @(posedge clk);
            if(out_valid) $fatal(1,"Unexpected extra output");
        end
        passed=1;
        $display("PASS: %0d rows, %0d bit-exact INT8 outputs, stalls/bubbles/invalid lengths/reset recovery; %0d cycles",row,outputs,cycles);
        $finish;
    end
endmodule
