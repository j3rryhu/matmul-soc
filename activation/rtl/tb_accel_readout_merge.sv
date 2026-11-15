`timescale 1ns/1ps
// Integration regression: real output RAM, remap, Avalon bursts and GELU.
module tb_accel_readout_merge;
    reg clk=0; always #5 clk=~clk;
    reg reset_n=0;
    reg [31:0] avs_address=0;
    reg avs_read=0, avs_write=0;
    reg [127:0] avs_writedata=0;
    reg [15:0] avs_byteenable='1;
    reg [10:0] avs_burstcount=1;
    wire [127:0] avs_readdata;
    wire avs_readdatavalid, avs_waitrequest;
    reg passed=0;
    reg [7:0] lut[0:255];
    reg [191:0] seed_addr;
    reg [127:0] seed_data;
    integer off, bank, row, word_no, lane, raw, biased, expected;
    accel_top dut(.*);
    task write_word(input [31:0] address, input [127:0] data);
        begin
            @(negedge clk); avs_address=address; avs_writedata=data; avs_write=1;
            @(posedge clk); while(avs_waitrequest) @(posedge clk);
            @(negedge clk); avs_write=0;
        end
    endtask
    task read_canvas(input bit activated);
        begin
            @(negedge clk); avs_address='h9000; avs_burstcount=64; avs_read=1;
            @(posedge clk); while(avs_waitrequest) @(posedge clk);
            @(negedge clk); avs_read=0;
            word_no=0;
            while(word_no<64) begin
                @(posedge clk); #1;
                if(avs_readdatavalid) begin
                    row=word_no/2;
                    for(lane=0;lane<16;lane=lane+1) begin
                        raw=(row/16)*32+(word_no%2)*16+lane+row%16;
                        biased=raw+row-16;
                        if(biased>127) biased=127;
                        if(biased< -128) biased=-128;
                        expected=activated ? lut[biased & 255] : raw;
                        if(avs_readdata[lane*8 +: 8] !== expected[7:0])
                            $fatal(1,"canvas activated=%0d row=%0d lane=%0d got=%0h expected=%0h",
                                activated,row,lane,avs_readdata[lane*8 +: 8],expected[7:0]);
                    end
                    word_no=word_no+1;
                end
            end
            repeat(5) begin
                @(posedge clk); #1;
                if(avs_readdatavalid) $fatal(1,"Extra Avalon response beat");
            end
        end
    endtask
    initial begin
        $readmemh("activation/example/gelu_int8.hex",lut);
        repeat(3) @(negedge clk); reset_n=1;
        // Seed two row segments through the real output RAM write ports.
        force dut.obuf_bank_wren=16'hffff;
        force dut.obuf_bank_waddr=seed_addr;
        force dut.obuf_bank_wdata=seed_data;
        for(off=0;off<64;off=off+1) begin
            @(negedge clk);
            for(bank=0;bank<16;bank=bank+1) begin
                seed_addr[bank*12 +: 12]=off;
                seed_data[bank*8 +: 8]=off+bank;
            end
            @(posedge clk); #1;
        end
        @(negedge clk);
        release dut.obuf_bank_wren;
        release dut.obuf_bank_waddr;
        release dut.obuf_bank_wdata;
        write_word('h40,16); // INPUT_COLS differs from full canvas width.
        write_word('h80,32);
        // Old enable address is now IBUF; it must not enable activation.
        write_word('h8100,1);
        if(dut.u_gelu_readout.enabled !== 0) $fatal(1,"IBUF aliases GELU enable");
        for(bank=0;bank<32;bank=bank+16) begin
            seed_data=0;
            for(lane=0;lane<16;lane=lane+1)
                seed_data[lane*8 +: 8]=bank+lane-16;
            write_word('h800+bank,seed_data);
        end
        read_canvas(0);
        write_word('h900,1);
        read_canvas(1);
        write_word('h900,0);
        read_canvas(0);
        passed=1;
        $display("PASS: merged Avalon/remap/GELU: 3072 bytes, 192 beats, no config alias");
        $finish;
    end
    initial begin #100000; $fatal(1,"Integration regression timed out"); end
endmodule
