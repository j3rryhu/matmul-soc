`timescale 1ns/1ps
module tb_matmul_gelu_readout;
    reg clock=0; always #5 clock=~clock;
    reg reset_n=0, config_write=0;
    reg [15:0] config_address=0, columns=32;
    reg [127:0] config_data=0, raw_data=0;
    reg [15:0] config_byteenable=0;
    reg [6:0] read_word_address=0;
    wire [127:0] result;
    reg [7:0] rom[0:255];
    reg passed=0;
    integer r, m, lane, v, s, q;
    integer cols, word_addr, row, bias_value, checked_words;
    matmul_gelu_readout dut(.*);
    initial $readmemh("activation/example/gelu_int8.hex",rom);
    task write_byte;
        input integer address;
        input integer data;
        begin
            @(negedge clock);
            config_write=1; config_address=address & ~15;
            config_byteenable=1 << (address%16);
            config_data=0; config_data[(address%16)*8 +: 8]=data;
            @(negedge clock); config_write=0;
        end
    endtask
    // Independent division-based reference; exercise one new address per clock.
    task check_address;
        input integer column_count;
        input integer address;
        begin
            @(negedge clock);
            columns=column_count; read_word_address=address;
            @(posedge clock); #1;
            row=(column_count >= 128 || column_count == 0) ? address/8 :
                ((address%8)*16/column_count)*16 + address/8;
            bias_value=row < 32 ? row-16 : row-128;
            for(lane=0;lane<16;lane=lane+1) begin
                v=((address*16+lane)%256)-128;
                raw_data[lane*8 +: 8]=v;
            end
            #1;
            for(lane=0;lane<16;lane=lane+1) begin
                v=((address*16+lane)%256)-128;
                s=v+bias_value; q=s>127?127:(s< -128?-128:s);
                if(column_count == 0 || column_count%16 != 0) begin
                    if(result[lane*8 +: 8] !== raw_data[lane*8 +: 8])
                        $fatal(1,"Invalid geometry bypass: columns %0d",column_count);
                end else if(result[lane*8 +: 8] !== rom[q & 255])
                    $fatal(1,"Mapping: columns %0d address %0d lane %0d row %0d",column_count,address,lane,row);
            end
            checked_words=checked_words+1;
        end
    endtask
    initial begin
        @(negedge clock); reset_n=1;
        raw_data={16{8'h78}}; #1;
        if(result !== raw_data) $fatal(1,"Bypass failed");
        for(r=0;r<32;r=r+1) write_byte('h8000+r,r-16);
        for(r=32;r<256;r=r+1) write_byte('h8000+r,r-128);
        write_byte('h8100,1);
        for(r=0;r<32;r=r+1)
            for(m=0;m<32;m=m+16) begin
                @(negedge clock);
                read_word_address=((r%16)*128+(r/16)*32+m)/16;
                @(posedge clock);
                // Mock the synchronous output RAM data for this address.
                #1;
                for(lane=0;lane<16;lane=lane+1) begin
                    v=((r*32+m+lane)%256)-128;
                    raw_data[lane*8 +: 8]=v;
                end
                #1;
                for(lane=0;lane<16;lane=lane+1) begin
                    v=((r*32+m+lane)%256)-128;
                    s=v+r-16; q=s>127?127:(s< -128?-128:s);
                    if(result[lane*8 +: 8] !== rom[q & 255])
                        $fatal(1,"Mismatch row %0d col %0d",r,m+lane);
                end
            end
        checked_words=0;
        // All aligned 16-bit column counts, every bank/word, nonsequential order.
        for(cols=0;cols<65536;cols=cols+16)
            for(word_addr=0;word_addr<128;word_addr=word_addr+1)
                check_address(cols,(word_addr*37)%128);
        for(cols=1;cols<128;cols=cols+1)
            if(cols%16 != 0)
                for(word_addr=0;word_addr<128;word_addr=word_addr+1)
                    check_address(cols,word_addr);
        check_address(65535,127);
        write_byte('h8100,0); #1;
        if(result !== raw_data) $fatal(1,"Disable failed");
        write_byte('h8100,1);
        @(negedge clock); reset_n=0;
        @(negedge clock); #1;
        if(result !== raw_data) $fatal(1,"Reset bypass failed");
        passed=1;
        $display("PASS: readout 1024 elements plus %0d mapping words, row biases, byte enables, bypass/reset",checked_words);
        $finish;
    end
    initial begin #10000000; $fatal(1,"Timeout"); end
endmodule
