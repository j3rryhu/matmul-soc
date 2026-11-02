# The working Platform Designer project requests 90 MHz on pll_0.outclk0,
# which drives matmul_accel_0.clock. This is a standalone core benchmark.
create_clock -name softmax_clk -period 11.111 [get_ports clk]
derive_clock_uncertainty
set_input_delay -clock softmax_clk -max 0.0 [get_ports {reset in_valid out_ready in_score[*] row_length[*]}]
set_input_delay -clock softmax_clk -min 0.0 [get_ports {reset in_valid out_ready in_score[*] row_length[*]}]
set_output_delay -clock softmax_clk -max 0.0 [get_ports {in_ready out_valid out_last out_probability[*]}]
set_output_delay -clock softmax_clk -min 0.0 [get_ports {in_ready out_valid out_last out_probability[*]}]
