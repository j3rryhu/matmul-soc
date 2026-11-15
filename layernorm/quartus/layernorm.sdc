# Same approximately 90 MHz target as the current matmul clock request.
create_clock -name layernorm_clk -period 11.111 [get_ports clk]
derive_clock_uncertainty
set_input_delay -clock layernorm_clk -max 0.0 [get_ports {reset in_valid out_ready in_data[*] in_gamma[*] in_beta[*] row_length[*]}]
set_input_delay -clock layernorm_clk -min 0.0 [get_ports {reset in_valid out_ready in_data[*] in_gamma[*] in_beta[*] row_length[*]}]
set_output_delay -clock layernorm_clk -max 0.0 [get_ports {in_ready out_valid out_last out_data[*]}]
set_output_delay -clock layernorm_clk -min 0.0 [get_ports {in_ready out_valid out_last out_data[*]}]
