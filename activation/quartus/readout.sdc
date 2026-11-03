create_clock -name readout_clk -period 11.111 [get_ports clock]
derive_clock_uncertainty
set data_inputs [get_ports {reset_n config_write config_address[*] config_byteenable[*] columns[*] config_data[*] raw_data[*] read_word_address[*]}]
set_input_delay -clock readout_clk -max 0.0 $data_inputs
set_input_delay -clock readout_clk -min 0.0 $data_inputs
set_output_delay -clock readout_clk -max 0.0 [get_ports {result[*]}]
set_output_delay -clock readout_clk -min 0.0 [get_ports {result[*]}]
