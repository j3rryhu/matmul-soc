# DE1-SoC 50 MHz reference into pll_0
create_clock -name c50 -period 20.000 [get_ports {CLOCK_50}]

# Derive PLL clocks from the compiled netlist (currently pll_0 is 90 MHz).
# Recompile after changing the PLL; an older fitted database retains its old rate.
derive_pll_clocks
derive_clock_uncertainty

# HPS and JTAG are handled by their own IP constraints; the JTAG master's
# clk_0 is the same CLOCK_50 net, already covered above. Their SDC files are
# included by adder_prototype/synthesis/adder_prototype.qip.
