# DE1-SoC 50 MHz reference into pll_0
create_clock -name CLOCK_50 -period 20.000 [get_ports clk_clk]

# pll_0_outclk0 and every other PLL output, derived automatically
derive_pll_clocks
derive_clock_uncertainty

# HPS and JTAG are handled by their own IP constraints; the JTAG master's
# clk_0 is the same CLOCK_50 net, already covered above.