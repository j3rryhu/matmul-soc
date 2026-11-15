# Run from repository root: vsim -c -do layernorm/rtl/run_modelsim.do
onerror {quit -code 1 -force}
if {![file exists layernorm/rtl/work]} {vlib layernorm/rtl/work}
vlog -sv -work layernorm/rtl/work layernorm/rtl/layernorm_row.sv layernorm/rtl/tb_layernorm_row.sv layernorm/rtl/tb_layernorm_directed.sv
vsim -onfinish stop -lib layernorm/rtl/work tb_layernorm_directed
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_layernorm_directed/passed] != 1} {quit -code 1 -force}
quit -sim
if {[info exists 1] && $1 eq "directed"} {quit -code 0 -force}
vsim -onfinish stop -lib layernorm/rtl/work tb_layernorm_row
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_layernorm_row/passed] != 1} {quit -code 1 -force}
quit -code 0 -force
