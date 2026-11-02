# Run from the repository root: vsim -c -do softmax/rtl/run_modelsim.do
onerror {quit -code 1 -force}
if {![file exists softmax/rtl/work]} {vlib softmax/rtl/work}
vmap work softmax/rtl/work
vlog -sv softmax/rtl/softmax_row.sv softmax/rtl/tb_softmax_row.sv
vsim -onfinish stop work.tb_softmax_row
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_softmax_row/passed] != 1} {quit -code 1 -force}
quit -code 0 -force
