# Run from the repository root: vsim -c -do softmax/rtl/run_modelsim.do
onerror {quit -code 1 -force}
if {![file exists softmax/rtl/work]} {vlib softmax/rtl/work}
vlog -sv -work softmax/rtl/work softmax/rtl/softmax_row.sv softmax/rtl/tb_softmax_row.sv softmax/rtl/tb_softmax_directed.sv
vsim -onfinish stop -lib softmax/rtl/work tb_softmax_directed
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_softmax_directed/passed] != 1} {quit -code 1 -force}
quit -sim
set vector_args {}
if {[info exists env(SOFTMAX_ROW_VECTORS)]} {
    lappend vector_args "+ROW_VECTORS=$env(SOFTMAX_ROW_VECTORS)"
}
vsim -onfinish stop -lib softmax/rtl/work tb_softmax_row {*}$vector_args
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_softmax_row/passed] != 1} {quit -code 1 -force}
quit -code 0 -force
