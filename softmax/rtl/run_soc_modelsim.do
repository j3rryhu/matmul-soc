# Run from the repository root.
onerror {quit -code 1 -force}
set lib softmax/rtl/work
if {![file exists $lib]} {vlib $lib}
set sources {}
set component [open matmul_accel_hw.tcl r]
while {[gets $component line] >= 0} {
    if {[regexp {^add_fileset_file \S+ (?:SYSTEM_VERILOG|VERILOG) PATH (\S+)$} $line -> source]} {
        lappend sources $source
    }
}
close $component
vlog -sv -work $lib {*}$sources bias_buffer.v psum_buffer.v softmax/rtl/tb_accel_softmax.sv
vsim -L altera_mf_ver -L lpm_ver -onfinish stop -lib $lib tb_accel_softmax
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_accel_softmax/passed] != 1} {quit -code 1 -force}
quit -code 0 -force
