# From the repository root, first run python layernorm/rtl/generate_canvas_test.py.
onerror {quit -code 1 -force}
if {![file exists layernorm/rtl/work]} {vlib layernorm/rtl/work}
set component [open matmul_accel_hw.tcl r]
set sources {}
while {[gets $component line] >= 0} {
    if {[regexp {^add_fileset_file \S+ (?:SYSTEM_VERILOG|VERILOG) PATH (\S+)} $line -> source]} {
        lappend sources $source
    }
}
close $component
# These two generated RAMs are brought in by their root-project QIPs.
vlog -sv -work layernorm/rtl/work {*}$sources bias_buffer.v psum_buffer.v layernorm/rtl/tb_layernorm_canvas.sv
vsim -onfinish stop -L altera_mf_ver -L lpm_ver -lib layernorm/rtl/work tb_layernorm_canvas
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_layernorm_canvas/passed] != 1} {quit -code 1 -force}
quit -code 0 -force
