# quartus_sta -t layernorm/quartus/report_paths.tcl [build directory]
package require ::quartus::project
package require ::quartus::sta
set source_dir [file dirname [file normalize [info script]]]
set build_dir [file join $source_dir build_timing2]
if {[llength $argv] > 0} {set build_dir [file normalize [lindex $argv 0]]}
cd $build_dir
project_open layernorm_benchmark
create_timing_netlist -model slow
read_sdc
update_timing_netlist
report_timing -setup -npaths 20 -detail full_path -file [file join $build_dir critical_setup.rpt]
delete_timing_netlist
project_close
