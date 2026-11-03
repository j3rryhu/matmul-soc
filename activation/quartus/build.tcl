# quartus_sh -t activation/quartus/build.tcl [build directory] [core RTL]
# Standalone timing benchmark; never assembles or programs a bitstream.
package require ::quartus::project
package require ::quartus::flow
set source_dir [file dirname [file normalize [info script]]]
set build_dir [file join $source_dir build_current]
if {[llength $argv] > 0} {set build_dir [file normalize [lindex $argv 0]]}
set core_file [file join $source_dir .. rtl matmul_gelu_readout.v]
if {[llength $argv] > 1} {set core_file [file normalize [lindex $argv 1]]}
file mkdir $build_dir
file copy -force $core_file [file join $build_dir readout_snapshot.v]
file copy -force [file join $source_dir .. example gelu_int8.hex] [file join $build_dir gelu_int8.hex]
cd $build_dir
project_new readout_benchmark -overwrite
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6
set_global_assignment -name TOP_LEVEL_ENTITY readout_benchmark
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name NUM_PARALLEL_PROCESSORS 2
set_global_assignment -name SEED 1
set_global_assignment -name VERILOG_FILE [file join $build_dir readout_snapshot.v]
set_global_assignment -name SYSTEMVERILOG_FILE [file join $source_dir readout_benchmark.sv]
set_global_assignment -name SDC_FILE [file join $source_dir readout.sdc]
set_global_assignment -name TIMING_ANALYZER_MULTICORNER_ANALYSIS ON
foreach port {reset_n config_write config_address[*] config_byteenable[*] columns[*] config_data[*] raw_data[*] read_word_address[*] result[*]} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $port
}
export_assignments
execute_module -tool map
execute_module -tool fit
execute_module -tool sta
project_close
set summary [file join $build_dir output_files readout_benchmark.sta.summary]
set handle [open $summary r]
set timing [read $handle]
close $handle
set entries [regexp -all -inline {Slack\s*:\s*(-?[0-9.]+)} $timing]
if {[llength $entries] == 0} {error "No timing slack entries found"}
foreach {entry slack} $entries {
    if {$slack < 0} {error "Timing failed: slack $slack ns; see $summary"}
}
