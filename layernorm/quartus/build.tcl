# quartus_sh -t layernorm/quartus/build.tcl [optional build directory]
package require ::quartus::project
package require ::quartus::flow
set source_dir [file dirname [file normalize [info script]]]
if {[llength $argv] > 0} {
    set build_dir [file normalize [lindex $argv 0]]
} else {
    set build_dir [file join $source_dir build_timing2]
}
file mkdir $build_dir
file copy -force [file join $source_dir .. example epsilon.hex] [file join $build_dir epsilon.hex]
file copy -force [file join $source_dir .. rtl layernorm_row.sv] [file join $build_dir layernorm_row_snapshot.sv]
cd $build_dir
project_new layernorm_benchmark -overwrite
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6
set_global_assignment -name TOP_LEVEL_ENTITY layernorm_benchmark
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name NUM_PARALLEL_PROCESSORS 2
set_global_assignment -name SEED 1
set_global_assignment -name SYSTEMVERILOG_FILE [file join $build_dir layernorm_row_snapshot.sv]
set_global_assignment -name SYSTEMVERILOG_FILE [file join $source_dir layernorm_benchmark.sv]
set_global_assignment -name SDC_FILE [file join $source_dir layernorm.sdc]
foreach port {reset in_valid out_ready in_data[*] in_gamma[*] in_beta[*] row_length[*] in_ready out_valid out_last out_data[*]} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $port
}
export_assignments
execute_module -tool map
execute_module -tool fit
execute_module -tool sta
project_close
set timing_file [file join $build_dir output_files layernorm_benchmark.sta.summary]
set handle [open $timing_file r]
set timing_text [read $handle]
close $handle
set entries [regexp -all -inline {Slack\s*:\s*(-?[0-9.]+)} $timing_text]
if {[llength $entries] == 0} {error "No timing slack entries found"}
foreach {entry slack} $entries {
    if {$slack < 0} {error "Timing failed: negative slack $slack ns; see $timing_file"}
}
