# Run from the repository root with quartus_sta -t activation/rtl/check_project_timing.tcl.
# This checks the existing fit; rebuild before using it to sign off changed RTL/IP.
project_open adder_prototype -revision adder_prototype
create_timing_netlist
read_sdc
update_timing_netlist
report_clocks -file output_files/clock_constraint_check.rpt
check_timing -file output_files/constraint_coverage_check.rpt
report_timing -setup -npaths 10 -detail full_path -file output_files/setup_path_check.rpt
delete_timing_netlist
project_close
