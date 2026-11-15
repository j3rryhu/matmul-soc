# Run from the repository root with ModelSim Intel FPGA Edition.
onerror {quit -code 1 -force}
if {![file exists activation/rtl/work]} {vlib activation/rtl/work}
vmap work activation/rtl/work
vlog -sv matmul_accel/accel_top.sv matmul_accel/input_buffer_32_bank.v matmul_accel/input_dispatch.v matmul_accel/output_buffer_32_bank.v matmul_accel/output_loader.v activation/rtl/matmul_gelu_readout.v matmul_accel/pe.v matmul_accel/pe_array.v matmul_accel/sync_fifo.v matmul_accel/weight_ctrl.v matmul_accel/weight_loader.v input_buffer.v weight_buffer.v output_buffer.v sync_fifo_w8_d32.v matmul_accel/rdl/ctrl_reg.sv/ctrl_rf_rf.sv matmul_accel/weight_buffer_32_bank.v matmul_accel/buf_addr_remap.v activation/rtl/tb_accel_readout_merge.sv
vsim -L altera_mf_ver -L lpm_ver -onfinish stop work.tb_accel_readout_merge
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_accel_readout_merge/passed] != 1} {quit -code 1 -force}
quit -code 0 -force
