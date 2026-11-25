// Cadence Conformal LEC: RTL (golden) vs Genus netlist (revised).
// Replace <GPDK180_LIB> with your liberty path; run from flow/.
read library -liberty <GPDK180_LIB>/slow.lib
read design -golden -sv \
    ../rtl/adaptive_mac_system.sv \
    ../rtl/adaptive_policy_controller.sv \
    ../rtl/approx_mac_top.sv \
    ../rtl/approx_multiplier.sv
read design -revised -verilog adaptive_mac_netlist.v
set system mode lec
add compared points -all
compare
