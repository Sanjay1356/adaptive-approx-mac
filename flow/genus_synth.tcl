# Cadence Genus synthesis of the adaptive MAC (GPDK180).
# Set LIB to your PDK's liberty file. For the fixed-K PPA runs, synthesize
# approx_multiplier with FIXED_K = 0/2/4/6 as separate runs.
set LIB $::env(GPDK180_LIB)   ;# e.g. .../180nm/dig/lib/slow.lib
set_db library $LIB

read_hdl -sv {
    ../rtl/adaptive_policy_controller.sv
    ../rtl/approx_multiplier.sv
    ../rtl/approx_mac_top.sv
    ../rtl/adaptive_mac_system.sv
}
elaborate adaptive_mac_system
read_sdc adaptive_mac.sdc            ;# 10 ns clock (100 MHz)

set_db syn_generic_effort high
set_db syn_map_effort     high
syn_generic
syn_map

report_area   > adaptive_mac_area.rpt
report_power  > adaptive_mac_power.rpt
report_timing > adaptive_mac_timing.rpt
write_netlist > adaptive_mac_netlist.v
write_sdc     > adaptive_mac_generated.sdc
