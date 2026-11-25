# Cadence Innovus RTL-to-GDSII for adaptive_mac_system (GPDK180).
# Set PDK to your GPDK180 install root; run from flow/ after genus_synth.tcl.
set PDK $::env(GPDK180_ROOT)

create_library_set -name LIB_SLOW -timing [list $PDK/dig/lib/slow.lib]
create_library_set -name LIB_FAST -timing [list $PDK/dig/lib/fast.lib]
create_constraint_mode -name CONSTRAINT -sdc_files {adaptive_mac_generated.sdc}
create_rc_corner -name RC_SLOW -cap_table $PDK/RC_Libs/libraries/captable/t018s6mlv.capTbl -T 25
create_rc_corner -name RC_FAST -cap_table $PDK/RC_Libs/libraries/captable/t018s6mlv.capTbl -T 25
create_delay_corner -name DC_SLOW -library_set LIB_SLOW -rc_corner RC_SLOW
create_delay_corner -name DC_FAST -library_set LIB_FAST -rc_corner RC_FAST
create_analysis_view -name AV_SETUP -constraint_mode CONSTRAINT -delay_corner DC_SLOW
create_analysis_view -name AV_HOLD  -constraint_mode CONSTRAINT -delay_corner DC_FAST
set_analysis_view -setup {AV_SETUP} -hold {AV_HOLD}

set init_mmmc_file viewDef.tcl
set init_verilog   adaptive_mac_netlist.v
set init_top_cell  adaptive_mac_system
set init_lef_file  $PDK/dig/lef/all.lef
set init_pwr_net   {VDD}
set init_gnd_net   {VSS}
init_design
setDesignMode -process 180

# Floorplan + power plan
floorPlan -r 1.0 0.7 10 10 10 10
globalNetConnect VDD -type pgpin -pin VDD -all
globalNetConnect VSS -type pgpin -pin VSS -all
addRing -nets {VDD VSS} -type core_rings -follow core \
    -layer {top Metal5 bottom Metal5 left Metal6 right Metal6} \
    -width {top 1.0 bottom 1.0 left 1.0 right 1.0} \
    -spacing {top 1.0 bottom 1.0 left 1.0 right 1.0} \
    -offset {top 2.0 bottom 2.0 left 2.0 right 2.0} -center 1
setAddStripeMode -extend_to_closest_target ring
addStripe -nets {VDD VSS} -layer Metal6 -direction vertical -width 1.0 \
    -spacing 1.0 -set_to_set_distance 10.0 -xleft_offset 5.0 -start_from left

# Place, CTS, route
setAnalysisMode -analysisType onChipVariation -cppr both
setPlaceMode -placeIOPins 1
placeDesign
ccopt_design
routeDesign

# Sign-off checks, filler, stream-out
# (in the original run these were done from the Innovus session; the stream-out
#  map file and GDS merge list depend on your GPDK180 install)
verify_drc
verifyConnectivity
addFiller
streamOut adaptive_mac_system.gds -mapFile $PDK/streamOut.map -merge [glob -nocomplain $PDK/dig/gds/*.gds]
