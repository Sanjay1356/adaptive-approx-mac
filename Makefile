# Icarus Verilog simulation for the adaptive approximate MAC.
#   make            -> run every testbench
#   make <name>     -> run one (mult | fixed_k | mac | policy | system)
IVERILOG ?= iverilog
VVP      ?= vvp
FLAGS    := -g2012
BUILD    := build

MULT   := rtl/approx_multiplier.sv
MAC    := $(MULT) rtl/approx_mac_top.sv
POLICY := rtl/adaptive_policy_controller.sv
SYSTEM := $(MAC) $(POLICY) rtl/adaptive_mac_system.sv

.PHONY: all mult fixed_k mac policy system clean
all: mult fixed_k mac policy system

$(BUILD):
	@mkdir -p $(BUILD)

mult: | $(BUILD)      ## exhaustive 262,144-vector runtime-level multiplier check
	$(IVERILOG) $(FLAGS) -o $(BUILD)/mult $(MULT) tb/tb_approx_mac.sv && $(VVP) -n $(BUILD)/mult
fixed_k: | $(BUILD)   ## exhaustive check of the compile-time FIXED_K = 0/2/4/6 variants
	$(IVERILOG) $(FLAGS) -o $(BUILD)/fixed_k $(MULT) tb/tb_fixed_k.sv && $(VVP) -n $(BUILD)/fixed_k
mac: | $(BUILD)       ## accumulator behaviour
	$(IVERILOG) $(FLAGS) -o $(BUILD)/mac $(MAC) tb/tb_mac_top.sv && $(VVP) -n $(BUILD)/mac
policy: | $(BUILD)    ## 71-check policy-controller testbench
	$(IVERILOG) $(FLAGS) -o $(BUILD)/policy $(POLICY) tb/tb_adaptive_policy_controller.sv && $(VVP) -n $(BUILD)/policy
system: | $(BUILD)    ## controller + MAC integration
	$(IVERILOG) $(FLAGS) -o $(BUILD)/system $(SYSTEM) tb/tb_adaptive_mac_system.sv && $(VVP) -n $(BUILD)/system

clean:
	rm -rf $(BUILD) *.vcd
