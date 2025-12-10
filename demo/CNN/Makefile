ROOT_DIR := $(realpath ../..)

include $(ROOT_DIR)/config.mk

VORTEX_RT_PATH ?= $(ROOT_DIR)/runtime
VORTEX_KN_PATH ?= $(ROOT_DIR)/kernel

PROJECT := CNN

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

VORTEX_RT_PATH ?= $(ROOT_DIR)/runtime
VORTEX_KN_PATH ?= $(ROOT_DIR)/kernel

ifeq ($(XLEN),64)
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv64imafdv_zve64d -mabi=lp64d # vector extension
	else
		VX_CFLAGS += -march=rv64imafd -mabi=lp64d
	endif
	STARTUP_ADDR ?= 0x180000000
else
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv32imafv_zve32f -mabi=ilp32f # vector extension
	else
		VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
	endif
	STARTUP_ADDR ?= 0x80000000
endif

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex
LLVM_CFLAGS += -Xclang -target-feature -Xclang +zicond
LLVM_CFLAGS += -mllvm -disable-loop-idiom-all # disable memset/memcpy loop idiom
#LLVM_CFLAGS += -mllvm -vortex-branch-divergence=0
#LLVM_CFLAGS += -mllvm -debug -mllvm -print-after-all
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0/$(RISCV_PREFIX)
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/$(RISCV_PREFIX)/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

VX_CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump
VX_CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

#VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
#VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-g++
#VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
#VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

VX_CFLAGS += -O3 -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_HOME)/kernel/include -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR)
VX_CFLAGS += -DXLEN_$(XLEN)
VX_CFLAGS += -DNDEBUG
VX_CFLAGS += $(CONFIGS)

VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc

VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a
#VX_LIBS += -lgcc

VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortex.a $(VX_LIBS)

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_HOME)/runtime/include -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR)
CXXFLAGS += $(CONFIGS)

LDFLAGS += -L$(VORTEX_RT_PATH) -lvortex


SRC_DIR := $(VORTEX_HOME)/demo/$(PROJECT)
SRCS := \
    $(SRC_DIR)/main.cpp \
	$(SRC_DIR)/cnn_pipeline.cpp \
    $(SRC_DIR)/layers/conv_gpu.cpp \
    $(SRC_DIR)/layers/pool_gpu.cpp \
    $(SRC_DIR)/layers/relu_gpu.cpp \
    $(SRC_DIR)/layers/fc_cpu.cpp

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# SRCS := $(SRC_DIR)/main.cpp

# VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n64

# include ../common.mk

# what we want is to run sim-x 
# run-simx: $(PROJECT)
# 	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

DATA_DIR := $(VORTEX_HOME)/demo/$(PROJECT)

.PHONY: run-simx prepare-data

prepare-data:
	@if [ ! -e images ]; then ln -s $(DATA_DIR)/images .; fi
	@if [ ! -e weights ]; then ln -s $(DATA_DIR)/weights .; fi
	@if [ ! -e layers ]; then ln -s $(DATA_DIR)/layers .; fi

run-simx: prepare-data $(PROJECT)
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)
