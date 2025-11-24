ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cnn_conv
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# Host program
SRCS := $(SRC_DIR)/main_old.cpp

# Device kernels (all compiled into one kernel.elf)
VX_SRCS := \
    $(SRC_DIR)/kernel_conv.cpp \

include ../common.mk

