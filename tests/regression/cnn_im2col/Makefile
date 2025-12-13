ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cnn_im2col
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# Host program
SRCS := $(SRC_DIR)/main.cpp

# Device kernels (all compiled into one kernel.elf)
VX_SRCS := \
    $(SRC_DIR)/kernel_im2col.cpp \

include ../common.mk
