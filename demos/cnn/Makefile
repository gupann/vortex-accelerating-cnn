ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C cnn_conv
	$(MAKE) -C cnn_pool
	$(MAKE) -C cnn_relu

run-simx:
	$(MAKE) -C cnn_conv run-simx
	$(MAKE) -C cnn_pool run-simx
	$(MAKE) -C cnn_relu run-simx

run-rtlsim:
	$(MAKE) -C cnn_conv run-rtlsim
	$(MAKE) -C cnn_pool run-rtlsim
	$(MAKE) -C cnn_relu run-rtlsim

clean:
	$(MAKE) -C cnn_conv clean
	$(MAKE) -C cnn_pool clean
	$(MAKE) -C cnn_relu clean
