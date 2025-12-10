ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: cnn

cnn:
	$(MAKE) -C cnn

clean:
	$(MAKE) -C cnn clean

.PHONY: all cnn