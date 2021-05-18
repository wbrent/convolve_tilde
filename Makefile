# Makefile for [convolve~]

# specify a location for Pd if desired
# PDDIR = /home/yourname/somedirectory/pd-0.51-4

lib.name = convolve~

# specify the location and name of the FFTW library
# for Linux and MacOS, use:
ldlibs = -L/usr/local/lib -lfftw3f
# for Windows, use:
# ldlibs = -L/usr/local/lib -lfftw3f-3

# specify the location of FFTW header file
cflags = -Iinclude -I/usr/local/include

$(lib.name).class.sources = ./src/convolve~.c

datafiles = $(lib.name)-help.pd

# provide the path to pd-lib-builder
PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
