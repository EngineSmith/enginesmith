NRT ?= /opt/aws/neuron
nrt_gemv: nrt_gemv.cpp
	g++ -O2 -std=c++17 -I$(NRT)/include $< -o $@ -L$(NRT)/lib -lnrt -Wl,-rpath,$(NRT)/lib
nrt_inplace: nrt_inplace.cpp
	g++ -O2 -std=c++17 -I$(NRT)/include $< -o $@ -L$(NRT)/lib -lnrt -Wl,-rpath,$(NRT)/lib
all: nrt_gemv nrt_inplace
