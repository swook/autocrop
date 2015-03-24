CC=clang++
CCOPT=-O3 -m64 -std=c++11

CLIBS= -lboost_program_options -lvl # Boost, vlfeat
CLIBS+=`pkg-config --libs opencv`   # OpenCV

all: retarget

retarget: retarget.cpp SLIC.hpp saliency.hpp retarget.hpp util.hpp
	$(CC) $(CCOPT) $(CLIBS) $(CVLIB) $< -o $@

clean:
	rm -f retarget

