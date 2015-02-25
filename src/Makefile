CC=clang++
CCOPT=-O3 -std=c++11
CLIBS=-lboost_program_options
CVLIB=`pkg-config --libs opencv`

all: retarget

retarget: retarget.cpp retarget.hpp util.hpp
	$(CC) $(CCOPT) $(CLIBS) $(CVLIB) $< -o $@

clean:
	rm -f retarget

