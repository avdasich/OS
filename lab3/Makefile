CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math

OPENBLAS_INC = /opt/homebrew/opt/openblas/include
OPENBLAS_LIB = /opt/homebrew/opt/openblas/lib

INCLUDES = -I. -I$(OPENBLAS_INC)
LIBS = -L$(OPENBLAS_LIB) -lopenblas -lpthread

all: dgemm trsm

dgemm: test_dgemm.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

trsm: test_trsm.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

run_dgemm: dgemm
	./dgemm

run_trsm: trsm
	./trsm

run_all: dgemm trsm
	./dgemm
	./trsm
clean:
	rm -f dgemm trsm