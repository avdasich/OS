CXX      = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math
OBLAS    = /opt/homebrew/opt/openblas
INCLUDES = -I. -I$(OBLAS)/include
LIBS     = -L$(OBLAS)/lib -lopenblas -lpthread

all: test_blas

test_blas: test_blas.cpp cblas.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) test_blas.cpp -o test_blas $(LIBS)

clean:
	rm -f test_blas
