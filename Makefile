NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++14

.PHONY: all run clean
all: bin/sobel

bin/sobel: src/main.cu src/sobel.cu src/sobel.h
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) src/main.cu src/sobel.cu -o $@

run: bin/sobel
	./bin/sobel data output

clean:
	rm -f bin/sobel
