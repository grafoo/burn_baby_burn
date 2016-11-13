all: build run

build:
	nvcc -o burn_gpu -lcurand -lcublas burn_gpu.cpp

run:
	./burn_gpu 9

clean:
	rm ./burn_gpu
