all: histogram.out

histogram.out: histogram_performance_test.cpp
	hipcc histogram_performance_test.cpp -I/opt/rocm/hip/include/ -o histogram.out

clean:
	rm histogram.out
