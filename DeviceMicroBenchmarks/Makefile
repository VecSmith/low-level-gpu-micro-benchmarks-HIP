all: device_micro_benchmarks.out

device_micro_benchmarks.out: device_micro_benchmarks.cpp
	hipcc device_micro_benchmarks.cpp -o device_micro_benchmarks.out

clean:
	rm device_micro_benchmarks.out
