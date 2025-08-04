#!/bin/bash
echo Warming up
./matmul_f16.o "NPU" "6144" && sleep 1
# ./matmul_f16.o "GPU" "8192" && sleep 1
# ./matmul_f16.o "CPU" "8192" && sleep 1

echo NPU test
./matmul_f16.o "NPU" "512" && sleep 1 &&
./matmul_f16.o "NPU" "1024" && sleep 1 &&
./matmul_f16.o "NPU" "2048" && sleep 1 &&
./matmul_f16.o "NPU" "4096" && sleep 1 &&
./matmul_f16.o "NPU" "6144" && sleep 1

# echo GPU test
# ./matmul_f16.o "GPU" "512" && sleep 1 &&
# ./matmul_f16.o "GPU" "1024" && sleep 1 &&
# ./matmul_f16.o "GPU" "2048" && sleep 1 &&
# ./matmul_f16.o "GPU" "4096" && sleep 1 &&
# ./matmul_f16.o "GPU" "6144" && sleep 1 &&
# ./matmul_f16.o "GPU" "8192" && sleep 1

# echo CPU test
# ./matmul_f16.o "CPU" "512" && sleep 1 &&
# ./matmul_f16.o "CPU" "1024" && sleep 1 &&
# ./matmul_f16.o "CPU" "2048" && sleep 1 &&
# ./matmul_f16.o "CPU" "4096" && sleep 1 &&
# ./matmul_f16.o "CPU" "6144" && sleep 1 &&
# ./matmul_f16.o "CPU" "8192" && sleep 1

echo NPU test
./matmul_i8.o "NPU" "512" && sleep 1 &&
./matmul_i8.o "NPU" "1024" && sleep 1 &&
./matmul_i8.o "NPU" "2048" && sleep 1 &&
./matmul_i8.o "NPU" "4096" && sleep 1 &&
./matmul_i8.o "NPU" "6144" && sleep 1

# echo GPU test
# ./matmul_i8.o "GPU" "512" && sleep 1 &&
# ./matmul_i8.o "GPU" "1024" && sleep 1 &&
# ./matmul_i8.o "GPU" "2048" && sleep 1 &&
# ./matmul_i8.o "GPU" "4096" && sleep 1 &&
# ./matmul_i8.o "GPU" "6144" && sleep 1 &&
# ./matmul_i8.o "GPU" "8192" && sleep 1

# echo CPU test
# ./matmul_i8.o "CPU" "512" && sleep 1 &&
# ./matmul_i8.o "CPU" "1024" && sleep 1 &&
# ./matmul_i8.o "CPU" "2048" && sleep 1 &&
# ./matmul_i8.o "CPU" "4096" && sleep 1 &&
# ./matmul_i8.o "CPU" "6144" && sleep 1 &&
# ./matmul_i8.o "CPU" "8192" && sleep 1
