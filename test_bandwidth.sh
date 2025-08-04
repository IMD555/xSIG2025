#!/bin/bash
echo Warming up
./bandwidth_matmul_i8.o "CPU" "8388608" && sleep 1
./bandwidth_matmul_i8.o "GPU" "8388608" && sleep 1
./bandwidth_matmul_i8.o "NPU" "8388608" && sleep 1

echo NPU test
./bandwidth_matmul_i8.o "NPU" "1048576" && sleep 1 &&
./bandwidth_matmul_i8.o "NPU" "2097152" && sleep 1 &&
./bandwidth_matmul_i8.o "NPU" "4194304" && sleep 1 &&
./bandwidth_matmul_i8.o "NPU" "8388608" && sleep 1

echo GPU test
./bandwidth_matmul_i8.o "GPU" "1048576" && sleep 1 &&
./bandwidth_matmul_i8.o "GPU" "2097152" && sleep 1 &&
./bandwidth_matmul_i8.o "GPU" "4194304" && sleep 1 &&
./bandwidth_matmul_i8.o "GPU" "8388608" && sleep 1 &&
./bandwidth_matmul_i8.o "GPU" "16777216" && sleep 1 &&
./bandwidth_matmul_i8.o "GPU" "33554432" && sleep 1 &&
./bandwidth_matmul_i8.o "GPU" "67108864" && sleep 1 &&
./bandwidth_matmul_i8.o "GPU" "134217728" && sleep 1

echo CPU test
./bandwidth_matmul_i8.o "CPU" "1048576" && sleep 1 &&
./bandwidth_matmul_i8.o "CPU" "2097152" && sleep 1 &&
./bandwidth_matmul_i8.o "CPU" "4194304" && sleep 1 &&
./bandwidth_matmul_i8.o "CPU" "8388608" && sleep 1 &&
./bandwidth_matmul_i8.o "CPU" "16777216" && sleep 1 &&
./bandwidth_matmul_i8.o "CPU" "33554432" && sleep 1 &&
./bandwidth_matmul_i8.o "CPU" "67108864" && sleep 1 &&
./bandwidth_matmul_i8.o "CPU" "134217728" && sleep 1

echo NPU test
./bandwidth_reduceSum.o "NPU" "256" && sleep 1 &&
./bandwidth_reduceSum.o "NPU" "512" && sleep 1 &&
./bandwidth_reduceSum.o "NPU" "1024" && sleep 1 &&
./bandwidth_reduceSum.o "NPU" "2048" && sleep 1 &&
./bandwidth_reduceSum.o "NPU" "4096" && sleep 1 &&
./bandwidth_reduceSum.o "NPU" "8192" && sleep 1 &&
./bandwidth_reduceSum.o "NPU" "16384" && sleep 1 &&
./bandwidth_reduceSum.o "NPU" "32768" && sleep 1

echo GPU test
./bandwidth_reduceSum.o "GPU" "256" && sleep 1 &&
./bandwidth_reduceSum.o "GPU" "512" && sleep 1 &&
./bandwidth_reduceSum.o "GPU" "1024" && sleep 1 &&
./bandwidth_reduceSum.o "GPU" "2048" && sleep 1 &&
./bandwidth_reduceSum.o "GPU" "4096" && sleep 1 &&
./bandwidth_reduceSum.o "GPU" "8192" && sleep 1 &&
./bandwidth_reduceSum.o "GPU" "16384" && sleep 1 &&
./bandwidth_reduceSum.o "GPU" "32768" && sleep 1

echo CPU test
./bandwidth_reduceSum.o "CPU" "256" && sleep 1 &&
./bandwidth_reduceSum.o "CPU" "512" && sleep 1 &&
./bandwidth_reduceSum.o "CPU" "1024" && sleep 1 &&
./bandwidth_reduceSum.o "CPU" "2048" && sleep 1 &&
./bandwidth_reduceSum.o "CPU" "4096" && sleep 1 &&
./bandwidth_reduceSum.o "CPU" "8192" && sleep 1 &&
./bandwidth_reduceSum.o "CPU" "16384" && sleep 1 &&
./bandwidth_reduceSum.o "CPU" "32768" && sleep 1
