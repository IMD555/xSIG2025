#!/bin/bash
echo Warming up
./power_matmul_i8.o "NPU" "4096" "300" && sleep 5
./power_matmul_i8.o "GPU" "4096" "300" && sleep 5
./power_matmul_i8.o "CPU" "4096" "300" && sleep 5

echo NPU test
./power_matmul_by_mul-RS_i32.o "NPU" "512" "300" && sleep 5

echo GPU test
./power_matmul_by_mul-RS_i32.o "GPU" "512" "300" && sleep 5

echo CPU test
./power_matmul_by_mul-RS_i32.o "CPU" "512" "300" && sleep 5

echo NPU test
./power_matmul_i8.o "NPU" "4096" "300" && sleep 5

echo GPU test
./power_matmul_i8.o "GPU" "4096" "300" && sleep 5

echo CPU test
./power_matmul_i8.o "CPU" "4096" "300" && sleep 5

