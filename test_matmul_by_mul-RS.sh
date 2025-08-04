#!/bin/bash
echo Warming up
./matmul_by_mul-RS_i8.o "NPU" "768" && sleep 1
./matmul_by_mul-RS_i8.o "GPU" "1024" && sleep 1
./matmul_by_mul-RS_i8.o "CPU" "1024" && sleep 1

echo NPU test
./matmul_by_mul-RS_i8.o "NPU" "128" && sleep 1 &&
./matmul_by_mul-RS_i8.o "NPU" "256" && sleep 1 &&
./matmul_by_mul-RS_i8.o "NPU" "512" && sleep 1 &&
./matmul_by_mul-RS_i8.o "NPU" "768" && sleep 1

echo GPU test
./matmul_by_mul-RS_i8.o "GPU" "128" && sleep 1 &&
./matmul_by_mul-RS_i8.o "GPU" "256" && sleep 1 &&
./matmul_by_mul-RS_i8.o "GPU" "512" && sleep 1 &&
./matmul_by_mul-RS_i8.o "GPU" "768" && sleep 1 &&
./matmul_by_mul-RS_i8.o "GPU" "1024" && sleep 1

echo CPU test
./matmul_by_mul-RS_i8.o "CPU" "128" && sleep 1 &&
./matmul_by_mul-RS_i8.o "CPU" "256" && sleep 1 &&
./matmul_by_mul-RS_i8.o "CPU" "512" && sleep 1 &&
./matmul_by_mul-RS_i8.o "CPU" "768" && sleep 1 &&
./matmul_by_mul-RS_i8.o "CPU" "1024" && sleep 1

echo NPU test
./matmul_by_mul-RS_i32.o "NPU" "128" && sleep 1 &&
./matmul_by_mul-RS_i32.o "NPU" "256" && sleep 1 &&
./matmul_by_mul-RS_i32.o "NPU" "512" && sleep 1

echo GPU test
./matmul_by_mul-RS_i32.o "GPU" "128" && sleep 1 &&
./matmul_by_mul-RS_i32.o "GPU" "256" && sleep 1 &&
./matmul_by_mul-RS_i32.o "GPU" "512" && sleep 1 &&
./matmul_by_mul-RS_i32.o "GPU" "768" && sleep 1 &&
./matmul_by_mul-RS_i32.o "GPU" "1024" && sleep 1

echo CPU test
./matmul_by_mul-RS_i32.o "CPU" "128" && sleep 1 &&
./matmul_by_mul-RS_i32.o "CPU" "256" && sleep 1 &&
./matmul_by_mul-RS_i32.o "CPU" "512" && sleep 1 &&
./matmul_by_mul-RS_i32.o "CPU" "768" && sleep 1 &&
./matmul_by_mul-RS_i32.o "CPU" "1024" && sleep 1
