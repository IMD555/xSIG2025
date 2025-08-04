cmake_minimum_required(VERSION 3.10)
project(openvino)

find_package(OpenVINO REQUIRED)

add_executable(matmul_i8.o matmul_i8.cpp)
target_link_libraries(matmul_i8.o PRIVATE openvino::runtime)

add_executable(matmul_f16.o matmul_f16.cpp)
target_link_libraries(matmul_f16.o PRIVATE openvino::runtime)

add_executable(matmul_by_mul-RS_i8.o matmul_by_mul-RS_i8.cpp)
target_link_libraries(matmul_by_mul-RS_i8.o PRIVATE openvino::runtime)

add_executable(matmul_by_mul-RS_u8.o matmul_by_mul-RS_u8.cpp)
target_link_libraries(matmul_by_mul-RS_u8.o PRIVATE openvino::runtime)

add_executable(matmul_by_mul-RS_u16.o matmul_by_mul-RS_u16.cpp)
target_link_libraries(matmul_by_mul-RS_u16.o PRIVATE openvino::runtime)

add_executable(matmul_by_mul-RS_u32.o matmul_by_mul-RS_u32.cpp)
target_link_libraries(matmul_by_mul-RS_u32.o PRIVATE openvino::runtime)

add_executable(matmul_by_mul-RS_i32.o matmul_by_mul-RS_i32.cpp)
target_link_libraries(matmul_by_mul-RS_i32.o PRIVATE openvino::runtime)

add_executable(matmul_by_mul-RS_bf16.o matmul_by_mul-RS_bf16.cpp)
target_link_libraries(matmul_by_mul-RS_bf16.o PRIVATE openvino::runtime)

add_executable(bandwidth_matmul_i8.o bandwidth_matmul_i8.cpp)
target_link_libraries(bandwidth_matmul_i8.o PRIVATE openvino::runtime)

add_executable(bandwidth_matmul_f16.o bandwidth_matmul_f16.cpp)
target_link_libraries(bandwidth_matmul_f16.o PRIVATE openvino::runtime)

add_executable(bandwidth_convolution.o bandwidth_convolution.cpp)
target_link_libraries(bandwidth_convolution.o PRIVATE openvino::runtime)

add_executable(bandwidth_reduceSum.o bandwidth_reduceSum.cpp)
target_link_libraries(bandwidth_reduceSum.o PRIVATE openvino::runtime)

add_executable(power_matmul_i8.o power_matmul_i8.cpp)
target_link_libraries(power_matmul_i8.o PRIVATE openvino::runtime)

add_executable(power_matmul_by_mul-RS_i32.o power_matmul_by_mul-RS_i32.cpp)
target_link_libraries(power_matmul_by_mul-RS_i32.o PRIVATE openvino::runtime)
