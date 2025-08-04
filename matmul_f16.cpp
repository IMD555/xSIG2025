#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

#define LOOP 100

int main(int argc, char *argv[])
{
    try
    {
        std::ofstream file("matmul_f16.csv", std::ios::app);

        file << std::endl;
        file << argv[1] << ",";
        file << argv[2] << ",";

        size_t size = std::stoi(argv[2]);

        ov::Core core;

        ov::Shape input_shape_A{size, size};
        ov::Shape input_shape_B{size, size};
        auto input_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape_A);
        auto input_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape_B);
        auto matmul_op = std::make_shared<ov::op::v0::MatMul>(input_A, input_B, false, true);
        auto result = std::make_shared<ov::op::v0::Result>(matmul_op);

        ov::ParameterVector inputs{input_A, input_B};
        ov::ResultVector results{result};
        auto model = std::make_shared<ov::Model>(results, inputs);

        ov::CompiledModel compiled_model = core.compile_model(model, argv[1]);

        auto infer_request = compiled_model.create_infer_request();

        std::cout << infer_request.get_input_tensor(0).get_element_type() << std::endl;
        std::cout << infer_request.get_input_tensor(0).get_shape() << std::endl;

        std::vector<ov::float16> input_data_A(size * size, 1.0f);
        std::vector<ov::float16> input_data_B(size * size, 1.0f);

        input_data_A[size * 0 + 0] = 2.0f;
        // input_data_B[size * 0 + 0] = 1.0f;

        ov::Tensor output_tensor;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point end;
        start = std::chrono::steady_clock::now();

        for (size_t loop = 0; loop < LOOP; ++loop)
        {
            infer_request.set_tensor(input_A, ov::Tensor(ov::element::f16, input_shape_A, input_data_A.data()));
            infer_request.set_tensor(input_B, ov::Tensor(ov::element::f16, input_shape_B, input_data_B.data()));

            infer_request.infer();
            infer_request.wait();

            output_tensor = infer_request.get_output_tensor();
        }

        end = std::chrono::steady_clock::now();

        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
        std::cout << output_tensor.get_element_type() << std::endl;
        std::cout << output_tensor.get_shape() << std::endl;

        file << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << ",";

        const ov::float16 *output_data = output_tensor.data<ov::float16>();

        file << std::setprecision(10);
        std::cout << std::setprecision(10);

        for (size_t j = 0; j < std::min((int)size, 10); ++j)
        {
            for (size_t i = 0; i < std::min((int)size, 10); ++i)
            {
                std::cout << output_data[j * size + i] << " ";
            }
            std::cout << std::endl;
        }

        file.close();
    }
    catch (std::exception e)
    {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}