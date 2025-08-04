#include <openvino/openvino.hpp>
#include <openvino/op/reduce_sum.hpp>
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
        std::ofstream file("bandwidth_reduceSum.csv", std::ios::app);

        file << std::endl;
        file << argv[1] << ",";
        file << argv[2] << ",";

        size_t size = std::stoi(argv[2]);

        ov::Core core;

        ov::Shape input_shape_A{8192, size};

        auto input_A = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, input_shape_A);

        auto reduceSum_op = std::make_shared<ov::op::v1::ReduceSum>(input_A, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
        auto result = std::make_shared<ov::op::v0::Result>(reduceSum_op);

        ov::ParameterVector inputs{input_A};
        ov::ResultVector results{result};
        auto model = std::make_shared<ov::Model>(results, inputs);

        ov::CompiledModel compiled_model = core.compile_model(model, argv[1]);

        auto infer_request = compiled_model.create_infer_request();

        std::cout << infer_request.get_input_tensor(0).get_element_type() << std::endl;
        std::cout << infer_request.get_input_tensor(0).get_shape() << std::endl;
        std::cout << infer_request.get_input_tensor(0).get_byte_size() << std::endl;

        std::vector<int32_t> input_data_A(8192 * size, 1);

        // input_data_A[size * 0 + 0] = 1;
        // input_data_B[size * 0 + 0] = 2;

        ov::Tensor output_tensor;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point end;
        start = std::chrono::steady_clock::now();

        for (size_t loop = 0; loop < LOOP; ++loop)
        {
            infer_request.set_tensor(input_A, ov::Tensor(ov::element::i32, input_shape_A, input_data_A.data()));

            infer_request.infer();
            infer_request.wait();

            output_tensor = infer_request.get_output_tensor();
        }

        end = std::chrono::steady_clock::now();

        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
        std::cout << infer_request.get_output_tensor(0).get_element_type() << std::endl;
        std::cout << infer_request.get_output_tensor(0).get_shape() << std::endl;
        std::cout << infer_request.get_output_tensor(0).get_byte_size() << std::endl;

        file << infer_request.get_input_tensor(0).get_byte_size() << ",";
        file << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << ",";

        const int32_t *output_data = output_tensor.data<int32_t>();

        file << std::setprecision(10);
        std::cout << std::setprecision(10);

        for (size_t i = 0; i < std::min((int)size, 10); ++i)
        {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;

        file.close();
    }
    catch (std::exception e)
    {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}