#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

#define LOOP 100

std::shared_ptr<ov::Node> create_fake_quantize(const std::shared_ptr<ov::Node> &input, float scale)
{
    auto input_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-scale - 1});
    auto input_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {scale});
    auto output_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-128});
    auto output_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {127});
    return std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 256);
}

int main(int argc, char *argv[])
{
    try
    {
        std::ofstream file("matmul_by_mul-RS_i8.csv", std::ios::app);
        file << std::endl;
        file << argv[1] << ",";
        file << argv[2] << ",";

        size_t size = std::stoi(argv[2]);

        ov::Core core;

        ov::Shape input_shape_A{size, 1, size};
        ov::Shape input_shape_B{1, size, size};

        auto input_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape_A);
        auto input_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape_B);

        float scale_A = 127.0f;
        float scale_B = 127.0f;
        auto quantized_A = create_fake_quantize(input_A, scale_A);
        auto quantized_B = create_fake_quantize(input_B, scale_B);

        auto multiply_op = std::make_shared<ov::op::v1::Multiply>(quantized_A, quantized_B);
        auto reduceSum_op = std::make_shared<ov::op::v1::ReduceSum>(multiply_op, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}), false);
        auto result = std::make_shared<ov::op::v0::Result>(reduceSum_op);

        ov::ParameterVector inputs{input_A, input_B};
        ov::ResultVector results{result};
        auto model = std::make_shared<ov::Model>(results, inputs);

        ov::CompiledModel compiled_model = core.compile_model(model, argv[1]);

        auto infer_request = compiled_model.create_infer_request();

        std::cout << infer_request.get_input_tensor(0).get_element_type() << std::endl;
        std::cout << infer_request.get_input_tensor(0).get_shape() << std::endl;

        std::vector<float> input_data_A(size * size, 1.0f);
        std::vector<float> input_data_B(size * size, 1.0f);

        input_data_A[size * 0 + 0] = 2.0f;
        // input_data_B[size * 0 + 0] = 1.0f;

        ov::Tensor output_tensor;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point end;
        start = std::chrono::steady_clock::now();

        for (size_t loop = 0; loop < LOOP; ++loop)
        {
            infer_request.set_tensor(input_A, ov::Tensor(ov::element::f32, input_shape_A, input_data_A.data()));
            infer_request.set_tensor(input_B, ov::Tensor(ov::element::f32, input_shape_B, input_data_B.data()));

            infer_request.infer();
            infer_request.wait();

            output_tensor = infer_request.get_output_tensor();
        }

        end = std::chrono::steady_clock::now();

        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
        std::cout << output_tensor.get_element_type() << std::endl;
        std::cout << output_tensor.get_shape() << std::endl;

        file << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << ",";

        const float *output_data = output_tensor.data<float>();

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