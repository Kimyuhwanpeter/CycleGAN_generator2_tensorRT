#include <algorithm> 
#include <chrono> 
#include <cstdlib> 
#include <cuda_runtime_api.h> 
#include <fstream> 
#include <iostream> 
#include <string> 
#include <sys/stat.h> 
#include <unordered_map> 
#include <cassert> 
#include <vector>
#include <memory>
#include "include/NvInfer.h" 
#include "include/NvUffParser.h" 
#include "include/NvUtils.h"
#include "include/NvOnnxParser.h"
#include "opencv2\core\cuda_types.hpp"
#include "opencv2\core\cuda.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;
using namespace cv;

#define re_width 256
#define re_height 256

class Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) override {
		// remove this 'if' if you need more logged info
		if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
			std::cout << msg << "n";
		}
	}
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
	template< class T >
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

std::vector<cv::cuda::GpuMat> preprocess_img(const std::string& image_path, float *gpu_input, const nvinfer1::Dims& dims)
{
	cv::Mat img = cv::imread(image_path);
	cv::Mat resize_img;
	cv::resize(img, resize_img, cv::Size(re_width, re_height));
	if (img.empty())
		assert(img.empty() == true && " There is no input image");
	else
	{
		cv::cuda::GpuMat img_gpu;
		img_gpu.upload(resize_img);	// upload image to GPU

		auto img_w = dims.d[2];
		auto img_h = dims.d[1];
		auto img_c = dims.d[0];

		cv::cuda::GpuMat flt_image;
		img_gpu.convertTo(flt_image, CV_32FC3, 1.f / 127.5f - 1.f);	// Normalize
		
		std::vector<cv::cuda::GpuMat> chw;

		for (size_t i = 0; i < img_c; i++)
		{
			chw.emplace_back(cv::cuda::GpuMat(cv::Size(img_w, img_h), CV_32FC1, gpu_input + i * img_w * img_h));
		}
		

		return chw;
	}
}

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
	TRTUniquePtr< nvinfer1::IExecutionContext >& context)
{
	TRTUniquePtr< nvinfer1::IBuilder > builder{ nvinfer1::createInferBuilder(gLogger) };
	TRTUniquePtr< nvinfer1::INetworkDefinition > network{ builder->createNetwork() };
	TRTUniquePtr< nvonnxparser::IParser > parser{ nvonnxparser::createParser(*network, gLogger) };
	// parse ONNX
	if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
	{
		std::cerr << "ERROR: could not parse the model.\n";
		return;
	}
	
	TRTUniquePtr< nvinfer1::IBuilderConfig > config{ builder->createBuilderConfig() };
	// allow TensorRT to use up to 1GB of GPU memory for tactic selection.
	config->setMaxWorkspaceSize(1ULL << 30);
	// use FP16 mode if possible
	if (builder->platformHasFastFp16())
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	// we have only one image in batch
	builder->setMaxBatchSize(1);

}
int main()
{
	std::string path = "C:/Users/Yuhwan/Pictures/±Ë¿Ø»Ø.jpg";
	std::string &img_path = path;

	TRTUniquePtr< nvinfer1::ICudaEngine > engine{ nullptr };
	TRTUniquePtr< nvinfer1::IExecutionContext > context{ nullptr };

	std::vector< nvinfer1::Dims > input_dims;
	std::vector< nvinfer1::Dims > output_dims;
	std::vector< void* > buffers(engine->getNbBindings());

	//vector<cv::cuda::GpuMat> chw = preprocess_img(img_path, (float *)buffers[0], input_dim);


	//int iNum = 10;

	//int *p = &iNum;

	//int i = *p;

	//int &ref = iNum;

	//std::cout << i << std::endl;
	//std::cout << ref << std::endl;



	return 0;
}
