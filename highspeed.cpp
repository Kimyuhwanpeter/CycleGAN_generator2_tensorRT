#include <algorithm> 
#include <chrono> 
#include <cstdlib> 
#include <fstream> 
#include <iostream> 
#include <string> 
#include <sys/stat.h> 
#include <unordered_map> 
#include <cassert> 
#include <vector>
#include <memory>
#include <iterator>

#include "include/NvInfer.h" 
//#include <cuda_runtime.h>
#include "cudaWrapper.h"
#include "include/NvOnnxParser.h"

using namespace nvinfer1;
std::ostream& operator<<(std::ostream& o, const ILogger::Severity severity);

////////////////////////////////////////////////////////////////////////////////////////////////////////
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

template <typename T>
struct Destroy
{
	void operator()(T* t) const
	{
		t->destroy();
	}
};

std::string getBasename(std::string const& path)
{
#ifdef _WIN32
	constexpr char SEPARATOR = '\\';
#else
	constexpr char SEPARATOR = '/';
#endif
	int baseId = path.rfind(SEPARATOR) + 1;
	return path.substr(baseId, path.rfind('.') - baseId);
}

// Returns empty string iff can't read the file
std::string readBuffer(std::string const& path)
{
	std::string buffer;
	std::ifstream stream(path.c_str(), std::ios::binary);

	if (stream)
	{
		stream >> std::noskipws;
		std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), std::back_inserter(buffer));
	}

	return buffer;
}

void writeBuffer(void* buffer, size_t size, std::string const& path)
{
	std::ofstream stream(path.c_str(), std::ios::binary);

	if (stream)
		stream.write(static_cast<char*>(buffer), size);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

ICudaEngine* createCudaEngine(std::string const &onnxModelPath)
{
	std::unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{ nvinfer1::createInferBuilder(gLogger) };
	std::unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{ builder->createNetwork() };
	std::unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{ nvonnxparser::createParser(*network, gLogger) };

	if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
	{
		std::cout << "ERROR: could not parse input engine." << std::endl;
		return nullptr;
	}
	return builder->buildCudaEngine(*network); // build and return TensorRT engine

}

ICudaEngine* getCudaEngine(std::string const & onnxModelPath, int batch_size)
{
	std::string enginePath{ getBasename(onnxModelPath) + ".onnx"};
	ICudaEngine* engine{ nullptr };

	std::string buffer = readBuffer(enginePath);

	if (buffer.size())
	{
		// Try to deserialize engine.
		std::unique_ptr<IRuntime, Destroy<IRuntime>> runtime{ createInferRuntime(gLogger) };
		engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);

	}

	if (!engine)
	{
		// Fallback to creating engine from scratch.
		engine = createCudaEngine(onnxModelPath);

		if (engine)
		{
			std::unique_ptr<IHostMemory, Destroy<IHostMemory>> engine_plan{ engine->serialize() };
			// Try to save engine for future uses.
			writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
		}
	}
	return engine;
}

static int getBindingInputIndex(IExecutionContext* context)
{
	return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

void launchInference(IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batch_size)
{
	int inputId = getBindingInputIndex(context);

	cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
	context->enqueue(batch_size, bindings, stream, nullptr);
	cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

void doInference(IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize)
{
	int ITERATIONS = 10;
	cudawrapper::CudaEvent start;
	cudawrapper::CudaEvent end;
	double totalTime = 0.0;

	for (int i = 0; i < ITERATIONS; ++i)
	{
		float elapsedTime;

		// Measure time it takes to copy input to GPU, run inference and move output back to CPU.
		cudaEventRecord(start, stream);
		launchInference(context, stream, inputTensor, outputTensor, bindings, batchSize);
		cudaEventRecord(end, stream);

		// Wait until the work is finished.
		cudaStreamSynchronize(stream);
		cudaEventElapsedTime(&elapsedTime, start, end);

		totalTime += elapsedTime;
	}

	std::cout << "Inference batch size " << batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << std::endl;
}


int main()
{

	std::unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{ nullptr };
	std::unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{ nullptr };
	std::vector<float> inputTensor;
	std::vector<float> outputTensor;
	std::vector<float> referenceTensor;
	void* bindings[2]{ 0 };
	std::vector<std::string> inputFiles;
	cudawrapper::CudaStream stream;
	

	std::string onnxModelPath = "C:/Users/Yuhwan/Documents/New/A2B_generator.onnx";
	int batch_size = 1;

	// Create Cuda Engine
	std::cout << "!!!!" << std::endl;
	engine.reset(getCudaEngine(onnxModelPath, batch_size));	// 이 부분에서 에러가 발생!
	std::cout << "Did it!!!" << std::endl;
	if (!engine)
	{
		std::cout << "비었다" << std::endl;
		return 1;
	}

	std::cout << "Did it!!!" << std::endl;

	return 0;
}
