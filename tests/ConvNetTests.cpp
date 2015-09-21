#define BOOST_TEST_MODULE ML_CONVNET
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <shark/Rng/GlobalRng.h>
#include <shark/Data/DataDistribution.h>
#include "derivativeTestHelper.h"


#include "../src/ConvNet.h"
#include "../src/layers/AbstractLayer.h"
#include "../src/layers/InputLayer.h"
#include "../src/layers/ConvolutionalLayer.h"
#include "../src/layers/FullyConnectedLayer.h"
#include "../src/layers/ActivationLayer.h"
#include "../src/layers/PoolingLayer.h"
#include "../src/layers/SoftmaxLayer.h"

using namespace std;
using namespace boost::archive;
using namespace shark;

struct CudaFixture {
	CudaFixture() {
		// ### Cuda handles
		cudaSetDevice(0);
		// Create a cuda context

		// Reset all cuda-errors:
		cudaGetLastError();
	}
	
	~CudaFixture() {
		cudaDeviceReset();
	}
	
	cudnnStatus_t  status;
	cublasStatus_t cubStatus;
};

//check that the structure is correct, i.e. matrice have the right form and setting parameters works
BOOST_FIXTURE_TEST_SUITE (Models_ConvNet, CudaFixture)

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_medium_1)
{
	//1 layer
	{
		int INPUT_SIZE = 7;
		typedef double PREC;
		InputLayer<PREC> *inLayer = new InputLayer<PREC>(3, INPUT_SIZE, INPUT_SIZE);
		AbstractLayer<PREC> *convLayer0 = new ConvolutionalLayer<PREC>(2, 2, 2);
		// AbstractLayer<PREC> *convLayer1 = new ConvolutionalLayer<PREC>(4, 4, 4);
		AbstractLayer<PREC> *outLayer = new FullyConnectedLayer<PREC>(2);

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(inLayer);
		layers.push_back(convLayer0);
		// layers.push_back(convLayer1);
		layers.push_back(outLayer);

		ConvNet<PREC> convnet;
		convnet.setStructure(layers);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 1);
		testWeightedDerivative(convnet,10);
		testWeightedInputDerivative(convnet,10);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_medium_2)
{
	//1 layer
	{
		int INPUT_SIZE = 7;
		typedef double PREC;
		InputLayer<PREC> *inLayer = new InputLayer<PREC>(3, INPUT_SIZE, INPUT_SIZE);
		AbstractLayer<PREC> *outLayer = new FullyConnectedLayer<PREC>(2);
		AbstractLayer<PREC> *convLayer0 = new ConvolutionalLayer<PREC>(2, 1, 2);
		// AbstractLayer<PREC> *convLayer1 = new ConvolutionalLayer<PREC>(4, 4, 4);

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(inLayer);
		layers.push_back(outLayer);
		layers.push_back(convLayer0);
		// layers.push_back(convLayer1);

		ConvNet<PREC> convnet;
		convnet.setStructure(layers);
		// testWeightedDerivativesSame<ConvNet<PREC> >(convnet);
		testWeightedDerivative(convnet,10);
		testWeightedInputDerivative(convnet,10);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_fully)
{
	//1 layer
	{
		int INPUT_SIZE = 12;
		typedef double PREC;
		InputLayer<PREC> *inLayer = new InputLayer<PREC>(3, INPUT_SIZE, INPUT_SIZE);
		AbstractLayer<PREC> *layer0 = new FullyConnectedLayer<PREC>(2);

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(inLayer);
		layers.push_back(layer0);
		// layers.push_back(layer1);

		ConvNet<PREC> convnet;
		convnet.setStructure(layers);
		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, 1.e-6, 1.e-10);
		testWeightedInputDerivative(convnet,10, 1.e-6, 1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_smax)
{
	//1 layer
	{
		int INPUT_SIZE = 12;
		typedef double PREC;
		InputLayer<PREC> *inLayer = new InputLayer<PREC>(3, INPUT_SIZE, INPUT_SIZE);
		AbstractLayer<PREC> *layer1 = new FullyConnectedLayer<PREC>(3);
		AbstractLayer<PREC> *layer0 = new SoftmaxLayer<PREC>();

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(inLayer);
		layers.push_back(layer0);
		layers.push_back(layer1);

		ConvNet<PREC> convnet;
		convnet.setStructure(layers);
		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10);
		testWeightedInputDerivative(convnet,10);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_deep_fully)
{
	// 20 small fully connected layers interspersed with sigmoid activation-layers.
	{
		int INPUT_SIZE = 8;
		typedef double PREC;
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(3, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new FullyConnectedLayer<PREC>(4));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers);
		testWeightedDerivative(convnet,10, 1.e-6, 1.e-10);
		testWeightedInputDerivative(convnet,10, 1.e-6, 1.e-16);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_deep_conv)
{
	// 15 small convolutional layers interspersed with sigmoid activation-layers.
	// Note "magic": each conv-layer pads exactly the size of the filter. Hence
	// the dimensions of each plane doesn't shrink across layers.
	{
		int INPUT_SIZE = 4;
		typedef double PREC;
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(3, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ConvolutionalLayer<PREC>(4, 2, 2, 1, 1));  // Beware magic sizes...
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(4, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ConvolutionalLayer<PREC>(4, 2, 2, 1, 1));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));



		ConvNet<PREC> convnet;
		convnet.setStructure(layers);
		testWeightedDerivative(convnet, 10, 1.e-6, 1.e-10);
		testWeightedInputDerivative(convnet, 10, 1.e-3, 1.e-11);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_all_layer_types)
{
	//1 layer
	{
		int INPUT_SIZE = 4;
		typedef double PREC;
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(1, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ConvolutionalLayer<PREC>(4, 2, 2));
		layers.push_back(new FullyConnectedLayer<PREC>(8));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new PoolingLayer<PREC>(CUDNN_POOLING_MAX, 1, 4));
		layers.push_back(new ConvolutionalLayer<PREC>(4, 1, 2));
		layers.push_back(new FullyConnectedLayer<PREC>(2));
		layers.push_back(new SoftmaxLayer<PREC>());

		ConvNet<PREC> convnet;
		convnet.setStructure(layers);
		testWeightedDerivative(convnet, 10, 1.e-6, 1.e-10);
		testWeightedInputDerivative(convnet, 10, 1.e-6, 1.e-10);
	}
}


class Problem : public LabeledDataDistribution<RealVector, unsigned int>
{
private:
    double m_noise;
public:
    Problem(double noise):m_noise(noise){}
    void draw(RealVector& input, unsigned int& label)const
    {
        label = Rng::discrete(0, 4);
        input.resize(2);
        input(0) = m_noise * Rng::gauss() + 3.0 * std::cos((double)label);
        input(1) = m_noise * Rng::gauss() + 3.0 * std::sin((double)label);
    }
};

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_all_layer_types_batcheval)
{
	//1 layer
	{
		int INPUT_SIZE = 4;
		typedef double PREC;
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(1, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ConvolutionalLayer<PREC>(4, 2, 2));
		layers.push_back(new FullyConnectedLayer<PREC>(8));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new PoolingLayer<PREC>(CUDNN_POOLING_MAX, 1, 4));
		layers.push_back(new ConvolutionalLayer<PREC>(4, 1, 2));
		layers.push_back(new FullyConnectedLayer<PREC>(2));
		layers.push_back(new SoftmaxLayer<PREC>());

		ConvNet<PREC> convnet;
		convnet.setStructure(layers);

		// Init weights randomly
		Rng::globalRng.seed(time(NULL));
		initRandomUniform(convnet, -0.1, 0.1);

		// Init problem data
		int samples = 1000;
		int sample_len = 16;
		RealMatrix problem;
		problem.resize(samples, sample_len);

		for (int s = 0; s < samples; s++) {
			for (int i = 0; i < sample_len; i++) {
				problem(s,i) = Rng::discrete(0, 4);
			}
		}

		testBatchEval(convnet, problem);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_WeightedDerivatives_mnistsetup)
{
	int INPUT_SIZE = 28;  // MNIST is 28x28 pixels and 1 channel
	typedef double PREC;
	std::vector<AbstractLayer<PREC>*> layers;
	AbstractLayer<PREC> *lay;
	layers.push_back(new InputLayer<PREC>(1, INPUT_SIZE, INPUT_SIZE));
	
	layers.push_back(new ConvolutionalLayer<PREC>(2, 5, 5,  // 5 feat maps, 5x5
																								0, 0,  // pad
																								2, 2));  // stride. res 11 x 11
	layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));

	layers.push_back(new ConvolutionalLayer<PREC>(2, 5, 5,  // 50 feat maps, 5x5
																								1,1,  // pad
																								2, 2));  // stride. res 6x6
	layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));

	layers.push_back(new FullyConnectedLayer<PREC>(100));  // 1x100
	layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));

	layers.push_back(new FullyConnectedLayer<PREC>(10));  // 1x10
	// layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
	layers.push_back(new SoftmaxLayer<PREC>());

	ConvNet<PREC> convnet (0);
	convnet.setStructure(layers, 1024);
	
	// Weight init
	Rng::globalRng.seed(time(NULL));
	// initRandomUniform(convnet, -0.1, 0.1);
	initRandomNormal(convnet, 0.005);
	
	
	testWeightedDerivative(convnet, 1, 1.e-3, 1.e-8);
	testWeightedInputDerivative(convnet, 1, 1.e-3, 1.e-11);
}

BOOST_AUTO_TEST_SUITE_END()
