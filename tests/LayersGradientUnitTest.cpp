#define BOOST_TEST_MODULE ML_CONVNET_LAYERS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <shark/Rng/GlobalRng.h>
#include "derivativeTestHelper.h"
#include <shark/Rng/Normal.h>

#include "../src/ConvNet.h"
#include "../src/layers/AbstractLayer.h"
#include "../src/layers/InputLayer.h"
#include "../src/layers/ActivationLayer.h"
#include "../src/layers/ConvolutionalLayer.h"
#include "../src/layers/FullyConnectedLayer.h"
#include "../src/layers/PoolingLayer.h"
#include "../src/layers/SoftmaxLayer.h"

using namespace boost::archive;
using namespace shark;

struct CudaFixture {
	CudaFixture() {
		// ### Cuda handles
		cudaSetDevice(0);

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
BOOST_FIXTURE_TEST_SUITE(Models_ConvNet_Layers, CudaFixture)

typedef double PREC;
PREC EPSILON = 1.e-9;
PREC EST_EPSILON = 1.e-6;
int CHANNELS = 2;
int INPUT_SIZE = 12;  // MNIST is 28x28 pixels and 1 channel




BOOST_AUTO_TEST_CASE( CONVNET_derivatives_fully_forward)
{
	//1 fully connected layers. 3 different batch sizes (1,2, and 3)
	{

		typedef double PREC;

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(1, 3, 1));
		layers.push_back(new FullyConnectedLayer<PREC>(2));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		RealVector weights = RealVector(convnet.numberOfParameters());
		for (int i=0; i < weights.size(); i++) {
			weights[i] = 1.0 / (i+1);
		}

		for (int i=weights.size() - convnet.outputSize(); i < weights.size(); i++) {
			weights[i] = 0.0;
		}
		PREC bias1 = 100.0;
		PREC bias2 = 1000.0;
		weights[6] = bias1;
		weights[7] = bias2;

		// std::cout << "weights " << weights << std::endl;

		convnet.setParameterVector(weights);

		// 2 patterns: 3x1 in size
		// Pattern = [[0,1,2], [3,4,5]]
		RealMatrix patterns (2, 3);
		double *elem = patterns.storage();
		for (int i=0; i < 6; i++) {
			elem[i] = (double) i;
		}

		// std::cout << "patterns " << patterns << std::endl;

		RealMatrix outputs;
		convnet.eval(patterns, outputs);
		// std::cout << "outputs " << outputs << std::endl;

		BOOST_CHECK_CLOSE(outputs(0,0), bias1 + 1.16666666666666666666666, EPSILON);
		BOOST_CHECK_CLOSE(outputs(0,1), bias2 + 0.53333333333333333333333, EPSILON);
		BOOST_CHECK_CLOSE(outputs(1,0), bias1 + 6.66666666666666666666666, EPSILON);
		BOOST_CHECK_CLOSE(outputs(1,1), bias2 + 2.38333333333333333333333, EPSILON);

		// 3 patterns: 3x1 in size
		// Pattern = [[0,1,2], [3,4,5], [6,7,8]]
		RealMatrix patterns2 (3, 3);
		elem = patterns2.storage();
		for (int i=0; i < patterns2.size1() * patterns2.size2(); i++) {
			elem[i] = (double) i;
		}

		// std::cout << "patterns " << patterns << std::endl;

		convnet.eval(patterns2, outputs);
		// std::cout << "outputs " << outputs << std::endl;
		BOOST_CHECK_CLOSE(outputs(0,0), bias1 + 1.16666666666666666666666, EPSILON);
		BOOST_CHECK_CLOSE(outputs(0,1), bias2 + 0.53333333333333333333333, EPSILON);
		BOOST_CHECK_CLOSE(outputs(1,0), bias1 + 6.66666666666666666666666, EPSILON);
		BOOST_CHECK_CLOSE(outputs(1,1), bias2 + 2.38333333333333333333333, EPSILON);
		BOOST_CHECK_CLOSE(outputs(2,0), bias1 + 12.16666666666666666666666, EPSILON);
		BOOST_CHECK_CLOSE(outputs(2,1), bias2 + 4.23333333333333333333333, EPSILON);


		// 1 patterns: 3x1 in size
		// Pattern = [[0,1,2]]
		RealMatrix patterns3 (1, 3);
		elem = patterns3.storage();
		for (int i=0; i < patterns3.size1() * patterns3.size2(); i++) {
			elem[i] = (double) i;
		}

		// std::cout << "patterns " << patterns << std::endl;

		convnet.eval(patterns3, outputs);
		// std::cout << "outputs " << outputs << std::endl;
		BOOST_CHECK_CLOSE(outputs(0,0), bias1 + 1.16666666666666666666666, EPSILON);
		BOOST_CHECK_CLOSE(outputs(0,1), bias2 + 0.53333333333333333333333, EPSILON);

	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_fully_backward_input)
{
	//1 fully connected layers. 3 different batch sizes (1,2, and 3)
	{

		typedef double PREC;

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(1, 3, 1));
		layers.push_back(new FullyConnectedLayer<PREC>(2));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		RealVector weights = RealVector(convnet.numberOfParameters());
		for (int i=0; i < weights.size(); i++) {
			weights[i] = 1.0 / (i+1);
		}

		// std::cout << "weights " << weights << std::endl;

		convnet.setParameterVector(weights);

		// 2 patterns: 3x1 in size
		// Pattern = [[0,1,2], [3,4,5]]
		RealMatrix patterns (2, 3);
		double *elem = patterns.storage();
		for (int i=0; i < 6; i++) {
			elem[i] = (double) i;
		}

		RealVector gradient (convnet.numberOfParameters());
		RealMatrix outputs (2, convnet.outputSize());

		// 2 patterns x 2 output nodes
		RealMatrix coeffs (2, convnet.outputSize());
		elem = coeffs.storage();
		for (int i=0; i < coeffs.size1() * coeffs.size2(); i++) {
			elem[i] = (double) i / 10.0;
		}
		boost::shared_ptr<State> s = convnet.createState();

		// convnet.eval(patterns, outputs);
		// convnet.weightedParameterDerivative(patterns, coeffs, *s, gradient);
		// std::cout << "weight gradient: " << gradient << std::endl;

		boost::shared_ptr<State> s2 = convnet.createState();
		RealMatrix gradients (2, convnet.inputSize());
		// convnet.weightedInputDerivative(patterns, coeffs, *s2, gradients);

		// std::cout << "input gradients: " << gradients << std::endl;
		//
		// std::cout << "weight gradient: " << gradient << std::endl;

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_fullylayer)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(1, 2, 2));
		layers.push_back(new FullyConnectedLayer<PREC>(2));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 1);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet, 1, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet, 1, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_fullylayer_2)
{
	//2 fully connected layers
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new FullyConnectedLayer<PREC>(3));
		layers.push_back(new FullyConnectedLayer<PREC>(2));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 1);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_activationlayer_sig)
{
	//1 layer
	{
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_activationlayer_sig_2)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_SIGMOID));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_activationlayer_relu)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_RELU));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_activationlayer_relu_2)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_RELU));
		layers.push_back(new ActivationLayer<PREC>(CUDNN_ACTIVATION_RELU));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_poolinglayer)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new PoolingLayer<PREC>(CUDNN_POOLING_MAX, 4, 4));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_poolinglayer_2)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new PoolingLayer<PREC>(CUDNN_POOLING_MAX, 4, 4));
		layers.push_back(new PoolingLayer<PREC>(CUDNN_POOLING_MAX, 4, 4));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_softmaxlayer)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new SoftmaxLayer<PREC>());

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_softmaxlayer_2)
{
	//1 layer
	{

		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new SoftmaxLayer<PREC>());
		layers.push_back(new SoftmaxLayer<PREC>());

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,10, EPSILON, EST_EPSILON);
		testWeightedInputDerivative(convnet,10, EPSILON, EST_EPSILON);
	}
}


BOOST_AUTO_TEST_CASE( CONVNET_derivatives_convlayer)
{
	//1 layer
	{
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(2, 2, 2));
		layers.push_back(new ConvolutionalLayer<PREC>(3, 2, 2));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,1, 1.e-8, 1.e-10);
		testWeightedInputDerivative(convnet,1, 1.e-7, 1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_convlayer_hand2)
{
	//1 layer
	{
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(2, 2, 2));
		layers.push_back(new ConvolutionalLayer<PREC>(3, 2, 2));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		double *elem;
		RealVector weights(convnet.numberOfParameters());
		elem = weights.storage();
		for (int i=0; i < convnet.numberOfParameters(); i++) { elem[i] = 0.5; }

		convnet.setParameterVector(weights);

		// std::cout << "weights" << std::endl << weights << std::endl;

		RealMatrix pattern;
		pattern.resize(2,2*2*2);
		elem = pattern.storage();
		for (int i=0; i < 2*2*2*2; i++) { elem[i] = 0.0; }
		elem[1] = 1.0;
		elem[2*2*2 + 4] = 2.0;
		// std::cout << "pattern" << std::endl << pattern << std::endl;


		RealMatrix output;
		convnet.eval(pattern, output);
		// std::cout << "output" << std::endl << output << std::endl;

		double eps = 0.01;
		elem[1] = elem[1] + eps;

		RealMatrix output2;
		convnet.eval(pattern, output2);
		// std::cout << "output2" << std::endl << output2 << std::endl;
		output -= output2;
		// std::cout << "output - output2" << std::endl << output << std::endl;

		boost::shared_ptr<State> s = convnet.createState();

		RealMatrix derivs_i;
		convnet.weightedInputDerivative(pattern, output, *s, derivs_i);
		// std::cout << "derivs_i" << std::endl << derivs_i << std::endl;

		pattern -= derivs_i;
		// std::cout << "pattern" << std::endl << pattern << std::endl;

		convnet.eval(pattern, output);
		// std::cout << "output" << std::endl << output << std::endl;

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet, 1, 1.e-8, 1.e-10);
		testWeightedInputDerivative(convnet, 1, 1.e-4, 1.e-6);


		// pattern += derivs_i;

		// // Back to original data
		// elem[1] = elem[1] - eps;
		//
		// elem = output.storage();
		// for (int i=0; i < output.size2(); i++) { elem[i] = 0.0; }
		// elem[2] = -1.0;
		//
		// // elem[5] = 6.0;
		//
		// std::cout << "output(coeffs)" << std::endl << output << std::endl;
		//
		// convnet.weightedInputDerivative(pattern, output, *s, derivs_i);
		// std::cout << "derivs_i" << std::endl << derivs_i << std::endl;
		//
		// pattern += derivs_i;
		// std::cout << "pattern" << std::endl << pattern << std::endl;
		//
		// convnet.eval(pattern, output);
		// std::cout << "output" << std::endl << output << std::endl;

		// RealVector derivs_w;
		// convnet.weightedParameterDerivative(pattern, output, *s, derivs_w);
		// std::cout << "derivs_w" << std::endl << derivs_w << std::endl;

	}
}

// BOOST_AUTO_TEST_CASE( CONVNET_derivatives_convlayer_hand3)
// {
// 	//1 layer
// 	{
// 		typedef float PREC1;
// 		int CHAN = 3;
// 		size_t numPatterns = 2;
// 		AbstractLayer<PREC1>* inpu1 = new InputLayer<PREC1>(2, 2, 2);
// 		AbstractWeightedLayer<PREC1>* conv1 = new ConvolutionalLayer<PREC1>(CHAN, 2, 2);
//
// 		inpu1->prep(numPatterns);
// 		inpu1->connect_forward(conv1);
// 		conv1->prep(numPatterns);
//
// 		PREC1 *elem;
// 		FloatVector weights(conv1->m_numWeights);
// 		elem = weights.storage();
// 		for (int i=0; i < conv1->m_numWeights; i++) { elem[i] = 0.5; }
//
// 		// Bias
// 		for (int i=conv1->m_numWeights - conv1->m_channels_out; i < conv1->m_numWeights; i++) { elem[i] = i*11; }
//
// 		// std::cout << "weights" << std::endl << weights << std::endl;
//
// 		conv1->copy_weights_to_gpu(weights.storage(), );
//
// 		FloatMatrix pattern;
// 		pattern.resize(2,2*2*2);
// 		elem = pattern.storage();
// 		for (int i=0; i < 2*2*2*2; i++) { elem[i] = 0.0; }
// 		elem[1] = 1.0;
// 		elem[5] = 3.0;
// 		elem[15] = 2.0;
// 		// std::cout << "pattern" << std::endl << pattern << std::endl;
//
//
// 		FloatMatrix output;
// 		output.resize(numPatterns, conv1->m_size_out);
//
// 		inpu1->copy_pattern_to_gpu(numPatterns, pattern.storage());
// 		inpu1->forward_gpu();
// 		conv1->forward_gpu();
// 		conv1->copy_response_to_host(output.storage());
//
// 		// std::cout << "output" << std::endl << output << std::endl;
// 		//
// 		// double eps = 0.01;
// 		// elem[1] = elem[1] + eps;
// 		//
// 		// RealMatrix output2;
// 		// convnet.eval(pattern, output2);
// 		// std::cout << "output2" << std::endl << output2 << std::endl;
// 		// output -= output2;
// 		// std::cout << "output - output2" << std::endl << output << std::endl;
// 		//
// 		// // Back to original data
// 		// elem[1] = elem[1] - eps;
// 		//
// 		// elem = output.storage();
// 		// for (int i=0; i < output.size2(); i++) { elem[i] = 0.0; }
// 		// elem[2] = -1.0;
// 		//
// 		// // elem[5] = 6.0;
// 		//
// 		// std::cout << "output(coeffs)" << std::endl << output << std::endl;
// 		//
// 		// boost::shared_ptr<State> s = convnet.createState();
// 		//
// 		// RealMatrix derivs_i;
// 		// convnet.weightedInputDerivative(pattern, output, *s, derivs_i);
// 		// std::cout << "derivs_i" << std::endl << derivs_i << std::endl;
// 		//
// 		// pattern += derivs_i;
// 		// std::cout << "pattern" << std::endl << pattern << std::endl;
// 		//
// 		// convnet.eval(pattern, output);
// 		// std::cout << "output" << std::endl << output << std::endl;
//
// 		// RealVector derivs_w;
// 		// convnet.weightedParameterDerivative(pattern, output, *s, derivs_w);
// 		// std::cout << "derivs_w" << std::endl << derivs_w << std::endl;
// 	}
// }

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_convlayer_setget_params)
{
	//1 layer
	{
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ConvolutionalLayer<PREC>(2, 2, 2,
		                       0, 0,
		                       1, 1,
		                       1, 1,
		                       "MiddleConvLayer."));
		layers.push_back(new ConvolutionalLayer<PREC>(2, 2, 2,
		                       0, 0,
		                       1, 1,
		                       1, 1,
		                       "EndConvLayer."));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		RealVector weights (convnet.numberOfParameters());
		double *elem = weights.storage();
		for (int i = 0; i < convnet.numberOfParameters(); i++) {
			elem[i] = (double) i;
		}

		convnet.setParameterVector(weights);

		RealVector outWeights = convnet.parameterVector();

		for (int i = 0; i < convnet.numberOfParameters(); i++) {
			BOOST_CHECK_CLOSE(weights(i), outWeights(i), 1.e-10);
		}
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_convlayer_batch)
{
	//1 layer
	{
		size_t LINPUT_SIZE = 2;
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(1, LINPUT_SIZE, LINPUT_SIZE));
		layers.push_back(new ConvolutionalLayer<PREC>(2, 2, 2,
		                       0, 0,  // padding
		                       1, 1,  // stride
		                       1, 1,  // upscale
		                       "EndConvLayer."));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		RealVector weights = RealVector(convnet.numberOfParameters());
		for (int i=0; i < weights.size(); i++) {
			weights[i] = 1.0 / (i+1);
		}

		// Setting bias'es
		int bias1 = 10.0;
		int bias2 = 100.0;
		weights[convnet.numberOfParameters() - 2] = bias1;
		weights[convnet.numberOfParameters() - 1] = bias2;

		// std::cout << "weights " << weights << std::endl;

		convnet.setParameterVector(weights);


		// 2 patterns: 2x2 in size
		// Pattern1 = [[0,1], [3,4]]
		// Pattern2 = [[5,6], [7,8]]
		RealMatrix patterns (2, LINPUT_SIZE * LINPUT_SIZE);
		double *elem = patterns.storage();
		for (int i=0; i < patterns.size1() * patterns.size2(); i++) {
			elem[i] = (double) i;
		}

		// std::cout << "patterns " << patterns << std::endl;

		RealMatrix outputs;
		convnet.eval(patterns, outputs);
		// std::cout << "outputs " << outputs << std::endl;

		BOOST_CHECK_CLOSE(outputs(0,0), bias1 + 1.91666666667, EPSILON);
		BOOST_CHECK_CLOSE(outputs(0,1), bias2 + 0.827380952381, EPSILON);
		BOOST_CHECK_CLOSE(outputs(1,0), bias1 + 10.25, EPSILON);
		BOOST_CHECK_CLOSE(outputs(1,1), bias2 + 3.36547619048, EPSILON);

		RealMatrix coeffs (2, convnet.outputSize());
		coeffs.clear();
		coeffs(0, 1) = -0.1;
		coeffs(1, 1) = -0.1;

		// std::cout << "coeffs " << coeffs << std::endl;

		boost::shared_ptr<State> s = convnet.createState();

		RealVector gradient;
		convnet.weightedParameterDerivative(patterns, coeffs, *s, gradient);

		// std::cout << "gradient " << gradient << std::endl;

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet, 1, 1.e-3, 1.e-10);
		testWeightedInputDerivative(convnet, 1, 1.e-3, 1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_convlayer_batch2)
{
	//1 layer
	{
		size_t LINPUT_SIZE = 3;
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(1, LINPUT_SIZE, LINPUT_SIZE));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2,
		                       0, 0,  // padding
		                       1, 1,  // stride
		                       1, 1,  // upscale
		                       "MidConvLayer."));
		layers.push_back(new ConvolutionalLayer<PREC>(1, 2, 2,
		                       0, 0,  // padding
		                       1, 1,  // stride
		                       1, 1,  // upscale
		                       "EndConvLayer."));


		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		RealVector weights = RealVector(convnet.numberOfParameters());
		for (int i=0; i < weights.size(); i++) {
			weights(i) = 1.0 / (i+1);
		}

		convnet.setParameterVector(weights);


		// 2 patterns: 2x2 in size
		// Pattern1 = [[0,1], [3,4]]
		// Pattern2 = [[5,6], [7,8]]
		RealMatrix patterns (2, LINPUT_SIZE * LINPUT_SIZE);
		double *elem = patterns.storage();
		for (int i=0; i < patterns.size1() * patterns.size2(); i++) {
			elem[i] = (double) i;
		}

		// std::cout << "patterns " << patterns << std::endl;

		RealMatrix outputs;
		convnet.eval(patterns, outputs);
		// std::cout << "outputs " << outputs << std::endl;
		// std::cout << "weights " << weights << std::endl;

		BOOST_CHECK_CLOSE(outputs(0,0), 3.57800925926, EPSILON);
		BOOST_CHECK_CLOSE(outputs(1,0), 13.8086640212, EPSILON);

		RealMatrix coeffs (2, convnet.outputSize());
		elem = coeffs.storage();
		for (int i=0; i < coeffs.size1() * coeffs.size2(); i++) {
			elem[i] = -0.1;
		}

		boost::shared_ptr<State> s = convnet.createState();

		RealVector gradient;
		convnet.weightedParameterDerivative(patterns, coeffs, *s, gradient);

		// std::cout << "gradient " << gradient << std::endl;

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet, 10, 1.e-6, 1.e-10);
		testWeightedInputDerivative(convnet, 10, 1.e-4, 1.e-6);
	}
}

BOOST_AUTO_TEST_CASE( CONVNET_derivatives_convlayer_2)
{
	//1 layer
	{
		INPUT_SIZE = 4;
		std::vector<AbstractLayer<PREC>*> layers;
		layers.push_back(new InputLayer<PREC>(CHANNELS, INPUT_SIZE, INPUT_SIZE));
		layers.push_back(new ConvolutionalLayer<PREC>(2, 2, 2,
		                       0, 0,
		                       1, 1,
		                       1, 1,
		                       "MiddleConvLayer."));
		layers.push_back(new ConvolutionalLayer<PREC>(2, 2, 2,
		                       0, 0,
		                       1, 1,
		                       1, 1,
		                       "EndConvLayer."));

		ConvNet<PREC> convnet;
		convnet.setStructure(layers, 2);

		testWeightedDerivativesSame<ConvNet<PREC> >(convnet, 10);
		testWeightedDerivative(convnet,1, 1.e-8, 1.e-6);
		testWeightedInputDerivative(convnet,1, 1.e-4, 1.e-6);
	}
}

BOOST_AUTO_TEST_SUITE_END()
