#define BOOST_TEST_MODULE ML_CONVNET_MNIST
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>

#include <shark/Rng/Normal.h>

#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> // loss during training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //loss for test performance
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h> //error function to connect data model and loss
#include <shark/ObjectiveFunctions/NoisyErrorFunction.h> //error function to connect data model and loss
#include <shark/ObjectiveFunctions/Regularizer.h> //L1 and L2 regularisation

#include <shark/Algorithms/StoppingCriteria/MaxIterations.h>

#include <shark/Algorithms/Trainers/OptimizationTrainer.h>

#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/Algorithms/GradientDescent/Rprop.h> //resilient propagation as optimizer

//evaluating probabilities
#include <shark/Models/Softmax.h> //transforms model output into probabilities
#include <shark/Models/ConcatenatedModel.h> //provides operator >> for concatenating models

#include <boost/progress.hpp>

#include "test_helpers.h"
#include "../src/layers/AbstractLayer.h"
#include "../src/layers/InputLayer.h"
#include "../src/layers/ConvolutionalLayer.h"
#include "../src/layers/FullyConnectedLayer.h"
#include "../src/layers/PoolingLayer.h"
#include "../src/layers/ActivationLayer.h"
#include "../src/layers/SoftmaxLayer.h"
#include "../src/layers/DistortLayer.h"
#include "../src/ConvNet.h"
#include "cudnn.h"
#include "cuda_profiler_api.h"
#include "Csv.h"

#include <time.h>

#include <cassert>


using namespace shark;

#define EPSILON 1.e-6
#define EST_EPSILON 1.e-7

int main()
{
	// ### Data
	// Load data, use 70% for training and 30% for testing.
	// The path is hard coded; make sure to invoke the executable
	// from a place where the data file can be found. It is located
	// under [shark]/examples/Supervised/data.
	LabeledData<RealVector, unsigned int> testdata;
	
	size_t batch_size = 2;
	
	// Testdata
	{  // ~ 90 secs on mnist_train.csv
		// ~ 452.38 secs on mnist_train_aug.csv
		// ~18.s on mnist_train_aug.csv shark-release flag: -O3
		boost::progress_timer t;
		importCSV(testdata, "data/mnist/mnist_test_normed.csv", shark::FIRST_COLUMN, ',', '#', 1024);
		splitAtElement(testdata, 0.002 * testdata.numberOfElements());
		std::cout << "Time to load test data from csv: ";
	}

	std::cout << " Testdata elements: " << testdata.numberOfElements()
							<< std::endl;

	// ### Convnet
	int INPUT_SIZE = 28;  // MNIST is 28x28 pixels and 1 channel
	typedef double PREC;
	std::vector<AbstractLayer<PREC>*> layers;
	layers.push_back(new InputLayer<PREC>(1, INPUT_SIZE, INPUT_SIZE));
	AbstractLayer<PREC>* distLayer = new DistortLayer<PREC>(3.0, 40,  // translation, rotation
		 																											5.0, 5);  // elastic trans, gauss-size
	layers.push_back(distLayer);
	
	ConvNet<PREC> convnet (0);
	convnet.setStructure(layers, 1024);
	
	convnet.print_info(std::cout);
	
	// Time for filename
	time_t rawtime;
	struct tm * timeinfo;
	time (&rawtime);
	timeinfo = localtime (&rawtime);
	
	// Filenames
	char fname_distort[80];// = "out/mnist_distort.csv";  // We write to this file
	
	//                                          %m%d%H%M%S = month-day-hour-minute-second
	// strftime (fname_distort, 80, "out/mnist_distort.csv", timeinfo);
	
	std::cout << "writing to "  << fname_distort << std::endl;
	
	for (int i=0; i < 5; i++){
		Data<RealVector> res = convnet(testdata.inputs());
		snprintf (fname_distort, 80, "out/mnist_distort_%i.csv", i);
		exportCSV(res, fname_distort);
		std::cout << "Done. ";
	}
}

// BOOST_AUTO_TEST_SUITE_END()