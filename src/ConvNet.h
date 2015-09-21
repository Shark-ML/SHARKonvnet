/*!
 *
 *
 * \brief		Implements a convolutional neural network on nvidia GPUs
 *
 *
 *
 * \author		A. Doerge
 * \date		2015
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 *
 *
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_MODELS_CONVNET_H
#define SHARK_MODELS_CONVNET_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Neurons.h>
#include <boost/serialization/vector.hpp>
#include <shark/LinAlg/BLAS/matrix_set.hpp>
#include <shark/Rng/GlobalRng.h>
#include "layers/AbstractLayer.h"
#include "layers/AbstractWeightedLayer.h"
#include "cudnn.h"
#include "cublas_v2.h"

namespace shark {

//! \brief Offers the functions to create and to work with a convolutional neural network.
//!
template<typename Precision>
class ConvNet :public AbstractModel<RealVector,RealVector>
{
	struct InternalState: public State {
		//!	 \brief Not used.
	};


public:

	//! \brief Creates an empty convolutional neural network.
	//!
	//! After the constructor is called, the #setStructure method needs to be
	//! called to define the network topology.
	ConvNet(int device=0)
		:m_numberOfNeurons(0),m_inputNeurons(0),m_outputNeurons(0), m_numParams(0) {
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
		m_features|=HAS_FIRST_INPUT_DERIVATIVE;

		// Cuda Pure
		cudaSetDevice(device);
		cudaStreamCreate(&m_stream);
		// Reset all cuda-errors:
		cudaGetLastError();

		// Create a cudnn context
		cudnnHandle_t cudnnHandle;
		cudnnStatus_t cudnnStatus;
		cudnnStatus = cudnnCreate(&cudnnHandle);
		cuda_status_check(cudnnStatus, __FILE__, __FUNCTION__, __LINE__);
		cudnnSetStream(cudnnHandle, m_stream);
		m_handles.cudnnHandle = cudnnHandle;

		// Create a cublas context
		cublasHandle_t cublasHandle;
		cublasStatus_t cublasStatus;
		cublasStatus = cublasCreate(&cublasHandle);
		cublasSetStream(cublasHandle, m_stream);
		cuda_status_check(cublasStatus, __FILE__, __FUNCTION__, __LINE__);
		m_handles.cublasHandle = cublasHandle;
	}

	~ConvNet() {
		cudnnStatus_t status;
		cublasStatus_t cublasStatus;
		cublasStatus = cublasDestroy(m_handles.cublasHandle);
		cuda_status_check(cublasStatus, __FILE__, __FUNCTION__, __LINE__);
		status = cudnnDestroy(m_handles.cudnnHandle);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	//! \brief From INameable: return the class name.
	std::string name() const
	{
		return "ConvNet";
	}

	//! \brief Number of input neurons.
	std::size_t inputSize()const {
		return m_inputNeurons;
	}

	//! \brief Number of output neurons.
	std::size_t outputSize()const {
		return m_outputNeurons;
	}

	//! \brief Total number of neurons, that is inputs+hidden+outputs.
	std::size_t numberOfNeurons()const {
		return m_numberOfNeurons;
	}

	//! \brief Total number of hidden neurons.
	std::size_t numberOfHiddenNeurons()const {
		return numberOfNeurons() - inputSize() -outputSize();
	}

	//! \brief Returns the matrices for every layer used by eval.
	std::vector<RealVector> const& layerMatrices()const {
		return m_weightMatrix;
	}

	//! \brief Returns the total number of parameters of the network.
	std::size_t numberOfParameters()const {
		return m_numParams;
	}

	//! \brief An estimate on GPU memory to be used with the current batch-size.
	std::size_t gpuMemUsage() const {
		std::size_t mem = 0;
		for (auto layer : m_layers) {
			mem += layer->m_gpu_mem;
		}
		return mem;
	}

	//! \brief Prints usefull info to the stream provided (e.g. std::cout)
	//!
	//! Call after #setStructure has been called to get info on estimated memory usage etc.
	void print_info(std::ostream &out) {
		out << "Model has " << this->numberOfParameters() << " parameters." << std::endl;
		out << "Mem and sizes:" << std::endl;
		for (auto clayer : m_layers) {
			out << "  " << clayer->m_layer_name << ":" << std::endl;
			out << "	" << clayer->m_channels_out << "x" << clayer->m_height_out << "x" << clayer->m_width_out;
			out << " - " << clayer->m_gpu_mem / 1.e6 << "MB" << std::endl;
		}
		out << "Total ConvNet gpu_mem ~ " << this->gpuMemUsage() / 1.e6 << "MB"<< std::endl << std::flush;
	}

	//! \brief Returns the vector of parameters specified by the layers in m_weightedLayers.
	RealVector parameterVector() const {
		RealVector parameters(numberOfParameters());
		Precision *weights = parameters.storage();
		SHARK_CRITICAL_REGION { // No asynch calls to the GPU please!
			for(auto wlayer : m_weightedLayers) {
				wlayer->copy_weights_to_host(weights);
				weights += wlayer->m_numWeights;
			}
		}
		return parameters;
	}

	//! \brief Sets the parameters of the model to those provided.
	void setParameterVector(RealVector const& newParameters) {
		init(newParameters) >> vectorSet(m_weightMatrix);

		int i = 0;
		SHARK_CRITICAL_REGION { // No asynch calls to the GPU please!
			for (auto clayer : m_weightedLayers) {
				clayer->copy_weights_to_gpu(m_weightMatrix[i++].storage());
			}
		}
	}

	boost::shared_ptr<State> createState()const {
		return boost::shared_ptr<State>(new InternalState());
	}

	//! \brief Forward propagation.
	//!
	//! Copies the inputs provided to the input-layer, and fprops through
	//! the network.
	void eval(RealMatrix const& patterns, RealMatrix &outputs,
	          State& state) const {
		SIZE_CHECK(patterns.size2() == m_inputNeurons);
		size_t numPats = patterns.size1();
		outputs.resize(numPats, m_outputNeurons);
		Precision *output = outputs.storage();
		SHARK_CRITICAL_REGION { // No asynch calls to the GPU please!
			//initialize the input layer using the patterns.
			m_layers[0]->prep(numPats);
			m_layers[0]->copy_pattern_to_gpu(numPats, patterns.storage());
			for(auto clayer : m_layers) {
				clayer->prep(numPats);
				clayer->forward_gpu();
			}
			AbstractLayer<Precision> *clayer = m_layers[m_layers.size()-1];
			clayer->copy_response_to_host(output);
		}
	}
	using AbstractModel<RealVector,RealVector>::eval;

	//! \brief Backwards propagation.
	//!
	//! Copies the coefficients provided to the last layer, and bprops the
	//! derivs through the network.
	void weightedParameterDerivative(BatchInputType const& patterns,
	                                 BatchOutputType const& coefficients,
	                                 State const& state,
	                                 RealVector& gradient) const {
		SIZE_CHECK(coefficients.size2() == m_outputNeurons);
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		std::size_t numPatterns = patterns.size1();

		RealMatrix outputs;
		boost::shared_ptr<State> s = this->createState();

		gradient.resize(m_numParams);
		gradient.clear();

		SHARK_CRITICAL_REGION { // No asynch calls to the GPU please!
			AbstractLayer<Precision> *clayer = m_layers[m_layers.size()-1];
			clayer->copy_coeff_to_gpu(coefficients.storage());
			for(std::size_t layer_n = m_layers.size()-1; layer_n > 0; layer_n--) {
				clayer = m_layers[layer_n];
				clayer->backward_gpu();
			}
			Precision *weights = gradient.storage();
			Precision *weights_end = weights + m_numParams;
			for(auto wlayer : m_weightedLayers) {
				// Bounds checking:
				if (weights <= weights_end) {
					wlayer->copy_weightgradients_to_host(numPatterns, weights);
					weights += wlayer->m_numWeights;
				} else {
					throw std::logic_error("ConvNet::weightedParameterDerivative(...) Number of weights and gradient size do not match.");
				}
			}
		}
		// TODO: Toy-example-cheap-ass-dropout here. Disabled.
		// RealVector dropOutMask;
		// dropOutMask.resize(m_numParams);
		// for (int i=0; i < m_numParams; i++) {dropOutMask(i) = Rng::coinToss();}
		//
		// gradient *= dropOutMask;
	}

	//! \brief Backwards propagation.
	//!
	//! Copies the coefficients provided to the last layer, and bprops the
	//! derivs through the network.
	void weightedInputDerivative(BatchInputType const& patterns,
	                             BatchOutputType const& coefficients,
	                             State const& state,
	                             BatchInputType& inputDerivative)const {
		SIZE_CHECK(coefficients.size2() == m_outputNeurons);
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		SIZE_CHECK(coefficients.size1() == m_layers[0]->m_numPatterns);
		SIZE_CHECK(patterns.size1()		== m_layers[0]->m_numPatterns);

		RealMatrix outputs;
		inputDerivative.resize(patterns.size1(), m_layers[0]->m_size_out);

		SHARK_CRITICAL_REGION { // No asynch calls to the GPU please!
			AbstractLayer<Precision> *clayer = m_layers[m_layers.size()-1];
			clayer->copy_coeff_to_gpu(coefficients.storage());
			for(std::size_t layer_n = m_layers.size()-1; layer_n > 0; layer_n--) {
				clayer = m_layers[layer_n];
				clayer->backward_gpu();
			}
			m_layers[0]->copy_coeff_to_host(inputDerivative.storage());
		}
	}

	//! \brief Initializes a convnet with the layers provided.
	//!
	//! \param layers contains pointers to layers implementing the virtual class
	//!               AbstractLayer. The order fully specifies the structure of
	//!               the convnet.
	//! \param numPatterns Initial batchsize. If this is set to a low number,
	//!                    the layers will resize automatically. Setting it
	//!                    correct (or bigger) avoids at least on free/allocate
	//!                    cycle.
	void setStructure(std::vector<AbstractLayer<Precision>*> const& layers,
	                  size_t numPatterns = 256) {
		SHARK_CRITICAL_REGION { // No asynch calls to the GPU please!
			m_numParams = 0;
			// Initialize the convolutional layers
			AbstractLayer<Precision> *clayer;
			for (int i = 0; i < layers.size(); i++) {
				clayer = layers[i];
				m_layers.push_back(clayer);
				clayer->init(m_handles, m_stream);
				clayer->prep(numPatterns);

				// TODO: (Semi-done, but ugly) Handle layers with no params such as ActivationLayer.
				if (clayer->m_numWeights != 0) {
					m_numParams += clayer->m_numWeights;
					m_weightMatrix.push_back(RealVector(clayer->m_numWeights));
					m_weightedLayers.push_back((AbstractWeightedLayer<Precision> *) clayer);
				}
				// Skip the last layer. We cannot connect that one forwards.
				if (i < layers.size()-1)
					clayer->connect_forward(layers[i+1]);
			}
			m_inputNeurons = layers[0]->m_size_in;
			m_outputNeurons = layers[layers.size()-1]->m_size_out;
		}
	}

	// TODO: Left hand implementation and shortcuts.
	//! \brief Xavier-Bengio initialization. (Often refered to as just Xavier-inititialization)
	//!
	//! There exists multiple version of this initialization scheme. It originates from:
	//! author = {Glorot, Xavier and Bengio, Yoshua},
	//! booktitle = {Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)},
	//! pages = {249--256},
	//! title = {{Understanding the difficulty of training deep feedforward neural networks}},
	//! url = {http://machinelearning.wustl.edu/mlpapers/paper\_files/AISTATS2010\_GlorotB10.pdf},
	//! volume = {9},
	//! year = {2010}
	//! but the constants used vary.
	void xavier_init() {
		for (auto wlayer : m_weightedLayers) {
			wlayer->xavier_init();
		}
	}

	//! From ISerializable, reads a model from an archive
	void read( InArchive & archive ) {
		archive>>m_numberOfNeurons;
		archive>>m_inputNeurons;
		archive>>m_outputNeurons;
		archive>>m_numParams;
		archive>>m_layers;
		// TODO: Write serialization mechanism. Talk to Oswin.
		// TODO: setStructure(m_layers, ____);
		// DO NOT archive the weighted layers. That would "double archive them" // archive<<m_weightedLayers;
		// Load the current weights from the GPU
		archive>>m_weightMatrix;
	}

	//! From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const {
		archive<<m_numberOfNeurons;
		archive<<m_inputNeurons;
		archive<<m_outputNeurons;
		archive<<m_numParams;
		archive<<m_layers;
		// TODO: Write serialization mechanism. Talk to Oswin.
		// Load the current weights from the GPU
		// std::vector<RealVector> weightMatrix;
		// SHARK_CRITICAL_REGION{  // No asynch calls to the GPU please!
		//	for(auto wlayer : m_weightedLayers){
		//		RealVector parameters(wlayer->m_numWeights);
		//		Precision *weights = parameters.storage();
		//		wlayer->copy_weights_to_host(weights);
		//		weightMatrix.push_back( parameters );
		//	}
		// }
		// archive<<weightMatrix;
	}

	//! \brief Number of all network neurons.
	//!
	//! This is the total number of neurons in the network, i.e.
	//! input, hidden and output neurons.
	std::size_t m_numberOfNeurons;
	std::size_t m_inputNeurons;
	std::size_t m_outputNeurons;

	//! \brief A list of all layers in the convnet.
	//!
	//! The order specifies the network structure.
	std::vector<AbstractLayer<Precision>*>         m_layers;
	//! \brief A list of vectors, where each vector specifies the weights for a layer.
	//!
	//! Usually layers are laid out in memory like:
	//! [w_0, w_1, ..., w_n, b_0, b_1, ..., b_n], where w_x is a weight and b_x
	//! is a bias parameter.
	std::vector<RealVector>                        m_weightMatrix;
	//! \brief A list of layers with parameters.
	std::vector<AbstractWeightedLayer<Precision>*> m_weightedLayers;

	//! \brief The total number of parameters in the convnet.
	std::size_t m_numParams;

	// ### CUDA
	cudaStream_t     m_stream;
	cudnnStatus_t    m_cudnnStatus;
	cublasStatus_t   m_cublasStatus;
	struct cuHandles m_handles;
};

}
#endif
