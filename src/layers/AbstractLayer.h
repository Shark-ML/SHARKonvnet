/*!
 *
 *
 * \brief Abstract base class for layers intended for a convnet utilizing
 *  nvidia gpu's
 *
 *
 *
 * \author      A. Doerge
 * \date        2015
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 *
 * <BR><HR>
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_MODELS_ABSTRACTLAYER_H
#define SHARK_MODELS_ABSTRACTLAYER_H

#include <string>
#include <stdexcept>
#include "cudnn.h"
#include "cublas_v2.h"

namespace shark {

struct cuHandles {
	cudnnHandle_t  cudnnHandle;
	cublasHandle_t cublasHandle;
};

template<typename Precision>
class AbstractLayer {
public:
	AbstractLayer(std::string layer_name) {
		throw std::logic_error("AbstractLayer::AbstractLayer was not specialized for template arg.");
	}
	//
	// ~AbstractLayer ();


	//!  \brief  Updates internal variables based on the previous layer.
	//!
	//!   Can be called at any time to update internal values relating to the
	//!   inputs of this layer. This includes the number of m_channels, width, and
	//!   m_height of the input and the pointers to it. The sizes are not expected to
	//!   change much between calls, but with this method it is allowed. The
	//!   pointers to the input however are expected to change from time to time.
	void fetch_backwards();

	//!  \brief  Connect this layer with the next.
	//!
	//!  Stores a pointer to this layer on the next.
	//!
	//!  \param  forward   pointer to the next layer.
	void connect_forward(AbstractLayer *forward);

	//!  \brief Copies batch of patterns to the GPU.
	//!
	//!  Automatically creates a network with
	//!  three different layers: An input layer with \em in input neurons,
	//!  an output layer with \em out output neurons and one hidden layer
	//!  with \em hidden neurons, respectively.
	//!
	//!  \param  numPatterns  number of input patterns to copy from *patternData.
	//!  \param  patternData  pointer to host-memory where at least numPatterns
	//!                       are laid out in NCHW format.
	void copy_pattern_to_gpu(std::size_t numPatterns, const Precision *patternData) const;

	//!  \brief Copies delta-values to the GPU for this layer.
	//!
	//!  This is usually only relevant for the last layer in
	//!  a network structure, but we allow any layer to be last by providing this
	//!  method.
	//!
	//!  \param  coeffs  pointer to host-memory. The layer already know how many
	//!                  patterns have been forward propagated.
	void copy_coeff_to_gpu(const Precision *coeffs);

	void copy_coeff_to_host(Precision *coeffs) ;

	//!  \brief Copies the result of forward propagation to host.
	//!
	//!  Copies the result of the latest forward propagation (or garbage if it
	//!  has not been called yet), to the memory location provided as an a
	//!  pointer argument. It is the responsibility of the caller to ensure that
	//!  there are enough space at the memory location.
	//!
	//!  \param  output  pointer to host-memory with enough space for the number
	//!                  of patterns that were evaluated.
	void copy_response_to_host(Precision *output) const;

	//!  \brief Allocates the correct amount of gpu-resources before usage.
	//!
	//!  Wrapper around prepare_gpu().
	//!  \param  numPatterns  the number of patterns to evaluate next time.
	void prep(std::size_t numPatterns);

	// #######################################
	// To be implemented by derived classes: #
	// #######################################

	//!  \brief Called once to allow setup that is not available at
	//!         construction-time.
	//!
	//!  \param  handles  a struct of all handle types. E.g. cudnn-handle,
	//!                   and cublas-handle.
	//!  \param  stream   the cuda-stream to use for all evaluations of this
	//!                   layer.
	virtual void init(cuHandles handles, cudaStream_t stream) = 0;

	//!  \brief Allocates the correct amount of gpu-resources before usage.
	//!
	//!  Should be called through prep() which handles some inital setup.
	//!  Can be called multiple times.
	//!  E.g.:
	//!    layer->prep(16);
	//!    layer->prep(16); // Nothing should happen as the resources are already allocated.
	//!    layer->prep(32); // Old allocations should be released and new ones created.
	virtual void prepare_gpu() = 0;

	//!  \brief Forward evaluation.
	//!
	//!  Forward propagates values from *mep_d_inputData to *mp_d_outputData.
	virtual void forward_gpu() const =0;

	//!  \brief Backward propagation.
	//!
	//!  ATTENTION: Assumes previous evaluation by forward_gpu().
	//!  Backward propagates values. Both with respect to weights and inputs.
	virtual void backward_gpu() const {};

	// User supplied name for layer. Otherwise default (layer-type).
	std::string m_layer_name;

	// Input from the previous layer
	int m_channels;
	int m_height;
	int m_width;
	size_t m_size_in;

	// Output from this layer
	int m_channels_out;
	int m_height_out;
	int m_width_out;
	size_t m_size_out;

	int m_numPatterns;

	// TODO: Ugly hack to test wether layer have weights.
	size_t m_numWeights;

	// TODO: Ugly hack. Setup variable.
	// bool m_setup;

	// TODO: Ugly hack.
	int m_numPatterns_old;

	size_t m_gpu_mem;

	bool m_test_eval;

	// cuDNN:
	/* Never malloc'ed. Pointing to the previous layers or for
	   input-layer, it's data. */
	// Input
	AbstractLayer<Precision>     *mep_prevLayer;
	Precision                    *mep_d_inputData; // prev->mp_d_outputData
	cudnnTensorDescriptor_t      m_inputDesc;
	// The error-data BEFORE this layer
	Precision                    *mep_d_prevDiffData;  // prev->mp_d_diffDataOut

	/* cudaMalloc'ed in prep */
	// Response
	Precision                    *mp_d_outputData;
	cudnnTensorDescriptor_t      m_outputDesc;

	// Bias
	Precision                    *mp_d_biasData;
	Precision                    *mp_d_biasDiffData;
	cudnnTensorDescriptor_t      m_biasDesc;

	// The error-data AFTER this layer
	Precision                    *mp_d_diffDataOut;

	// Precision enum for cudnn routines
	cudnnDataType_t m_cudnn_prec;

	cudnnHandle_t m_handle;

	cudaStream_t m_stream;
};  // End Abstract class def

// Explicit template specialization
template<>
AbstractLayer<float>::AbstractLayer (std::string layer_name);

template<>
AbstractLayer<double>::AbstractLayer (std::string layer_name);

}  // namespace shark

// BOOST_IS_ABSTRACT(shark::AbstractLayer<float>)
// BOOST_IS_ABSTRACT(shark::AbstractLayer<double>)

#endif
