/*!
 *
 *
 * \brief       Implements a convolutional layer on nvidia gpu's.
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
#ifndef SHARK_MODELS_ACTIVATIONLAYER_H
#define SHARK_MODELS_ACTIVATIONLAYER_H

#include "cudnn.h"
#include "shark_cuda_helpers.h"
#include "AbstractLayer.h"

namespace shark {

//! \brief Provides an Activation layer that can be used anywhere within a ConvNet.
//!
//! An activation layer applies some function h(x) to all input values provided
//! by the previous layer. The functions are predefined and can be one of
//! CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH.
//!
//! For now this layer does not compute the activations or derivatives in place.
template<typename Precision>
class ActivationLayer: public AbstractLayer<Precision> {
public:

	ActivationLayer (cudnnActivationMode_t mode,
	                 std::string layer_name = "ActivationLayer")
		: AbstractLayer<Precision>(layer_name) {
		this->m_act_mode = mode;
	}

	// Using inherited destructor.
	// ~ActivationLayer () {
	// }

	void init(cuHandles handles, cudaStream_t stream) {
		this->m_handle = handles.cudnnHandle;
		this->m_stream = stream;
		cudnnStatus_t status;
		status = cudnnCreateTensorDescriptor(&this->m_outputDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void prepare_gpu() {
		cudnnStatus_t status;

		this->m_channels_out = this->m_channels;
		this->m_height_out = this->m_height;
		this->m_width_out = this->m_width;
		this->m_size_out = this->m_size_in;

		// /* Set and allocate output tensor descriptor */
		cudnnSetTensor4dDescriptor(this->m_outputDesc,
		                           CUDNN_TENSOR_NCHW,  // Data-layout
		                           this->m_cudnn_prec,  // Data-type
		                           this->m_numPatterns,        // Number of input images
		                           this->m_channels_out,       // number of feature maps
		                           this->m_height_out,         // Feature map m_height
		                           this->m_width_out);         // Feature map width
		// Output descriptor end

		if (this->m_numPatterns_old < this->m_numPatterns) {
			if (this->m_numPatterns_old != 0) {
				cudaFree(this->mp_d_outputData);
				cudaFree(this->mp_d_diffDataOut);
				// cudaFree(this->mp_d_weightsDiffData);
			}
			size_t outSize = this->m_numPatterns * this->m_size_out * sizeof(Precision);
			cudaMalloc(&this->mp_d_outputData, outSize);
			cudaMalloc(&this->mp_d_diffDataOut, outSize);
			// cudaMalloc(&this->mp_d_weightsDiffData, this->m_numWeights * sizeof(Precision));
			cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
			this->m_gpu_mem = 2 * outSize; // + 2 * this->m_numWeights * sizeof(Precision);
		}
		// Malloc end
	}

	void forward_gpu() const {
		cudnnStatus_t status;

		// Mode must be one of:
		// CUDNN_ADD_IMAGE = h,w must match. source.(n,c) = 1.
		// CUDNN_ADD_FEATURE_MAP = c,h,w must match. source.n = 1.
		// CUDNN_ADD_FULL_TENSOR = n,c,h,w must match.
		// CUDNN_ADD_SAME_C = c must match. source.(n,h,w) = 1
		cudnnAddMode_t mode = CUDNN_ADD_FULL_TENSOR;

		Precision alpha = 1.0;
		Precision beta = 0.0;
		// status = cudnnAddTensor(this->m_handle,
		//                         mode,
		//                         &alpha,
		//                         this->m_inputDesc, this->mep_d_inputData,
		//                         &beta,
		//                         this->m_outputDesc, this->mp_d_outputData);
		// beta = 1.0;
		// status = cudnnAddTensor(this->m_handle,
		//                         mode,
		//                         &alpha,
		//                         this->m_weightsTens, this->mp_d_weightsData,
		//                         &beta,
		//                         this->m_outputDesc, this->mp_d_outputData);
		// cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		beta = 0.0;
		status = cudnnActivationForward(this->m_handle,
		                                m_act_mode,
		                                &alpha,
		                                this->m_inputDesc, this->mep_d_inputData,
		                                &beta,
		                                this->m_outputDesc, this->mp_d_outputData);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void backward_gpu() const {
		cudnnStatus_t status;

		// Calculate the error for the previous layer
		Precision alpha = 1.0;
		Precision beta = 0.0;
		status = cudnnActivationBackward(this->m_handle,
		                                 this->m_act_mode,
		                                 &alpha,
		                                 this->m_outputDesc, this->mp_d_outputData,
		                                 this->m_outputDesc, this->mp_d_diffDataOut,
		                                 this->m_inputDesc, this->mep_d_inputData,
		                                 &beta,
		                                 this->m_inputDesc, this->mep_d_prevDiffData);  // *gradData
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}


	bool m_setup;

	/* m_act_mode is one of:
		 CUDNN_ACTIVATION_SIGMOID
		 CUDNN_ACTIVATION_RELU
		 CUDNN_ACTIVATION_TANH
	*/
	cudnnActivationMode_t m_act_mode;
};  // end class ActivationLayer

}  // end namespace shark
#endif
