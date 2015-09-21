/*!
 *
 *
 * \brief       Implements a softmax layer on nvidia gpu's.
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
#ifndef SHARK_MODELS_SOFTMAX_LAYER_H
#define SHARK_MODELS_SOFTMAX_LAYER_H

#include "cudnn.h"
#include "shark_cuda_helpers.h"
#include "AbstractLayer.h"

namespace shark {

//! \brief     Provides a Softmax layer.
//!
//! Takes as parameters the cuDNN algorithm to use, either fast or accurate,
//! and a mode. The mode is one of:
//!   CUDNN_SOFTMAX_MODE_INSTANCE
//!       The softmax operation is computed per image (N) across the dimensions
//!       C,H,W.
//!   CUDNN_SOFTMAX_MODE_CHANNEL
//!       The softmax operation is computed per spatial location (H,W) per image
//!       (N) across the dimension C.
//!
//! Default parameters are: accurate, and instance-mode. Utilizes the cuDNN-library.
template<typename Precision>
class SoftmaxLayer: public AbstractLayer<Precision> {
public:

	SoftmaxLayer (cudnnSoftmaxAlgorithm_t soft_algo = CUDNN_SOFTMAX_ACCURATE,
	              cudnnSoftmaxMode_t soft_mode = CUDNN_SOFTMAX_MODE_INSTANCE,
	              std::string layer_name = "SoftmaxLayer")
		: AbstractLayer<Precision>(layer_name) {
		this->m_soft_algo = soft_algo;
		this->m_soft_mode = soft_mode;
	}

	// Using inherited destructor.
	// ~SoftmaxLayer () {
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

		// Malloc
		if (this->m_numPatterns_old < this->m_numPatterns) {
			if (this->m_numPatterns_old != 0) {
				cudaFree(this->mp_d_outputData);
				cudaFree(this->mp_d_diffDataOut);
			}
			size_t outSize = this->m_numPatterns * this->m_size_out * sizeof(Precision);
			cudaMalloc(&this->mp_d_outputData, outSize);
			cudaMalloc(&this->mp_d_diffDataOut, outSize);
			cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
			this->m_gpu_mem = 2 * outSize;
		}
		// Malloc end
	}

	void forward_gpu() const {
		cudnnStatus_t status;

		Precision alpha = 1.0;
		Precision beta = 0.0;
		status = cudnnSoftmaxForward(this->m_handle,
		                             m_soft_algo, m_soft_mode,
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
		status = cudnnSoftmaxBackward(this->m_handle,
		                              this->m_soft_algo, this->m_soft_mode,
		                              &alpha,
		                              this->m_outputDesc, this->mp_d_outputData,
		                              this->m_outputDesc, this->mp_d_diffDataOut,
		                              &beta,
		                              this->m_inputDesc, this->mep_d_prevDiffData);  // *gradData
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}
	/* m_soft_algo is one of:
	   CUDNN_SOFTMAX_FAST
	   CUDNN_SOFTMAX_ACCURATE
	*/
	cudnnSoftmaxAlgorithm_t m_soft_algo;

	/* m_soft_mode is one of:
		 CUDNN_SOFTMAX_MODE_INSTANCE
			The softmax operation is computed per image (N) across the dimensions
			C,H,W.
		 CUDNN_SOFTMAX_MODE_CHANNEL
			The softmax operation is computed per spatial location (H,W) per image
			(N) across the dimension C.
	*/
	cudnnSoftmaxMode_t m_soft_mode;
};  // end class SoftmaxLayer

}  // end namespace shark
#endif