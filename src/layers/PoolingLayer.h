/*!
 *
 *
 * \brief       Implements a pooling layer on nvidia gpu's.
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
#ifndef SHARK_MODELS_POOLING_LAYER_H
#define SHARK_MODELS_POOLING_LAYER_H

#include "cudnn.h"
#include "shark_cuda_helpers.h"
#include "AbstractLayer.h"

namespace shark {

//! \brief Provides a Pooling layer.
//!
//! The type of the layer is one of:
//!    CUDNN_POOLING_MAX:
//!         The maximum value inside the pooling window will be used.
//!    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
//!         The values inside the pooling window will be averaged. The number of
//!         padded values will be taken into account when computing the average
//!         value.
//!    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
//!         The values inside the pooling window will be averaged. The number of
//!         padded values will not be taken into account when computing the
//!         average value.
//!
//! The size and stride of the pooling is given as parameters.
//! Utilizes the cuDNN-library.
template<typename Precision>
class PoolingLayer: public AbstractLayer<Precision> {
public:

	PoolingLayer (cudnnPoolingMode_t pool_mode, int feats_h, int feats_w,
	              int pad_h=0, int pad_w=0,
	              int stride_h=1, int stride_w=1,
	              int upscale_h=1, int upscale_w=1,
	              std::string layer_name = "PoolingLayer")
		: AbstractLayer<Precision> (layer_name) {
		this->m_pool_mode = pool_mode;
		this->m_feats_h   = feats_h;
		this->m_feats_w   = feats_w;
		this->m_pad_h     = pad_h;
		this->m_pad_w     = pad_w;
		this->m_stride_h  = stride_h;
		this->m_stride_w  = stride_w;
		this->m_upscale_h = upscale_h;
		this->m_upscale_w = upscale_w;
	}

	~PoolingLayer () {
		// Only free the parts that this layer is responsible for.
		cudnnDestroyPoolingDescriptor(m_poolingDesc);
		// Warn of errors when freeing/destroying.
		cuda_status_check_warn(__FILE__, __FUNCTION__, __LINE__);
	}

	void init(cuHandles handles, cudaStream_t stream) {
		this->m_handle = handles.cudnnHandle;
		this->m_stream = stream;
		cudnnStatus_t status;
		status = cudnnCreateTensorDescriptor(&this->m_outputDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		status = cudnnCreatePoolingDescriptor(&this->m_poolingDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void prepare_gpu() {
		cudnnStatus_t status;

		SIZE_CHECK(this->m_feats_h <= this->m_height + this->m_pad_h);
		SIZE_CHECK(this->m_feats_w <= this->m_width + this->m_pad_w);

		this->m_channels_out = this->m_channels;
		this->m_height_out = ((this->m_pad_h + this->m_height - this->m_feats_h) / this->m_stride_h) + 1;
		this->m_width_out  = ((this->m_pad_w + this->m_width  - this->m_feats_w) / this->m_stride_w) + 1;
		this->m_size_out = this->m_channels* this->m_height_out * this->m_width_out;

		// TODO: Sanity check of out-sizes of all layers.
		SIZE_CHECK(this->m_channels_out > 0 && "Convnet structure invalid. Output dims must be > 0");
		SIZE_CHECK(this->m_height_out > 0 && "Convnet structure invalid. Output dims must be > 0");
		SIZE_CHECK(this->m_width_out > 0 && "Convnet structure invalid. Output dims must be > 0");

		// /* Set and allocate output tensor descriptor */
		cudnnSetTensor4dDescriptor(this->m_outputDesc,
		                           CUDNN_TENSOR_NCHW,  // Data-layout
		                           this->m_cudnn_prec,  // Data-type
		                           this->m_numPatterns,        // Number of input images
		                           this->m_channels_out,       // number of feature maps
		                           this->m_height_out,         // Feature map m_height
		                           this->m_width_out);         // Feature map width
		// Output descriptor end

		status = cudnnSetPooling2dDescriptor(this->m_poolingDesc,
		                                     this->m_pool_mode,
		                                     this->m_feats_h, this->m_feats_w,
		                                     this->m_pad_h, this->m_pad_w,
		                                     this->m_stride_h, this->m_stride_w);

		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
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
		status = cudnnPoolingForward(this->m_handle,
		                             this->m_poolingDesc,
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
		status = cudnnPoolingBackward(this->m_handle,
		                              this->m_poolingDesc,
		                              &alpha,
		                              this->m_outputDesc, this->mp_d_outputData,
		                              this->m_outputDesc, this->mp_d_diffDataOut,
		                              this->m_inputDesc, this->mep_d_inputData,
		                              &beta,
		                              this->m_inputDesc, this->mep_d_prevDiffData);  // *gradData
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void copy_weights_to_gpu(const Precision *weights, cudaStream_t stream) {
		return;
	}
	void copy_weights_to_host(const Precision *weights, cudaStream_t stream) {
		return;
	}

	/* pool_mode is one of:
	   CUDNN_POOLING_MAX
	        The maximum value inside the pooling window will be used.
	   CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
	        The values inside the pooling window will be averaged. The number of
	        padded values will be taken into account when computing the average
	        value.
	   CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
	        The values inside the pooling window will be averaged. The number of
	        padded values will not be taken into account when computing the
	        average value.
	*/
	cudnnPoolingMode_t m_pool_mode;

	// Desciption of the poolings in this layer
	// NOT USED FOR NOW. int m_feats_n;  // Number of feature maps
	int m_feats_h;  // Height of each feature map
	int m_feats_w;  // Width of each feature map
	int m_pad_h;  // Size of zero-padding of input before convolution
	int m_pad_w;
	int m_stride_h; // Striding of the filter over the input during the convolution
	int m_stride_w;
	int m_upscale_h; // Upscaling of input
	int m_upscale_w;

	// cuDNN:
	cudnnPoolingDescriptor_t m_poolingDesc;


};  // end class PoolingLayer

}  // end namespace shark
#endif