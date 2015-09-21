/*!
 *
 *
 * \brief       Implements a Convnet on nvidia gpu's
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
#ifndef SHARK_MODELS_INPUTLAYER_H
#define SHARK_MODELS_INPUTLAYER_H

#include "cudnn.h"
#include "shark_cuda_helpers.h"
#include "AbstractLayer.h"

namespace shark {

//! \brief Provides a dummy-layer to specify the size of the input.
template<typename Precision>
class InputLayer : public AbstractLayer<Precision> {
public:
	InputLayer (int channels, int height, int width,
	            std::string layer_name = "InputLayer")
		: AbstractLayer<Precision>(layer_name) {
		// InputLayer (int channels, int m_height, int width) {
		this->m_channels= channels;
		this->m_height = height;
		this->m_width = width;
		this->m_channels_out = channels;
		this->m_height_out = height;
		this->m_width_out = width;
		this->m_size_in = channels * height * width;
		this->m_size_out = this->m_size_in;
	}

	// Using inherited destructor.
	// ~InputLayer () {
	// }

	void init(cuHandles handles, cudaStream_t stream) {
		// No need for the handles.
		this->m_stream = stream;
		cudnnStatus_t status;
		status = cudnnCreateTensorDescriptor(&this->m_outputDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void prepare_gpu() {
		cudnnStatus_t status;

		// Special case of prep_gpu above.
		// No filter-descriptors etc. should be initialized.
		/* Set decriptors */
		// Input descriptor start

		// Tensor-formats describes the order in which data is laid out in mem:
		// CUDNN_TENSOR_NCHW = image, feature map, rows, columns.
		// CUDNN_TENSOR_NHWC = image, rows, columns, features maps.
		status = cudnnSetTensor4dDescriptor(this->m_outputDesc,
		                                    CUDNN_TENSOR_NCHW,  // Data-layout
		                                    this->m_cudnn_prec,  // Data-type
		                                    this->m_numPatterns,        // Number of input images
		                                    this->m_channels,           // number of feature maps
		                                    this->m_height,             // Feature map m_height
		                                    this->m_width);             // Feature map width
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		// printf("Input descriptor created L:%i\n", __LINE__);

		// Allocate input/output
		// Malloc
		this->m_size_out = this->m_size_in;
		if (this->m_numPatterns_old < this->m_numPatterns) {
			if (this->m_numPatterns_old != 0) {
				cudaFree(this->mp_d_outputData);
				cudaFree(this->mp_d_diffDataOut);
			}
			size_t outSize = this->m_numPatterns * this->m_size_out * sizeof(Precision);
			cudaMalloc(&this->mp_d_outputData, outSize);
			cudaMalloc(&this->mp_d_diffDataOut, outSize);
			cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
			// This is an input layer. Nothing should happen.
			this->m_gpu_mem = 2 * outSize;
		}
	}

	void forward_gpu() const {
		return;
	}
	void backward_gpu() const {
		return;
	}

	//! From ISerializable, reads a model from an archive
	void read( InArchive & archive ) {
	}

	//! From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const {
		std::cout << "write inputlayer" << std::endl;
	}
};

}
#endif