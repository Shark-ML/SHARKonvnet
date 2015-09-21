/*!
 *
 *
 * \brief Abstract base class for layers intended for a Convnet-model utilizing
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
#ifndef SHARK_MODELS_ABSTRACTWEIGHTEDLAYER_H
#define SHARK_MODELS_ABSTRACTWEIGHTEDLAYER_H

#include "cudnn.h"
#include "curand.h"  // cuda random

#include <shark/Core/Exception.h>

#include "shark_cuda_helpers.h"
#include "AbstractLayer.h"

namespace shark {

template<typename Precision>
class AbstractWeightedLayer : public AbstractLayer<Precision> {
public:


	AbstractWeightedLayer (std::string layer_name)
		: AbstractLayer<Precision> (layer_name) {
	}

	~AbstractWeightedLayer () {
		// Only free the parts that this layer is responsible for.
		cudaFree(mp_d_weightsData);
		cudnnDestroyTensorDescriptor(m_weightsTens);
		cudaFree(mp_d_weightsDiffData);
		// cudnnDestroyTensorDescriptor(weightsGradTens);
		// Warn of errors when freeing/destroying.
		cuda_status_check_warn(__FILE__, __FUNCTION__, __LINE__);
		// TODO: Alloc/free bias?
	}

	virtual void copy_weights_to_gpu(const Precision *weights) {
		cudaMemcpyAsync(mp_d_weightsData, weights, this->m_numWeights * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		cudaStreamSynchronize(this->m_stream);
		cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
	}

	virtual void copy_weights_to_host(Precision *weights) {
		cudaMemcpyAsync(weights, mp_d_weightsData, this->m_numWeights * sizeof(Precision), cudaMemcpyDeviceToHost, this->m_stream);
		cudaStreamSynchronize(this->m_stream);
		cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
	}

	virtual void copy_weightgradients_to_host(size_t numPatterns, Precision *weights) {
		cudaMemcpyAsync(weights, mp_d_weightsDiffData, this->m_numWeights * sizeof(Precision), cudaMemcpyDeviceToHost, this->m_stream);
		cudaStreamSynchronize(this->m_stream);
		cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
	}

	// Forward declaration. Specialized in .cu-file.
	void xavier_init();


	// cuDNN:
	/* cudaMalloc'ed in prep */
	// Forward weights
	Precision                    *mp_d_weightsData;
	cudnnTensorDescriptor_t      m_weightsTens;

	// Backward weights
	Precision                    *mp_d_weightsDiffData;

	size_t m_numBias;
};  // End AbstractWeighted class def

}  // end namespace shark
#endif
