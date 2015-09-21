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
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <shark/Core/Exception.h>
#include "cudnn.h"
#include "cublas_v2.h"
#include "shark_cuda_helpers.h"

#include "AbstractLayer.h"

namespace shark {


// ~AbstractLayer () {
// 	// Only free the parts that this layer is responsible for.
// 	cudnnDestroyTensorDescriptor(inputDesc);
// 	cudaFree(mp_d_outputData);
// 	cudnnDestroyTensorDescriptor(m_outputDesc);
// 	cudaFree(mp_d_diffDataOut);
// 	// Warn of errors when freeing/destroying.
// 	cuda_status_check_warn(__FILE__, __FUNCTION__, __LINE__);
// 	// TODO: Alloc/free bias?
// }


// Checks in with the previous layer and updates the internal variables in
// this layer.
template<typename Precision>
void AbstractLayer<Precision>::fetch_backwards() {
	// Do we even have a previous layer??
	if (!mep_prevLayer)
		return;
	m_channels = mep_prevLayer->m_channels_out;
	m_height = mep_prevLayer->m_height_out;
	m_width = mep_prevLayer->m_width_out;
	m_size_in = mep_prevLayer->m_size_out;
	m_inputDesc = mep_prevLayer->m_outputDesc;
	mep_d_inputData = mep_prevLayer->mp_d_outputData;
	mep_d_prevDiffData = mep_prevLayer->mp_d_diffDataOut;
}

template<typename Precision>
void AbstractLayer<Precision>::prep(std::size_t numPatterns_) {
	// If the batch-size did not change, don't do anything. 0 by default.
	if (numPatterns_ == this->m_numPatterns)
		return;
	this->m_numPatterns_old = this->m_numPatterns;
	this->m_numPatterns = numPatterns_;
	this->fetch_backwards();
	this->prepare_gpu();
}

template<typename Precision>
void AbstractLayer<Precision>::connect_forward(AbstractLayer *forward) {
	forward->mep_prevLayer = this;
}

template<typename Precision>
void AbstractLayer<Precision>::copy_pattern_to_gpu(std::size_t numPatterns, const Precision *patternData) const {
	SIZE_CHECK(numPatterns == this->m_numPatterns);
	size_t data_size = numPatterns * m_size_in * sizeof(Precision);
	cudaMemcpyAsync(mp_d_outputData, patternData, data_size, cudaMemcpyHostToDevice, m_stream);
	cudaStreamSynchronize(m_stream);
	cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
}

template<typename Precision>
void AbstractLayer<Precision>::copy_coeff_to_gpu(const Precision *coeffs) {
	size_t outSize = m_numPatterns * m_size_out * sizeof(Precision);
	cudaMemcpyAsync(mp_d_diffDataOut, coeffs, outSize, cudaMemcpyHostToDevice, m_stream);
	cudaStreamSynchronize(m_stream);
	cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
}

template<typename Precision>
void AbstractLayer<Precision>::copy_coeff_to_host(Precision *coeffs) {
	size_t outSize = m_numPatterns * m_size_out * sizeof(Precision);
	cudaMemcpyAsync(coeffs, mp_d_diffDataOut, outSize, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
}

template<typename Precision>
void AbstractLayer<Precision>::copy_response_to_host(Precision *output) const {
	size_t outSize = m_numPatterns * m_size_out * sizeof(Precision);
	cudaMemcpyAsync(output, mp_d_outputData, outSize, cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
}

template<>
AbstractLayer<float>::AbstractLayer (std::string layer_name)
	: m_numPatterns(0), m_numWeights(0), m_layer_name(layer_name),
	  m_cudnn_prec(CUDNN_DATA_FLOAT), mep_prevLayer(NULL) {
};

template<>
AbstractLayer<double>::AbstractLayer (std::string layer_name)
	: m_numPatterns(0), m_numWeights(0), m_layer_name(layer_name),
	  m_cudnn_prec(CUDNN_DATA_DOUBLE), mep_prevLayer(NULL) {
};

template class AbstractLayer<float>;
template class AbstractLayer<double>;

}  // namespace shark
