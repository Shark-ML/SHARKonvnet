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
#include <cstdlib>
#include <ctime>

#include <cuda_runtime.h>
#include <curand.h>  // cuda random

#include "shark_cuda_helpers.h"

#include "AbstractWeightedLayer.h"

namespace shark {


// X and Y are assumed to be filled with random numbers
template <typename Precision>
__global__
void scale_kernel(Precision *weights, size_t num_weights, Precision range, size_t numBias)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < num_weights) {
		// if(x > num_weights - numBias) {
		// 	weights[x] = 0.0;
		// } else {
		weights[x] = weights[x] * 2 * range - range;
		// }
	}
}

template<>
void AbstractWeightedLayer<float>::xavier_init() {
	typedef float Precision;
	curandGenerator_t m_cuGen;

	curandCreateGenerator(&m_cuGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_cuGen, time(NULL));
	curandSetStream(m_cuGen, this->m_stream);

	// curandGenerateNormal(m_cuGen, this->mp_d_weightsData, this->m_numWeights, 0.0, sqrt(2.0/this->m_size_in));

	{
		curandGenerateUniform(m_cuGen, this->mp_d_weightsData, this->m_numWeights);
		// TODO: GPU-device dependant: threadsPerBlock = dim3(32, 32);
		int threadsPerBlock = 32 * 32;
		// Rounding up:
		// q = x/y + (x % y != 0);
		int numBlocks = this->m_numWeights / threadsPerBlock + (this->m_numWeights % threadsPerBlock != 0);
		Precision norm_term = sqrt(2.0 / this->m_size_in);
		scale_kernel<Precision><<<numBlocks, threadsPerBlock, 0, this->m_stream>>>(this->mp_d_weightsData, this->m_numWeights, norm_term, this->m_numBias);
	}
	cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
}

template<>
void AbstractWeightedLayer<double>::xavier_init() {
	typedef double Precision;
	curandGenerator_t m_cuGen;

	curandCreateGenerator(&m_cuGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_cuGen, time(NULL));
	curandSetStream(m_cuGen, this->m_stream);

	// curandGenerateNormalDouble(m_cuGen, this->mp_d_weightsData, this->m_numWeights, 0.0, sqrt(2.0/this->m_size_in));

	{
		curandGenerateUniformDouble(m_cuGen, this->mp_d_weightsData, this->m_numWeights);
		// TODO: GPU-device dependant: threadsPerBlock = dim3(32, 32);
		int threadsPerBlock = 32 * 32;
		// Rounding up:
		// q = x/y + (x % y != 0);
		int numBlocks = this->m_numWeights / threadsPerBlock + (this->m_numWeights % threadsPerBlock != 0);
		Precision norm_term = sqrt(2.0 / this->m_size_in);
		scale_kernel<Precision><<<numBlocks, threadsPerBlock, 0, this->m_stream>>>(this->mp_d_weightsData, this->m_numWeights, norm_term, this->m_numBias);
	}
}

// Explicit instantiation
template class AbstractWeightedLayer<float>;
template class AbstractWeightedLayer<double>;

}  // shark namespace end

