/*!
 *
 *
 * \brief       Implements a distortion layer on nvidia gpu's.
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
#include <cudnn.h>  // cuda deep neural networks
#include <curand.h>  // cuda random
#include <npp.h>  // cuda image-processing

// #include <shark/Rng/GlobalRng.h>
#include "shark_cuda_helpers.h"

#include "DistortLayer.h"

namespace shark {


// X and Y are assumed to be filled with random numbers
template <typename Precision>
__global__
void warp_assemble(Precision *X, Precision *Y,
                   Precision moffset, Precision rotation,
                   Precision x_off, Precision y_off,
                   int height, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// translation so origo is in the center
	Precision tx = width / 2.0;
	Precision ty = height / 2.0;

	// rotation
	Precision cosTheta = cos(rotation);
	Precision sinTheta = sin(rotation);

	Precision xdist = X[y * width + x] * moffset +  // the random relative offset
	                  cosTheta * (x-tx) - sinTheta * (y-ty) + tx + x_off;  // the rotated absolute coordinate
	Precision ydist = Y[y * width + x] * moffset +  // the random relative offset
	                  sinTheta * (x-tx) + cosTheta * (y-ty) + ty + y_off;  // the rotated absolute coordinate


	if (x < width && y < height) {
		// X[y * width + x] = x;
		// Y[y * width + x] = y;
		if (xdist < 0) {
			X[y * width + x] = 0.0;
		} else if (xdist >= width) {
			X[y * width + x] = width-1;
		} else {
			X[y * width + x] = xdist;
		}

		if (ydist < 0) {
			Y[y * width + x] = 0.0;
		} else if (ydist >= height) {
			Y[y * width + x] = height-1;
		} else {
			Y[y * width + x] = ydist;
		}
	}
}

// // forward_gpu float specialization start
template<>
void DistortLayer<float>::forward_gpu() const {
	typedef float Precision;
	// TODO: Optimizations. Is slow.
	// TODO: Check if in-place modification of mep_d_inputData is possible.

	// By-pass distortions when not training:
	if (this->m_test_eval) {
		cudaMemcpyAsync(mp_d_outputData, mep_d_inputData, this->m_numPatterns * this->m_size_in * sizeof(Precision), cudaMemcpyDeviceToDevice, this->m_stream);
		return;
	}

	Precision rot_rand;
	Precision x_off;
	Precision y_off;
	for (int n=0; n < this->m_numPatterns; n++) {
		// Generate initial vector field:
		curandGenerateNormal(m_cuGen, mp_d_pXMap, m_data_size, 0.0, 0.5);
		curandGenerateNormal(m_cuGen, mp_d_pYMap, m_data_size, 0.0, 0.5);

		// Generate random rotation and translation on CPU:
		// TODO: Use Rng::uni(-1.0, 1.0) here. Error in shark::Dirichlet:l96? stderr: /home/hnr137/Shark/include/shark/Rng/Dirichlet.h(96): error: a pointer to a bound function may only be used to call the function
		// Angle range from [-rotation, rotation]:
		rot_rand = (-1.0 + (static_cast <Precision> (rand()) / static_cast <Precision> (RAND_MAX/2)))
		           * m_max_rotation;
		// TODO: Use Rng::uni(-1.0, 1.0) here.
		// Translate range from [-m_max_translation, m_max_translation]:
		x_off = (-1.0 + (static_cast <Precision> (rand()) / static_cast <Precision> (RAND_MAX/2)))
		        * m_max_translation;
		y_off = (-1.0 + (static_cast <Precision> (rand()) / static_cast <Precision> (RAND_MAX/2)))
		        * m_max_translation;

		// Smooth the vector field with a gaussian
		nppiFilter_32f_C1R(mp_d_pXMap, m_nSrcStep,
		                   mp_d_pXMap, m_nSrcStep,
		                   m_oSrcSize,
		                   mp_d_kernel,
		                   m_kernelSize, {0, 0});
		nppiFilter_32f_C1R(mp_d_pYMap, m_nSrcStep,
		                   mp_d_pYMap, m_nSrcStep,
		                   m_oSrcSize,
		                   mp_d_kernel,
		                   m_kernelSize, {0, 0});
		// Mix relative offsets with random rotations and convert to absolute offsets:
		warp_assemble<Precision><<<m_numBlocks, m_threadsPerBlock, 0, this->m_stream>>>(mp_d_pXMap, mp_d_pYMap,
		                                                                                m_max_elastic_offset, rot_rand,
		                                                                                x_off, y_off,
		                                                                                this->m_height, this->m_width);
		for (int c=0; c < this->m_channels; c++) {
			// TODO: Maybe switch on 3-channels? I.e. use nppiRemap_64f_C3R when appropriate.
			// TODO: Bogus offsets. Assumes size is the same.
			nppiRemap_32f_C1R(mep_d_inputData + n * this->m_size_in + c * m_data_size, m_oSrcSize, m_nSrcStep,
			                  m_oSrcROI,
			                  mp_d_pXMap, m_nSrcStep,
			                  mp_d_pYMap, m_nSrcStep,
			                  mp_d_outputData + n * this->m_size_out + c * m_data_size, m_nDstStep, m_oDstSize, m_eInterpolation);
		}
		cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
	}
} // forward_gpu float specialization end


// forward_gpu double specialization start
template<>
void DistortLayer<double>::forward_gpu() const {
	typedef double Precision;
	// TODO: Optimizations. Is slow.
	// TODO: Check if in-place modification of mep_d_inputData is possible.

	// By-pass distortions when not training:
	if (this->m_test_eval) {
		cudaMemcpyAsync(mp_d_outputData, mep_d_inputData, this->m_numPatterns * this->m_size_in * sizeof(Precision), cudaMemcpyDeviceToDevice, this->m_stream);
		return;
	}

	Precision rot_rand;
	Precision x_off;
	Precision y_off;
	// srand (static_cast <unsigned> (time(0)));
	for (int n=0; n < this->m_numPatterns; n++) {
		// Generate initial vector field:
		curandGenerateNormalDouble(m_cuGen, mp_d_pXMap, m_data_size, 0.0, 0.5);
		curandGenerateNormalDouble(m_cuGen, mp_d_pYMap, m_data_size, 0.0, 0.5);

		// Generate random rotation and translation on CPU:
		// TODO: Use Rng::uni(-1.0, 1.0) here. Error in shark::Dirichlet:l96? stderr: /home/hnr137/Shark/include/shark/Rng/Dirichlet.h(96): error: a pointer to a bound function may only be used to call the function
		// Angle range from [-rotation, rotation]:
		rot_rand = (-1.0 + (static_cast <Precision> (rand()) / static_cast <Precision> (RAND_MAX/2)))
		           * m_max_rotation;
		// TODO: Use Rng::uni(-1.0, 1.0) here.
		// Translate range from [-m_max_translation, m_max_translation]:
		x_off = (-1.0 + (static_cast <Precision> (rand()) / static_cast <Precision> (RAND_MAX/2)))
		        * m_max_translation;
		y_off = (-1.0 + (static_cast <Precision> (rand()) / static_cast <Precision> (RAND_MAX/2)))
		        * m_max_translation;

		// Smooth the vector field with a gaussian
		// TODO: Replace with cuDNN operations.
		nppiFilter_64f_C1R(mp_d_pXMap, m_nSrcStep,
		                   mp_d_pXMap, m_nSrcStep,
		                   m_oSrcSize,
		                   mp_d_kernel,
		                   m_kernelSize, {0, 0});
		nppiFilter_64f_C1R(mp_d_pYMap, m_nSrcStep,
		                   mp_d_pYMap, m_nSrcStep,
		                   m_oSrcSize,
		                   mp_d_kernel,
		                   m_kernelSize, {0, 0});
		// Mix relative offsets with random rotations and convert to absolute offsets:
		warp_assemble<Precision><<<m_numBlocks, m_threadsPerBlock, 0, this->m_stream>>>(mp_d_pXMap, mp_d_pYMap,
		        m_max_elastic_offset, rot_rand,
		        x_off, y_off,
		        this->m_height, this->m_width);
		for (int c=0; c < this->m_channels; c++) {
			// TODO: Maybe switch on 3-channels? I.e. use nppiRemap_64f_C3R when appropriate.
			// TODO: Bogus d_* offsets. Assumes size is the same.
			nppiRemap_64f_C1R(mep_d_inputData + n * this->m_size_in + c * m_data_size, m_oSrcSize, m_nSrcStep,
			                  m_oSrcROI,
			                  mp_d_pXMap, m_nSrcStep,
			                  mp_d_pYMap, m_nSrcStep,
			                  mp_d_outputData + n * this->m_size_out + c * m_data_size, m_nDstStep, m_oDstSize, m_eInterpolation);
			// cudaMemcpyAsync(mp_d_outputData, mp_d_pYMap, this->m_size_in * sizeof(Precision), cudaMemcpyDeviceToHost, this->m_stream);
		}
	}
	cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
} // forward_gpu double specialization end

}  // end namespace shark
