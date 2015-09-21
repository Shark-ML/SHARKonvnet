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
#ifndef SHARK_MODELS_DISTORT_LAYER_H
#define SHARK_MODELS_DISTORT_LAYER_H

#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <curand.h>  // cuda random
#include <npp.h>  // cuda image-processing
#include "shark_cuda_helpers.h"

#include "AbstractLayer.h"

namespace shark {

//! \brief Applies affine and elastic distortions to the input.
//!
//! Assumes the input is 2D with possible multiple channels, and distorts
//! uniformly across the channels. I.e. no new colors can occur because of
//! different distortions applied to the different channels.
//!
//! Parameters control the maximum affine translation, and affine rotation, and
//! strength, and \emph{smoothness} of the elastic distortion.
//!
//! Currently the smoothness kernels are hard-coded and limited to uneven sizes
//! between 3 and 15.
template<typename Precision>
class DistortLayer: public AbstractLayer<Precision> {
public:

	DistortLayer (Precision max_translation, Precision max_rotation,
	              Precision max_elastic_offset, int gauss_size,
	              std::string layer_name = "DistortLayer",
	              bool write_to_disk=false)
		: AbstractLayer<Precision> (layer_name) {
		this->m_max_elastic_offset = max_elastic_offset;
		this->m_max_rotation = max_rotation * 0.01745329251994;  // degrees-to-radians: rotation * PI/180
		this->m_max_translation = max_translation;
		m_eInterpolation = NPPI_INTER_LINEAR;
		this->m_gauss_size = gauss_size;
		this->m_test_eval = false;
	}

	void init(cuHandles handles, cudaStream_t stream) {
		this->fetch_backwards();
		this->m_handle = handles.cudnnHandle;
		this->m_stream = stream;

		srand (static_cast <unsigned> (time(0)));

		nppSetStream(stream);

		curandCreateGenerator(&m_cuGen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(m_cuGen, time(NULL));
		curandSetStream(m_cuGen, stream);


		m_data_size = this->m_height * this->m_width;

		cudaMalloc(&mp_d_pXMap, m_data_size * sizeof(Precision));
		cudaMalloc(&mp_d_pYMap, m_data_size * sizeof(Precision));
		cudaMalloc(&mp_d_kernel, m_gauss_size * m_gauss_size * sizeof(Precision));

		m_kernelSize = {m_gauss_size, m_gauss_size};

		// TODO: GPU-device dependant: m_threadsPerBlock = dim3(32, 32);
		m_threadsPerBlock = dim3(32, 32);
		// Rounding up:
		// q = x/y + (x % y != 0);
		m_numBlocks = dim3(this->m_width / m_threadsPerBlock.x + (this->m_width % m_threadsPerBlock.x != 0),
		                   this->m_height / m_threadsPerBlock.y + (this->m_height % m_threadsPerBlock.y != 0));

		if (m_gauss_size == 3) {
			// For double (gaussian kernel):
			Precision kernel [] = {0.07511361, 0.1238414, 0.07511361, 0.1238414,
			                       0.20417996, 0.1238414, 0.07511361, 0.1238414, 0.07511361
			                      };
			cudaMemcpyAsync(mp_d_kernel, kernel, m_gauss_size * m_gauss_size * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		} else if (m_gauss_size == 5) {
			// For double (gaussian kernel):
			Precision kernel [] = { 0.0317564010287, 0.0375157550304,
			                        0.0396589455083, 0.0375157550304, 0.0317564010287, 0.0375157550304,
			                        0.0443196278517, 0.0468515082395, 0.0443196278517, 0.0375157550304,
			                        0.0396589455083, 0.0468515082395, 0.0495280292438, 0.0468515082395,
			                        0.0396589455083, 0.0375157550304, 0.0443196278517, 0.0468515082395,
			                        0.0443196278517, 0.0375157550304, 0.0317564010287, 0.0375157550304,
			                        0.0396589455083, 0.0375157550304, 0.0317564010287
			                      };
			cudaMemcpyAsync(mp_d_kernel, kernel, m_gauss_size * m_gauss_size * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		} else if (m_gauss_size == 7) {
			// For double (gaussian kernel):
			Precision kernel [] = { 1.9651916124e-05, 0.000239409349497,
			                        0.00107295826498, 0.00176900911404, 0.00107295826498, 0.000239409349497,
			                        1.9651916124e-05, 0.000239409349497, 0.00291660295439, 0.0130713075832,
			                        0.0215509428483, 0.0130713075832, 0.00291660295439, 0.000239409349497,
			                        0.00107295826498, 0.0130713075832, 0.0585815363306, 0.0965846250186,
			                        0.0585815363306, 0.0130713075832, 0.00107295826498, 0.00176900911404,
			                        0.0215509428483, 0.0965846250186, 0.159241125691, 0.0965846250186,
			                        0.0215509428483, 0.00176900911404, 0.00107295826498, 0.0130713075832,
			                        0.0585815363306, 0.0965846250186, 0.0585815363306, 0.0130713075832,
			                        0.00107295826498, 0.000239409349497, 0.00291660295439, 0.0130713075832,
			                        0.0215509428483, 0.0130713075832, 0.00291660295439, 0.000239409349497,
			                        1.9651916124e-05, 0.000239409349497, 0.00107295826498, 0.00176900911404,
			                        0.00107295826498, 0.000239409349497, 1.9651916124e-05
			                      };
			cudaMemcpyAsync(mp_d_kernel, kernel, m_gauss_size * m_gauss_size * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		} else if (m_gauss_size == 9) {
			// For double (gaussian kernel):
			Precision kernel [] = { 0.0112359866226, 0.0116362091941,
			                        0.0119307812323, 0.0121110919, 0.0121717990008, 0.0121110919,
			                        0.0119307812323, 0.0116362091941, 0.0112359866226, 0.0116362091941,
			                        0.0120506875772, 0.0123557521856, 0.0125424854667, 0.0126053549368,
			                        0.0125424854667, 0.0123557521856, 0.0120506875772, 0.0116362091941,
			                        0.0119307812323, 0.0123557521856, 0.0126685395413, 0.012859999998,
			                        0.0129244610162, 0.012859999998, 0.0126685395413, 0.0123557521856,
			                        0.0119307812323, 0.0121110919, 0.0125424854667, 0.012859999998,
			                        0.0130543540089, 0.0131197892307, 0.0130543540089, 0.012859999998,
			                        0.0125424854667, 0.0121110919, 0.0121717990008, 0.0126053549368,
			                        0.0129244610162, 0.0131197892307, 0.0131855524479, 0.0131197892307,
			                        0.0129244610162, 0.0126053549368, 0.0121717990008, 0.0121110919,
			                        0.0125424854667, 0.012859999998, 0.0130543540089, 0.0131197892307,
			                        0.0130543540089, 0.012859999998, 0.0125424854667, 0.0121110919,
			                        0.0119307812323, 0.0123557521856, 0.0126685395413, 0.012859999998,
			                        0.0129244610162, 0.012859999998, 0.0126685395413, 0.0123557521856,
			                        0.0119307812323, 0.0116362091941, 0.0120506875772, 0.0123557521856,
			                        0.0125424854667, 0.0126053549368, 0.0125424854667, 0.0123557521856,
			                        0.0120506875772, 0.0116362091941, 0.0112359866226, 0.0116362091941,
			                        0.0119307812323, 0.0121110919, 0.0121717990008, 0.0121110919,
			                        0.0119307812323, 0.0116362091941, 0.0112359866226
			                      };
			cudaMemcpyAsync(mp_d_kernel, kernel, m_gauss_size * m_gauss_size * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		} else if (m_gauss_size == 11) {
			// For double (gaussian kernel):
			Precision kernel [] = { 2.21033494564e-12, 1.9896801102e-10,
			                        6.58891561024e-09, 8.02694246293e-08, 3.59742603044e-07, 5.93115281615e-07,
			                        3.59742603044e-07, 8.02694246293e-08, 6.58891561024e-09, 1.9896801102e-10,
			                        2.21033494564e-12, 1.9896801102e-10, 1.79105295726e-08, 5.93115281615e-07,
			                        7.22562333627e-06, 3.23829971326e-05, 5.33905361815e-05, 3.23829971326e-05,
			                        7.22562333627e-06, 5.93115281615e-07, 1.79105295726e-08, 1.9896801102e-10,
			                        6.58891561024e-09, 5.93115281615e-07, 1.96412806143e-05, 0.000239279782464,
			                        0.00107237758582, 0.00176805173597, 0.00107237758582, 0.000239279782464,
			                        1.96412806143e-05, 5.93115281615e-07, 6.58891561024e-09, 8.02694246293e-08,
			                        7.22562333627e-06, 0.000239279782464, 0.00291502450479, 0.0130642334629,
			                        0.0215392795956, 0.0130642334629, 0.00291502450479, 0.000239279782464,
			                        7.22562333627e-06, 8.02694246293e-08, 3.59742603044e-07, 3.23829971326e-05,
			                        0.00107237758582, 0.0130642334629, 0.0585498323229, 0.0965323539467,
			                        0.0585498323229, 0.0130642334629, 0.00107237758582, 3.23829971326e-05,
			                        3.59742603044e-07, 5.93115281615e-07, 5.33905361815e-05, 0.00176805173597,
			                        0.0215392795956, 0.0965323539467, 0.159154945263, 0.0965323539467,
			                        0.0215392795956, 0.00176805173597, 5.33905361815e-05, 5.93115281615e-07,
			                        3.59742603044e-07, 3.23829971326e-05, 0.00107237758582, 0.0130642334629,
			                        0.0585498323229, 0.0965323539467, 0.0585498323229, 0.0130642334629,
			                        0.00107237758582, 3.23829971326e-05, 3.59742603044e-07, 8.02694246293e-08,
			                        7.22562333627e-06, 0.000239279782464, 0.00291502450479, 0.0130642334629,
			                        0.0215392795956, 0.0130642334629, 0.00291502450479, 0.000239279782464,
			                        7.22562333627e-06, 8.02694246293e-08, 6.58891561024e-09, 5.93115281615e-07,
			                        1.96412806143e-05, 0.000239279782464, 0.00107237758582, 0.00176805173597,
			                        0.00107237758582, 0.000239279782464, 1.96412806143e-05, 5.93115281615e-07,
			                        6.58891561024e-09, 1.9896801102e-10, 1.79105295726e-08, 5.93115281615e-07,
			                        7.22562333627e-06, 3.23829971326e-05, 5.33905361815e-05, 3.23829971326e-05,
			                        7.22562333627e-06, 5.93115281615e-07, 1.79105295726e-08, 1.9896801102e-10,
			                        2.21033494564e-12, 1.9896801102e-10, 6.58891561024e-09, 8.02694246293e-08,
			                        3.59742603044e-07, 5.93115281615e-07, 3.59742603044e-07, 8.02694246293e-08,
			                        6.58891561024e-09, 1.9896801102e-10, 2.21033494564e-12
			                      };
			cudaMemcpyAsync(mp_d_kernel, kernel, m_gauss_size * m_gauss_size * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		} else if (m_gauss_size == 13) {
			// For double (gaussian kernel):
			Precision kernel [] = { 3.69163520111e-17, 9.03313350574e-15,
			                        8.13136764841e-13, 2.69273914719e-11, 3.28042783984e-10, 1.47018575959e-09,
			                        2.42392653371e-09, 1.47018575959e-09, 3.28042783984e-10, 2.69273914719e-11,
			                        8.13136764841e-13, 9.03313350574e-15, 3.69163520111e-17, 9.03313350574e-15,
			                        2.21033489192e-12, 1.98968006184e-10, 6.5889154501e-09, 8.02694226785e-08,
			                        3.597425943e-07, 5.931152672e-07, 3.597425943e-07, 8.02694226785e-08,
			                        6.5889154501e-09, 1.98968006184e-10, 2.21033489192e-12, 9.03313350574e-15,
			                        8.13136764841e-13, 1.98968006184e-10, 1.79105291373e-08, 5.931152672e-07,
			                        7.22562316067e-06, 3.23829963455e-05, 5.33905348839e-05, 3.23829963455e-05,
			                        7.22562316067e-06, 5.931152672e-07, 1.79105291373e-08, 1.98968006184e-10,
			                        8.13136764841e-13, 2.69273914719e-11, 6.5889154501e-09, 5.931152672e-07,
			                        1.96412801369e-05, 0.000239279776649, 0.00107237755976, 0.001768051693,
			                        0.00107237755976, 0.000239279776649, 1.96412801369e-05, 5.931152672e-07,
			                        6.5889154501e-09, 2.69273914719e-11, 3.28042783984e-10, 8.02694226785e-08,
			                        7.22562316067e-06, 0.000239279776649, 0.00291502443394, 0.0130642331454,
			                        0.0215392790721, 0.0130642331454, 0.00291502443394, 0.000239279776649,
			                        7.22562316067e-06, 8.02694226785e-08, 3.28042783984e-10, 1.47018575959e-09,
			                        3.597425943e-07, 3.23829963455e-05, 0.00107237755976, 0.0130642331454,
			                        0.0585498308999, 0.0965323516006, 0.0585498308999, 0.0130642331454,
			                        0.00107237755976, 3.23829963455e-05, 3.597425943e-07, 1.47018575959e-09,
			                        2.42392653371e-09, 5.931152672e-07, 5.33905348839e-05, 0.001768051693,
			                        0.0215392790721, 0.0965323516006, 0.159154941395, 0.0965323516006,
			                        0.0215392790721, 0.001768051693, 5.33905348839e-05, 5.931152672e-07,
			                        2.42392653371e-09, 1.47018575959e-09, 3.597425943e-07, 3.23829963455e-05,
			                        0.00107237755976, 0.0130642331454, 0.0585498308999, 0.0965323516006,
			                        0.0585498308999, 0.0130642331454, 0.00107237755976, 3.23829963455e-05,
			                        3.597425943e-07, 1.47018575959e-09, 3.28042783984e-10, 8.02694226785e-08,
			                        7.22562316067e-06, 0.000239279776649, 0.00291502443394, 0.0130642331454,
			                        0.0215392790721, 0.0130642331454, 0.00291502443394, 0.000239279776649,
			                        7.22562316067e-06, 8.02694226785e-08, 3.28042783984e-10, 2.69273914719e-11,
			                        6.5889154501e-09, 5.931152672e-07, 1.96412801369e-05, 0.000239279776649,
			                        0.00107237755976, 0.001768051693, 0.00107237755976, 0.000239279776649,
			                        1.96412801369e-05, 5.931152672e-07, 6.5889154501e-09, 2.69273914719e-11,
			                        8.13136764841e-13, 1.98968006184e-10, 1.79105291373e-08, 5.931152672e-07,
			                        7.22562316067e-06, 3.23829963455e-05, 5.33905348839e-05, 3.23829963455e-05,
			                        7.22562316067e-06, 5.931152672e-07, 1.79105291373e-08, 1.98968006184e-10,
			                        8.13136764841e-13, 9.03313350574e-15, 2.21033489192e-12, 1.98968006184e-10,
			                        6.5889154501e-09, 8.02694226785e-08, 3.597425943e-07, 5.931152672e-07,
			                        3.597425943e-07, 8.02694226785e-08, 6.5889154501e-09, 1.98968006184e-10,
			                        2.21033489192e-12, 9.03313350574e-15, 3.69163520111e-17, 9.03313350574e-15,
			                        8.13136764841e-13, 2.69273914719e-11, 3.28042783984e-10, 1.47018575959e-09,
			                        2.42392653371e-09, 1.47018575959e-09, 3.28042783984e-10, 2.69273914719e-11,
			                        8.13136764841e-13, 9.03313350574e-15, 3.69163520111e-17
			                      };
			cudaMemcpyAsync(mp_d_kernel, kernel, m_gauss_size * m_gauss_size * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		} else if (m_gauss_size == 15) {
			// For double (gaussian kernel):
			Precision kernel [] = { 0.0, 0.0, 0.0, 1.22250168147e-15,
			                        4.04836957021e-14, 4.93192378398e-13, 2.21033489184e-12, 3.64422615155e-12,
			                        2.21033489184e-12, 4.93192378398e-13, 4.04836957021e-14, 1.22250168147e-15,
			                        0.0, 0.0, 0.0, 0.0, 3.69163520097e-17, 9.03313350541e-15, 8.13136764811e-13,
			                        2.6927391471e-11, 3.28042783972e-10, 1.47018575953e-09, 2.42392653362e-09,
			                        1.47018575953e-09, 3.28042783972e-10, 2.6927391471e-11, 8.13136764811e-13,
			                        9.03313350541e-15, 3.69163520097e-17, 0.0, 0.0, 9.03313350541e-15,
			                        2.21033489184e-12, 1.98968006177e-10, 6.58891544986e-09, 8.02694226755e-08,
			                        3.59742594287e-07, 5.93115267178e-07, 3.59742594287e-07, 8.02694226755e-08,
			                        6.58891544986e-09, 1.98968006177e-10, 2.21033489184e-12, 9.03313350541e-15,
			                        0.0, 1.22250168147e-15, 8.13136764811e-13, 1.98968006177e-10,
			                        1.79105291366e-08, 5.93115267178e-07, 7.2256231604e-06, 3.23829963444e-05,
			                        5.33905348819e-05, 3.23829963444e-05, 7.2256231604e-06, 5.93115267178e-07,
			                        1.79105291366e-08, 1.98968006177e-10, 8.13136764811e-13, 1.22250168147e-15,
			                        4.04836957021e-14, 2.6927391471e-11, 6.58891544986e-09, 5.93115267178e-07,
			                        1.96412801362e-05, 0.00023927977664, 0.00107237755972, 0.00176805169293,
			                        0.00107237755972, 0.00023927977664, 1.96412801362e-05, 5.93115267178e-07,
			                        6.58891544986e-09, 2.6927391471e-11, 4.04836957021e-14, 4.93192378398e-13,
			                        3.28042783972e-10, 8.02694226755e-08, 7.2256231604e-06, 0.00023927977664,
			                        0.00291502443383, 0.0130642331449, 0.0215392790714, 0.0130642331449,
			                        0.00291502443383, 0.00023927977664, 7.2256231604e-06, 8.02694226755e-08,
			                        3.28042783972e-10, 4.93192378398e-13, 2.21033489184e-12, 1.47018575953e-09,
			                        3.59742594287e-07, 3.23829963444e-05, 0.00107237755972, 0.0130642331449,
			                        0.0585498308978, 0.096532351597, 0.0585498308978, 0.0130642331449,
			                        0.00107237755972, 3.23829963444e-05, 3.59742594287e-07, 1.47018575953e-09,
			                        2.21033489184e-12, 3.64422615155e-12, 2.42392653362e-09, 5.93115267178e-07,
			                        5.33905348819e-05, 0.00176805169293, 0.0215392790714, 0.096532351597,
			                        0.159154941389, 0.096532351597, 0.0215392790714, 0.00176805169293,
			                        5.33905348819e-05, 5.93115267178e-07, 2.42392653362e-09, 3.64422615155e-12,
			                        2.21033489184e-12, 1.47018575953e-09, 3.59742594287e-07, 3.23829963444e-05,
			                        0.00107237755972, 0.0130642331449, 0.0585498308978, 0.096532351597,
			                        0.0585498308978, 0.0130642331449, 0.00107237755972, 3.23829963444e-05,
			                        3.59742594287e-07, 1.47018575953e-09, 2.21033489184e-12, 4.93192378398e-13,
			                        3.28042783972e-10, 8.02694226755e-08, 7.2256231604e-06, 0.00023927977664,
			                        0.00291502443383, 0.0130642331449, 0.0215392790714, 0.0130642331449,
			                        0.00291502443383, 0.00023927977664, 7.2256231604e-06, 8.02694226755e-08,
			                        3.28042783972e-10, 4.93192378398e-13, 4.04836957021e-14, 2.6927391471e-11,
			                        6.58891544986e-09, 5.93115267178e-07, 1.96412801362e-05, 0.00023927977664,
			                        0.00107237755972, 0.00176805169293, 0.00107237755972, 0.00023927977664,
			                        1.96412801362e-05, 5.93115267178e-07, 6.58891544986e-09, 2.6927391471e-11,
			                        4.04836957021e-14, 1.22250168147e-15, 8.13136764811e-13, 1.98968006177e-10,
			                        1.79105291366e-08, 5.93115267178e-07, 7.2256231604e-06, 3.23829963444e-05,
			                        5.33905348819e-05, 3.23829963444e-05, 7.2256231604e-06, 5.93115267178e-07,
			                        1.79105291366e-08, 1.98968006177e-10, 8.13136764811e-13, 1.22250168147e-15,
			                        0.0, 9.03313350541e-15, 2.21033489184e-12, 1.98968006177e-10,
			                        6.58891544986e-09, 8.02694226755e-08, 3.59742594287e-07, 5.93115267178e-07,
			                        3.59742594287e-07, 8.02694226755e-08, 6.58891544986e-09, 1.98968006177e-10,
			                        2.21033489184e-12, 9.03313350541e-15, 0.0, 0.0, 3.69163520097e-17,
			                        9.03313350541e-15, 8.13136764811e-13, 2.6927391471e-11, 3.28042783972e-10,
			                        1.47018575953e-09, 2.42392653362e-09, 1.47018575953e-09, 3.28042783972e-10,
			                        2.6927391471e-11, 8.13136764811e-13, 9.03313350541e-15, 3.69163520097e-17,
			                        0.0, 0.0, 0.0, 0.0, 1.22250168147e-15, 4.04836957021e-14, 4.93192378398e-13,
			                        2.21033489184e-12, 3.64422615155e-12, 2.21033489184e-12, 4.93192378398e-13,
			                        4.04836957021e-14, 1.22250168147e-15, 0.0, 0.0, 0.0
			                      };
			cudaMemcpyAsync(mp_d_kernel, kernel, m_gauss_size * m_gauss_size * sizeof(Precision), cudaMemcpyHostToDevice, this->m_stream);
		}
		// cudaStreamSynchronize(this->m_stream);
		cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
	}

	void prepare_gpu() {
		this->m_channels_out = this->m_channels;
		this->m_height_out = this->m_height;
		this->m_width_out = this->m_width;
		this->m_size_out = this->m_size_in;
		this->m_outputDesc = this->mep_prevLayer->m_outputDesc;

		m_data_size = this->m_height * this->m_width;

		m_oSrcSize = {this->m_height, this->m_width};
		m_oDstSize = {this->m_height, this->m_width};
		m_nSrcStep = m_oSrcSize.width * sizeof(Precision);
		m_nDstStep = m_oDstSize.width * sizeof(Precision);
		m_oSrcROI = {0, 0, m_oSrcSize.width, m_oSrcSize.height};

		// Malloc
		if (this->m_numPatterns_old < this->m_numPatterns) {
			if (this->m_numPatterns_old != 0) {
				cudaFree(this->mp_d_outputData);
				// cudaFree(this->mp_d_diffDataOut);
			}
			size_t outSize = this->m_numPatterns * this->m_size_out * sizeof(Precision);
			cudaMalloc(&this->mp_d_outputData, outSize);
			// TODO: proper distortion handling. I.e. reverse distort on backwards_gpu.
			// cudaMalloc(&this->mp_d_diffDataOut, outSize);
			this->mp_d_diffDataOut = this->mep_d_prevDiffData;
			// TODO: only alloc'ing for a single channel. Assumes sequential launch of distortion-kernel.

			cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
			this->m_gpu_mem = outSize  // mp_d_outputData
			                  + m_data_size * 2 * sizeof(Precision)  // mp_d_pXMap and mp_d_pYMap
			                  + m_gauss_size * m_gauss_size * sizeof(Precision);  // mp_d_kernel
		}
		cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
		// Malloc end
	}

	// Forward declaration. Specialized below.
	void forward_gpu() const {
		throw std::logic_error("DistortLayer::forward_gpu was not specialized for template arg.");
	};

	void backward_gpu() const {}


	curandGenerator_t m_cuGen;

	Precision *mp_d_pXMap;
	Precision *mp_d_pYMap;

	size_t m_data_size;
	NppiSize m_oSrcSize;
	NppiSize m_oDstSize;
	NppiSize m_kernelSize;
	int m_nSrcStep;
	int m_nDstStep;
	NppiRect m_oSrcROI;

	// NPP_MASK_SIZE_1_X_3
	// NPP_MASK_SIZE_1_X_5
	// NPP_MASK_SIZE_3_X_1
	// NPP_MASK_SIZE_5_X_1
	// NPP_MASK_SIZE_3_X_3
	// NPP_MASK_SIZE_5_X_5
	// NPP_MASK_SIZE_7_X_7
	// NPP_MASK_SIZE_9_X_9
	// NPP_MASK_SIZE_11_X_11
	// NPP_MASK_SIZE_13_X_13
	// NPP_MASK_SIZE_15_X_15
	NppiMaskSize m_maskSize;

	// Amount of distortion offset and rotation
	Precision m_max_elastic_offset;
	Precision m_max_rotation;
	Precision m_max_translation;

	// Amount of distortion smoothing
	int m_gauss_size;

	// One of:
	// NPPI_INTER_NN
	// NPPI_INTER_LINEAR
	// NPPI_INTER_CUBIC
	// NPPI_INTER_CUBIC2P_BSPLINE
	// NPPI_INTER_CUBIC2P_CATMULLROM
	// NPPI_INTER_CUBIC2P_B05C03
	// NPPI_INTER_- LANCZOS
	int m_eInterpolation;

	// Only used for double
	Precision *mp_d_kernel;

	bool m_test_eval;

	dim3 m_threadsPerBlock;
	dim3 m_numBlocks;
};  // end class DistortLayer

}  // end namespace shark
#endif