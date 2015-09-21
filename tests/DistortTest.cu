/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// #include <ImagesCPU.h>
// #include <ImagesNPP.h>
// #include <ImageIO.h>
// #include <Exceptions.h>

#ifndef DISTORTTEST
#define DISTORTTEST

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>  // cuda deep neural networks
#include <curand.h>  // cuda random
#include <npp.h>  // cuda image-processing

template <typename Precision>
__global__
void warp_assemble(Precision *X, Precision *Y, Precision offset)
{
																							// random num in (0,1] centered around 0.
	X[threadIdx.y * blockDim.x + threadIdx.x] = (X[threadIdx.y * blockDim.x + threadIdx.x] - 0.5) * offset +  // the random relative offset
																							blockIdx.x * blockDim.x + threadIdx.x;  // the absolute coordinate
	Y[threadIdx.x * blockDim.y + threadIdx.y] = (Y[threadIdx.x * blockDim.y + threadIdx.y] - 0.5) * offset +
																							blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
	typedef float Precision;
	size_t data_size = 28*28;
	NppiSize m_oSrcSize = {28, 28};
	NppiSize m_oDstSize = {28, 28};
	int m_nSrcStep = m_oSrcSize.width * sizeof(Precision);
	int m_nDstStep = m_oDstSize.width * sizeof(Precision);
	NppiRect m_oSrcROI = {0,0,
											m_oSrcSize.width, m_oSrcSize.height};
	
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
	NppiMaskSize m_maskSize = NPP_MASK_SIZE_9_X_9;
	
	curandGenerator_t m_cuGen;
	curandCreateGenerator(&m_cuGen, CURAND_RNG_PSEUDO_DEFAULT);
	
	curandSetPseudoRandomGeneratorSeed(m_cuGen, time(NULL));

	Precision *d_inputData;
	Precision *mp_d_pXMap;
	Precision *mp_d_pYMap;
	// Precision *mp_d_pXMap_;
	// Precision *mp_d_pYMap_;
	Precision *d_outputData;

	cudaMalloc(&d_inputData, data_size * sizeof(Precision));
	cudaMalloc(&mp_d_pXMap, data_size * sizeof(Precision));
	cudaMalloc(&mp_d_pYMap, data_size * sizeof(Precision));
	// cudaMalloc(&mp_d_pXMap_, data_size * sizeof(Precision));
	// cudaMalloc(&mp_d_pYMap_, data_size * sizeof(Precision));
	cudaMalloc(&d_outputData, data_size * sizeof(Precision));
	
	// curandGenerateUniformDouble
	// curandGenerateNormal(m_cuGen, mp_d_pXMap, data_size, 0.0, 10.0);
	// curandGenerateNormal(m_cuGen, mp_d_pYMap, data_size, 0.0, 10.0);
	// Random numbers (0,1]
	curandGenerateUniform(m_cuGen, mp_d_pXMap, data_size);
	curandGenerateUniform(m_cuGen, mp_d_pYMap, data_size);
	
	// Blurr with a gaussian
	nppiFilterGauss_32f_C1R(mp_d_pXMap, m_nSrcStep,
													mp_d_pXMap, m_nSrcStep,
													m_oSrcSize, m_maskSize);
	nppiFilterGauss_32f_C1R(mp_d_pYMap, m_nSrcStep,
													mp_d_pYMap, m_nSrcStep,
													m_oSrcSize, m_maskSize);

	// Convert from relative offsets to absolute offsets:
	dim3 threadsPerBlock(m_oSrcSize.width, m_oSrcSize.height);
	dim3 numBlocks(m_oSrcSize.width / threadsPerBlock.x, m_oSrcSize.height / threadsPerBlock.y);
	
	warp_assemble<Precision><<<numBlocks, threadsPerBlock>>>(mp_d_pXMap, mp_d_pYMap, 5.0);

	Precision distort[784];
	cudaMemcpy(distort, mp_d_pYMap, data_size * sizeof(Precision), cudaMemcpyDeviceToHost);
	for (int i=0; i < 784; i++)
		{std::cout << i << ": "<< distort[i] << std::endl;}
	std::cout << std::endl << std::endl;
	
	
	Precision patternData[784] = {0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.012,0.071,0.071,0.071,0.494,0.533,0.686,0.102,0.651,1.000,0.969,0.498,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.118,0.141,0.369,0.604,0.667,0.992,0.992,0.992,0.992,0.992,0.882,0.675,0.992,0.949,0.765,0.251,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.192,0.933,0.992,0.992,0.992,0.992,0.992,0.992,0.992,0.992,0.984,0.365,0.322,0.322,0.220,0.153,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.071,0.859,0.992,0.992,0.992,0.992,0.992,0.776,0.714,0.969,0.945,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.314,0.612,0.420,0.992,0.992,0.804,0.043,0.000,0.169,0.604,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.055,0.004,0.604,0.992,0.353,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.545,0.992,0.745,0.008,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.043,0.745,0.992,0.275,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.137,0.945,0.882,0.627,0.424,0.004,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.318,0.941,0.992,0.992,0.467,0.098,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.176,0.729,0.992,0.992,0.588,0.106,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.063,0.365,0.988,0.992,0.733,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.976,0.992,0.976,0.251,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.180,0.510,0.718,0.992,0.992,0.812,0.008,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.153,0.580,0.898,0.992,0.992,0.992,0.980,0.714,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.094,0.447,0.867,0.992,0.992,0.992,0.992,0.788,0.306,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.090,0.259,0.835,0.992,0.992,0.992,0.992,0.776,0.318,0.008,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.071,0.671,0.859,0.992,0.992,0.992,0.992,0.765,0.314,0.035,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.216,0.675,0.886,0.992,0.992,0.992,0.992,0.957,0.522,0.043,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.533,0.992,0.992,0.992,0.831,0.529,0.518,0.063,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000};

	cudaMemcpy(d_inputData, patternData, data_size * sizeof(Precision), cudaMemcpyHostToDevice);


	// One of:
	// NPPI_INTER_NN
	// NPPI_INTER_LINEAR
	// NPPI_INTER_CUBIC
	// NPPI_INTER_CUBIC2P_BSPLINE
	// NPPI_INTER_CUBIC2P_CATMULLROM
	// NPPI_INTER_CUBIC2P_B05C03
	// NPPI_INTER_- LANCZOS
	int m_eInterpolation = NPPI_INTER_LINEAR;
	// int m_eInterpolation = NPPI_INTER_NN;
	
	nppiRemap_32f_C1R(d_inputData, m_oSrcSize, m_nSrcStep,
										m_oSrcROI,
										mp_d_pXMap, m_nSrcStep,
										mp_d_pYMap, m_nSrcStep,
										d_outputData, m_nDstStep, m_oDstSize, m_eInterpolation);
	
	
	cudaMemcpy(patternData, d_outputData, data_size * sizeof(Precision), cudaMemcpyDeviceToHost);
}

#endif  // DISTORTTEST
