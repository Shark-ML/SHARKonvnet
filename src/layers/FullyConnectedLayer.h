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
#ifndef SHARK_MODELS_OutputLayer_H
#define SHARK_MODELS_OutputLayer_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Neurons.h>
#include <boost/serialization/vector.hpp>
#include <shark/LinAlg/BLAS/matrix_set.hpp>
#include "cudnn.h"
#include "cuda_runtime.h"
#include "cublas_api.h"
#include "cublas_v2.h"
#include "shark_cuda_helpers.h"
#include "AbstractLayer.h"
#include "AbstractWeightedLayer.h"

namespace shark {

//! \brief Provides a fully connected layer.
//!
//! Every node in the input layer is connected to every node of the output
//! layer, with individual weights for each connection. Each output node also
//! have a bias-parameter.
//!
//! Currently a gemm-based solution using cuBLAS.
template<typename Precision>
class FullyConnectedLayer: public AbstractWeightedLayer<Precision> {
public:

	FullyConnectedLayer (int out_neurons,
	                     std::string layer_name = "FullyConnectedLayer")
		: AbstractWeightedLayer<Precision> (layer_name) {
		this->m_size_out = out_neurons;
		mp_d_biasData = NULL;
		mp_d_biasDiffData = NULL;
		m_setup = true;
	}

	~FullyConnectedLayer () {
		// Only free the parts that this layer is responsible for.
		// Warn of errors when freeing/destroying.
		cuda_status_check_warn(__FILE__, __FUNCTION__, __LINE__);
	}

	void init(cuHandles handles, cudaStream_t stream) {
		this->m_handle = handles.cudnnHandle;
		this->m_cublasHandle = handles.cublasHandle;
		this->m_stream = stream;
		cudnnStatus_t status;
		status = cudnnCreateTensorDescriptor(&this->m_outputDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		status = cudnnCreateTensorDescriptor(&m_biasTens);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void prepare_gpu() {
		cudnnStatus_t status;

		this->m_channels_out = 1;
		this->m_height_out = 1;
		this->m_width_out = this->m_size_out;
		this->m_numBias = this->m_size_out;

		// /* Set and allocate output tensor descriptor */
		cudnnSetTensor4dDescriptor(this->m_outputDesc,
		                           CUDNN_TENSOR_NCHW,  // Data-layout
		                           this->m_cudnn_prec,  // Data-type
		                           this->m_numPatterns,        // Number of input images
		                           this->m_channels_out,       // number of feature maps
		                           this->m_height_out,         // Feature map m_height
		                           this->m_width_out);         // Feature map width
		// Output descriptor end

		this->m_numWeights = this->m_size_in * this->m_size_out +  // weights
		                     this->m_numBias;  // bias

		// bias tensor start
		cudnnSetTensor4dDescriptor(m_biasTens,
		                           CUDNN_TENSOR_NCHW,
		                           this->m_cudnn_prec,
		                           1, 1, 1,
		                           this->m_numBias);
		// bias tensor end


		// Malloc
		if (this->m_numPatterns_old < this->m_numPatterns) {
			if (this->m_numPatterns_old != 0) {
				cudaFree(this->mp_d_outputData);
				cudaFree(this->mp_d_diffDataOut);
				cudaFree(mp_d_biasOnes);
			}
			size_t outSize = this->m_numPatterns * this->m_size_out * sizeof(Precision);
			cudaMalloc(&this->mp_d_outputData, outSize);
			cudaMalloc(&this->mp_d_diffDataOut, outSize);
			cudaMalloc(&mp_d_biasOnes, this->m_numBias * sizeof(Precision));
			Precision one = 1.0;
			status = cudnnSetTensor(this->m_handle, m_biasTens, mp_d_biasOnes, &one);
			cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
			if (m_setup) {
				cudaMalloc(&this->mp_d_weightsData, this->m_numWeights * sizeof(Precision));
				this->mp_d_biasData = this->mp_d_weightsData + this->m_size_in * this->m_size_out;
				cudaMalloc(&this->mp_d_weightsDiffData, this->m_numWeights * sizeof(Precision));
				this->mp_d_biasDiffData = this->mp_d_weightsDiffData + this->m_size_in * this->m_size_out;
				m_setup = false;
			}
			this->m_gpu_mem = 2 * outSize + 2 * this->m_numWeights * sizeof(Precision) + this->m_numBias * sizeof(Precision);
			cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
		}
		// Malloc end
	}  // prep end

	// Forward declaration. Specialized below.
	void forward_gpu() const {
		throw std::logic_error("FullyConnectedLayer::backward_gpu was not specialized for template arg.");
	};
	// Forward declaration. Specialized below.
	void backward_gpu() const {
		throw std::logic_error("FullyConnectedLayer::backward_gpu was not specialized for template arg.");
	};

	cublasHandle_t m_cublasHandle;

	bool m_setup;

	Precision               *mp_d_biasData;
	Precision               *mp_d_biasDiffData;
	cudnnTensorDescriptor_t m_biasTens;
	Precision               *mp_d_biasOnes;
};  // end FullyConnectedLayer class declaration

// forward_gpu float specialization start
template<>
void FullyConnectedLayer<float>::forward_gpu() const {
	float alpha = 1.0;
	float beta = 0.0;
	cublasStatus_t status;

	// TODO: Doc the column-major-reversedDimension-reversedTransposition setup here.
	//       Did this: Setup every thing like row-major. Then:
	//       * Reverse transpositions
	//       * Reverse lda, ldb with the other dim of their respective matrices.
	status = cublasSgemm(this->m_cublasHandle,
	                     CUBLAS_OP_T, CUBLAS_OP_N,
	                     this->m_size_out,  // m: number of rows of matrix op(A) and C.
	                     this->m_numPatterns,  // n: number of columns of matrix op(B) and C.
	                     this->m_size_in,  // k: number of columns of op(A) and rows of op(B).
	                     &alpha,
	                     this->mp_d_weightsData,  // A: dims: lda x k -> size_out x size_in
	                     this->m_size_in,  // lda
	                     this->mep_d_inputData,  // B: dims: ldb x k -> numPatterns x size_in
	                     this->m_size_in,  // ldb
	                     &beta,
	                     this->mp_d_outputData,  // C: dims: m x n -> size_out x numPatterns
	                     this->m_size_out);
	cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

	// Adding bias
	// y = x * alpha + y
	for (int i = 0; i < this->m_numPatterns; i++) {
		status = cublasSaxpy(this->m_cublasHandle,
		                     this->m_size_out,  // size of vector
		                     &alpha,
		                     this->mp_d_biasData, 1,  // x
		                     this->mp_d_outputData + i * this->m_size_out, 1);  // y
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}
} // forward_gpu float specialization end


// forward_gpu double specialization start
template<>
void FullyConnectedLayer<double>::forward_gpu() const {
	double alpha = 1.0;
	double beta = 0.0;
	cublasStatus_t status;

	// TODO: Doc the column-major-reversedDimension-reversedTransposition setup here.
	//       Did this: Setup every thing like row-major. Then:
	//       * Reverse transpositions
	//       * Reverse lda, ldb with the other dim of their respective matrices.
	status = cublasDgemm(this->m_cublasHandle,
	                     CUBLAS_OP_T, CUBLAS_OP_N,
	                     this->m_size_out,  // m: number of rows of matrix op(A) and C.
	                     this->m_numPatterns,  // n: number of columns of matrix op(B) and C.
	                     this->m_size_in,  // k: number of columns of op(A) and rows of op(B).
	                     &alpha,
	                     this->mp_d_weightsData,  // A: dims: lda x k -> size_out x size_in
	                     this->m_size_in,  // lda
	                     this->mep_d_inputData,  // B: dims: ldb x k -> numPatterns x size_in
	                     this->m_size_in,  // ldb
	                     &beta,
	                     this->mp_d_outputData,  // C: dims: m x n -> size_out x numPatterns
	                     this->m_size_out);
	cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

	// Adding bias
	// y = x * alpha + y
	for (int i = 0; i < this->m_numPatterns; i++) {
		status = cublasDaxpy(this->m_cublasHandle,
		                     this->m_size_out,  // size of vector
		                     &alpha,
		                     this->mp_d_biasData, 1,  // x
		                     this->mp_d_outputData + i * this->m_size_out, 1);  // y
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}
} // forward_gpu double specialization end


// backward_gpu float specialization start
template<>
void FullyConnectedLayer<float>::backward_gpu() const {
	cublasStatus_t status;
	float alpha = 1.0;
	float beta = 0.0;

	// prevDiff
	// delta_i = w_{ji}^T * delta_{j}^T
	// TODO: Doc the column-major-reversedDimension-reversedTransposition setup here.
	//       Did this: Setup every thing like row-major. Then:
	//       * Reverse transpositions
	//       * Reverse lda, ldb with the other dim of their respective matrices.
	status = cublasSgemm(this->m_cublasHandle,
	                     CUBLAS_OP_N, CUBLAS_OP_N,
	                     this->m_size_in,  // m: number of rows of matrix op(A) and C.
	                     this->m_numPatterns,  // n: number of columns of matrix op(B) and C.
	                     this->m_size_out,  // k: number of columns of op(A) and rows of op(B).
	                     &alpha,
	                     this->mp_d_weightsData,  // A: dims: lda x k -> size_out x size_in
	                     this->m_size_in,  // lda
	                     this->mp_d_diffDataOut,  // B: dims: ldb x k -> numPatterns x size_out
	                     this->m_size_out,  // ldb
	                     &beta,
	                     this->mep_d_prevDiffData,  // C: dims: m x n -> size_in x numPatterns
	                     this->m_size_in);
	cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

	// Weight diff
	// w_{ji} = delta_j * z_i^T
	// TODO: Doc the column-major-reversedDimension-reversedTransposition setup here.
	//       Did this: Setup every thing like row-major. Then:
	//       * Keep transpositions.
	//       * reverse m and n.
	//       * Switch A with B and lda with ldb.
	//       * reverse ldc with its counterpart.
	// w_{ji} = z_i * delta_j^T
	// Batched eval. We start by setting the destination to zero.
	float zero = 0.0;
	cudnnStatus_t cudnn_status;
	cudnn_status = cudnnSetTensor(this->m_handle, this->m_weightsTens, this->mp_d_weightsDiffData, &zero);
	cuda_status_check(cudnn_status, __FILE__, __FUNCTION__, __LINE__);
	int delta_offset = 0;
	int z_offset = 0;
	alpha = 1.0;
	beta = 1.0;  // beta = 1.0 adds the result onto the existing destination data.
	// Iterate through the patterns
	for (int i = 0; i < this->m_numPatterns; i++) {
		delta_offset = i * this->m_size_out;
		z_offset = i * this->m_size_in;
		status = cublasSgemm(this->m_cublasHandle,
		                     CUBLAS_OP_N, CUBLAS_OP_T,
		                     this->m_size_in,  // n: number of columns of matrix op(B) and C.
		                     this->m_size_out,  // m: number of rows of matrix op(A) and C.
		                     1,  // k: number of columns of op(A) and rows of op(B).
		                     &alpha,
		                     this->mep_d_inputData + z_offset,  // B: dims: ldb x k -> size_in x 1
		                     this->m_size_in,  // ldb
		                     this->mp_d_diffDataOut + delta_offset,  // A: dims: lda x k -> size_out x 1
		                     this->m_size_out,  // lda
		                     &beta,
		                     this->mp_d_weightsDiffData,  // C: dims: m x n -> size_out x size_in
		                     this->m_size_in);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		// Bias diff
		status = cublasSaxpy(this->m_cublasHandle,
		                     this->m_size_out,  // size of vector
		                     &alpha,
		                     this->mp_d_diffDataOut + delta_offset, 1,  // x
		                     this->mp_d_biasDiffData, 1);  // y
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}
}  // backward_gpu float specialization end

// backward_gpu double specialization start
template<>
void FullyConnectedLayer<double>::backward_gpu() const {
	typedef double Precision;
	Precision alpha = 1.0;
	Precision beta = 0.0;
	cublasStatus_t status;

	// prevDiff
	// delta_i = w_{ji}^T * delta_{j}^T
	// TODO: Doc the column-major-reversedDimension-reversedTransposition setup here.
	//       Did this: Setup every thing like row-major. Then:
	//       * Reverse transpositions
	//       * Reverse lda, ldb with the other dim of their respective matrices.
	status = cublasDgemm(this->m_cublasHandle,
	                     CUBLAS_OP_N, CUBLAS_OP_N,
	                     this->m_size_in,  // m: number of rows of matrix op(A) and C.
	                     this->m_numPatterns,  // n: number of columns of matrix op(B) and C.
	                     this->m_size_out,  // k: number of columns of op(A) and rows of op(B).
	                     &alpha,
	                     this->mp_d_weightsData,  // A: dims: lda x k -> size_out x size_in
	                     this->m_size_in,  // lda
	                     this->mp_d_diffDataOut,  // B: dims: ldb x k -> numPatterns x size_out
	                     this->m_size_out,  // ldb
	                     &beta,
	                     this->mep_d_prevDiffData,  // C: dims: m x n -> size_in x numPatterns
	                     this->m_size_in);
	cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

	// Weight diff
	// w_{ji} = delta_j * z_i^T
	// TODO: Doc the column-major-reversedDimension-reversedTransposition setup here.
	//       Did this: Setup every thing like row-major. Then:
	//       * Keep transpositions.
	//       * reverse m and n.
	//       * Switch A with B and lda with ldb.
	//       * reverse ldc with its counterpart.
	// w_{ji} = z_i * delta_j^T
	alpha = 1.0;
	beta = 0.0;  // beta = 1.0 adds the result onto the existing destination data.
	status = cublasDgemm(this->m_cublasHandle,
	                     CUBLAS_OP_N, CUBLAS_OP_T,
	                     this->m_size_in,  // m: number of rows of matrix op(A) and C.
	                     this->m_size_out,  // n: number of columns of matrix op(B) and C.
	                     this->m_numPatterns,  // k: number of columns of op(A) and rows of op(B).
	                     &alpha,
	                     this->mep_d_inputData,  // A: dims: lda x k -> size_in x n
	                     this->m_size_in,  // lda
	                     this->mp_d_diffDataOut,  // B: dims: ldb x k -> size_out x n
	                     this->m_size_out,  // ldb
	                     &beta,
	                     this->mp_d_weightsDiffData,  // C: dims: m x n -> size_in x size_out
	                     this->m_size_in);
	cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

	// Bias diff
	// b_j = \sum_n \delta_j^n => b = \delta^n * ones
	status = cublasDgemv(this->m_cublasHandle,
	                     CUBLAS_OP_N,
	                     this->m_size_out, this->m_numPatterns,
	                     &alpha,
	                     this->mp_d_diffDataOut, this->m_size_out,
	                     mp_d_biasOnes, 1,
	                     &beta,
	                     this->mp_d_biasDiffData, 1);
	cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
}  // backward_gpu double specialization end


}  // end namespace shark
#endif
