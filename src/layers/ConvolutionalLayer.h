/*!
 *
 *
 * \brief       Implements a convolutional layer on nvidia gpu's.
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
#ifndef SHARK_MODELS_CONVOLUTIONAL_LAYER_H
#define SHARK_MODELS_CONVOLUTIONAL_LAYER_H

#include "cudnn.h"
#include "shark_cuda_helpers.h"
#include "AbstractLayer.h"
#include "AbstractWeightedLayer.h"

namespace shark {

//! \brief Provides a ConvolutionalLayer that can be used anywhere within a network.
//!
//! It is required that the inputs to this layer are equal to or larger than the
//! kernels specified. This is checked at runtime.
//!
//! Takes the number of output-planes and a kernel-size as parameters. Other
//! parameters with default values are padding, and striding. This layer also
//! applies a feature-map-wise bias parameter to its output. Notice there are no
//! activation function applied to the output. It is up to the user to decide
//! wether to follow this layer with an ActivationLayer, implement a custom
//! kernel, or do nothing.
template<typename Precision>
class ConvolutionalLayer: public AbstractWeightedLayer<Precision> {
public:
	ConvolutionalLayer(int feats_n, int feats_h, int feats_w,
	                   int pad_h=0, int pad_w=0,
	                   int stride_h=1, int stride_w=1,
	                   int upscale_h=1, int upscale_w=1,
	                   std::string layer_name = "ConvolutionalLayer")
		: AbstractWeightedLayer<Precision> (layer_name) {
		SIZE_CHECK(feats_n   > 0);
		SIZE_CHECK(feats_h   > 0);
		SIZE_CHECK(feats_w   > 0);
		SIZE_CHECK(pad_h     >= 0);
		SIZE_CHECK(pad_w     >= 0);
		SIZE_CHECK(stride_h  > 0);
		SIZE_CHECK(stride_w  > 0);
		SIZE_CHECK(upscale_h > 0);
		SIZE_CHECK(upscale_w > 0);
		this->m_feats_n   = feats_n;
		this->m_feats_h   = feats_h;
		this->m_feats_w   = feats_w;
		this->m_pad_h     = pad_h;
		this->m_pad_w     = pad_w;
		this->m_stride_h  = stride_h;
		this->m_stride_w  = stride_w;
		this->m_upscale_h = upscale_h;
		this->m_upscale_w = upscale_w;
		m_workspaceSize = 0;
		m_workspaceSize_old = 0;
		m_setup = true;
	}

	~ConvolutionalLayer () {
		// Only free the parts that this layer is responsible for.
		cudnnDestroyFilterDescriptor(m_filterDesc);
		cudnnDestroyConvolutionDescriptor(m_convDesc);
		// Warn of errors when freeing/destroying.
		cuda_status_check_warn(__FILE__, __FUNCTION__, __LINE__);
	}

	void init(cuHandles handles, cudaStream_t stream) {
		this->m_handle = handles.cudnnHandle;
		this->m_stream = stream;
		cudnnStatus_t  status;
		status = cudnnCreateFilterDescriptor(&m_filterDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		status = cudnnCreateConvolutionDescriptor(&m_convDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		status = cudnnCreateTensorDescriptor(&this->m_outputDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		status = cudnnCreateTensorDescriptor(&this->m_biasDesc);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void prepare_gpu() {
		cudnnStatus_t status;
		SIZE_CHECK(m_feats_h <= this->m_height + m_pad_h);
		SIZE_CHECK(m_feats_w <= this->m_width + m_pad_w);
		SIZE_CHECK(this->m_channels> 0);

		this->m_channels_out = m_feats_n;


		// Convolution descriptor start
		status = cudnnSetConvolution2dDescriptor(m_convDesc,
		                                         m_pad_h, m_pad_w,  // 0-pad the input img
		                                         m_stride_h, m_stride_w, // stride: vertical, horizontal
		                                         m_upscale_h, m_upscale_w,  // upscale: x-dir,y-dir
		                                         CUDNN_CROSS_CORRELATION);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		// Convolution descriptor end

		// Filter descriptor start
		status = cudnnSetFilter4dDescriptor(m_filterDesc,
		                                    this->m_cudnn_prec,
		                                    this->m_channels_out,  // output feature maps
		                                    this->m_channels, // input feature maps
		                                    m_feats_h, m_feats_w);  // m_height, width of each filter
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		// Filter descriptor end
		// std::cout << "set filter to " << this->m_channels_out << ", " << this->m_channels<< std::endl;

		int n_out, c_out;
		status = cudnnGetConvolution2dForwardOutputDim(m_convDesc, this->m_inputDesc,
		                                               m_filterDesc,
		                                               &n_out,  // # output imgs
		                                               &this->m_channels_out,  // # output feat maps per img
		                                               &this->m_height_out, &this->m_width_out);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		// std::cout << "this->m_channels_out " << this->m_channels_out << std::endl;
		// /* Set and allocate output tensor descriptor */
		status = cudnnSetTensor4dDescriptor(this->m_outputDesc,
		                                    CUDNN_TENSOR_NCHW,  // Data-layout
		                                    this->m_cudnn_prec,  // Data-type
		                                    this->m_numPatterns,  // Number of input images
		                                    this->m_channels_out,  // number of feature maps
		                                    this->m_height_out, this->m_width_out);  // Feature map m_height x width
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		// Output descriptor end
		this->m_size_out = this->m_channels_out * this->m_height_out * this->m_width_out;

		// TODO: Sanity check of out-sizes of all layers.
		// TODO: Sort of bad misuse of SIZE_CHECK. Oswin?
		// SIZE_CHECK(samples_out > 0 && "Convnet structure invalid. Output dims must be > 0");
		SIZE_CHECK(this->m_channels_out > 0 && "Convnet structure invalid. Output dims must be > 0");
		SIZE_CHECK(this->m_height_out > 0 && "Convnet structure invalid. Output dims must be > 0");
		SIZE_CHECK(this->m_width_out > 0 && "Convnet structure invalid. Output dims must be > 0");


		status = cudnnGetConvolutionForwardAlgorithm(this->m_handle,
		                                             this->m_inputDesc,
		                                             m_filterDesc,
		                                             m_convDesc,
		                                             this->m_outputDesc,
		                                             CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		                                             0,
		                                             &m_algo);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		// workspacesize in bytes
		m_workspaceSize_old = m_workspaceSize;
		m_workspaceSize = 0;
		mp_d_workspace = NULL;
		status = cudnnGetConvolutionForwardWorkspaceSize(this->m_handle,
		                                                 this->m_inputDesc,
		                                                 m_filterDesc,
		                                                 m_convDesc,
		                                                 this->m_outputDesc,
		                                                 m_algo,
		                                                 &m_workspaceSize);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

		// Bias descriptor start
		status = cudnnSetTensor4dDescriptor(this->m_biasDesc,
		                                    CUDNN_TENSOR_NCHW,  // Data-layout
		                                    this->m_cudnn_prec,  // Data-type
		                                    1,  // Number of input images
		                                    this->m_channels_out,  // number of feature maps
		                                    1,  // Feature map m_height
		                                    1);  // Feature map width
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
		this->m_numBias = this->m_channels_out;
		// Bias descriptor end

		// Malloc
		if (m_setup) {
			m_bias_offset = this->m_channels * m_feats_n * m_feats_w * m_feats_h;
			// std::cout << "allocing # weights " << m_bias_offset << std::endl;
			this->m_numWeights = m_bias_offset +  // filter-weights
			                     this->m_channels_out;  // bias-weights
			// this->m_numWeights = this->m_channels_out * m_feats_w * m_feats_h;
			cudaMalloc(&this->mp_d_weightsData, this->m_numWeights * sizeof(Precision));
			cudaMalloc(&this->mp_d_weightsDiffData, this->m_numWeights * sizeof(Precision));
			this->mp_d_biasData = this->mp_d_weightsData + m_bias_offset;
			m_setup = false;
		}


		if (m_workspaceSize > 0) {
			if (m_workspaceSize > m_workspaceSize_old && m_workspaceSize_old > 0) {
				cudaFree(mp_d_workspace);
			}
			cudaMalloc(&mp_d_workspace, m_workspaceSize);
		}

		if (this->m_numPatterns_old < this->m_numPatterns) {
			if (this->m_numPatterns_old != 0) {
				cudaFree(this->mp_d_outputData);
				cudaFree(this->mp_d_diffDataOut);
			}
			size_t batchSize = this->m_numPatterns * this->m_size_out * sizeof(Precision);
			cudaMalloc(&this->mp_d_outputData, batchSize);
			cudaMalloc(&this->mp_d_diffDataOut, batchSize);
			// cudaMalloc(&this->mp_d_biasData, this->m_size_out * sizeof(Precision));
			// TODO: Think about overwriting weightsData with the gradients to save mem. The layer will be "invalid" until new weights are set.
			cuda_status_check(__FILE__, __FUNCTION__, __LINE__);
			this->m_gpu_mem = m_workspaceSize + 2 * batchSize + 2 * this->m_numWeights * sizeof(Precision);
			// Malloc end
		}
	}

	void forward_gpu() const {
		cudnnStatus_t status;
		Precision alpha = 1.0;
		Precision beta = 0.0;

		// Convolution
		status = cudnnConvolutionForward(this->m_handle,
		                                 &alpha,  // alpha. Multiplied onto every input element
		                                 this->m_inputDesc, this->mep_d_inputData,
		                                 m_filterDesc, this->mp_d_weightsData,
		                                 m_convDesc,
		                                 m_algo,  // algorithm to use
		                                 mp_d_workspace,  // pointer to workspace if required by algoithm
		                                 m_workspaceSize,  // Size of workspace
		                                 &beta, // beta. Multiplied onto every elem of the output tensor
		                                 // prior to adding the result of this convolution.
		                                 this->m_outputDesc, this->mp_d_outputData);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

		// Bias
		beta = 1.0;
		cudnnAddTensor(this->m_handle,
		               CUDNN_ADD_SAME_C,
		               &alpha,
		               this->m_biasDesc, this->mp_d_biasData,
		               &beta,
		               this->m_outputDesc, this->mp_d_outputData);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	void backward_gpu() const {
		cudnnStatus_t status;
		Precision alpha = 1.0;
		Precision beta = 0.0;

		// Bias
		status = cudnnConvolutionBackwardBias(this->m_handle,
		                                      &alpha,
		                                      this->m_outputDesc, this->mp_d_diffDataOut,
		                                      &beta,
		                                      this->m_biasDesc, this->mp_d_weightsDiffData + m_bias_offset);
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

		// Calculate the error with respect to the weights
		alpha = 1.0;
		beta = 0.0;
		status = cudnnConvolutionBackwardFilter(this->m_handle,
		                                        &alpha,  // const
		                                        this->m_inputDesc, this->mep_d_inputData,  // const srcDesc, const *srcData,
		                                        this->m_outputDesc, this->mp_d_diffDataOut,  // const diffDesc, const *diffData,
		                                        m_convDesc,  // const m_convDesc,
		                                        &beta,  // const
		                                        m_filterDesc, this->mp_d_weightsDiffData);  // const filterdescriptor, *gradData
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);

		// Calculate the error for the previous layer
		alpha = 1.0;
		beta = 0.0;
		status = cudnnConvolutionBackwardData(this->m_handle,
		                                      &alpha,  // const
		                                      m_filterDesc, this->mp_d_weightsData,  // const filterDesc, const filterData
		                                      this->m_outputDesc, this->mp_d_diffDataOut,  // const diffDesc, const diffData
		                                      m_convDesc,  // const m_convDesc
		                                      &beta,  // const
		                                      this->m_inputDesc, this->mep_d_prevDiffData);  // const gradDesc, gradData
		cuda_status_check(status, __FILE__, __FUNCTION__, __LINE__);
	}

	//! From ISerializable, reads a model from an archive
	void read( InArchive & archive ) {
		archive>>m_feats_n;
		archive>>m_feats_h;
		archive>>m_feats_w;
		archive>>m_pad_h;
		archive>>m_pad_w;
		archive>>m_stride_h;
		archive>>m_stride_w;
		archive>>m_upscale_h;
		archive>>m_upscale_w;
	}

	//! From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const {
		archive<<m_feats_n;
		archive<<m_feats_h;
		archive<<m_feats_w;
		archive<<m_pad_h;
		archive<<m_pad_w;
		archive<<m_stride_h;
		archive<<m_stride_w;
		archive<<m_upscale_h;
		archive<<m_upscale_w;
	}

	bool m_setup;

	// Desciption of the convolutions in this layer
	int m_feats_n;  // Number of feature maps
	int m_feats_h;  // Height of each feature map
	int m_feats_w;  // Width of each feature map
	int m_pad_h;  // Size of zero-padding of input before convolution
	int m_pad_w;
	int m_stride_h; // Striding of the filter over the input during the convolution
	int m_stride_w;
	int m_upscale_h; // Upscaling of input
	int m_upscale_w;

	// cuDNN:
	/* cudaMalloc'ed in prep */
	cudnnFilterDescriptor_t      m_filterDesc;
	cudnnConvolutionDescriptor_t m_convDesc;

	size_t m_bias_offset;

	size_t                    m_workspaceSize;
	size_t                    m_workspaceSize_old;
	Precision                 *mp_d_workspace;
	cudnnConvolutionFwdAlgo_t m_algo;

};  // end class ConvolutionalLayer

}  // end namespace shark
#endif