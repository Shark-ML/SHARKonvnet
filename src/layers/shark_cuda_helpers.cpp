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

#include <string>
#include "cudnn.h"
#include "cublas_v2.h"

#include "shark_cuda_helpers.h"

namespace shark {

void cuda_status_check_warn(const char* file,
                            const char* func, const int line) {
	cudaError_t status = cudaGetLastError();

	if (status == cudaSuccess)
		return;

	printf("\e[31mWarning: %s in file %s:%i - %s(...)\e[39m\n",
	       cudaGetErrorString(status), file, line, func);
}

void cuda_status_check(const char* file,
                       const char* func, const int line) {
	cudaError_t status = cudaGetLastError();

	if (status == cudaSuccess)
		return;

	printf("\e[31m%s in file %s:%i - %s(...)\e[39m\n",
	       cudaGetErrorString(status), file, line, func);
	throw status;
}

void cuda_status_check(const cudnnStatus_t status, const char* file,
                       const char* func, const int line) {
	if (status == CUDNN_STATUS_SUCCESS)
		return;

	printf("\e[31m%s in file %s:%i - %s(...)\e[39m\n",
	       cudnnGetErrorString(status), file, line, func);
	throw status;
}

void cuda_status_check(const cublasStatus_t status, const char* file,
                       const char* func, const int line) {
	if (status == CUBLAS_STATUS_SUCCESS)
		return;

	std::string errString = "Unknown cuBlas error";
	if (status == CUBLAS_STATUS_NOT_INITIALIZED)
		errString = "CUBLAS_STATUS_NOT_INITIALIZED";
	if (status == CUBLAS_STATUS_ALLOC_FAILED)
		errString = "CUBLAS_STATUS_ALLOC_FAILED";
	if (status == CUBLAS_STATUS_INVALID_VALUE)
		errString = "CUBLAS_STATUS_INVALID_VALUE";
	if (status == CUBLAS_STATUS_ARCH_MISMATCH)
		errString = "CUBLAS_STATUS_ARCH_MISMATCH";
	if (status == CUBLAS_STATUS_MAPPING_ERROR)
		errString = "CUBLAS_STATUS_MAPPING_ERROR";
	if (status == CUBLAS_STATUS_EXECUTION_FAILED)
		errString = "CUBLAS_STATUS_EXECUTION_FAILED";
	if (status == CUBLAS_STATUS_INTERNAL_ERROR)
		errString = "CUBLAS_STATUS_INTERNAL_ERROR";
	if (status == CUBLAS_STATUS_NOT_SUPPORTED)
		errString = "CUBLAS_STATUS_NOT_SUPPORTED";
	if (status == CUBLAS_STATUS_LICENSE_ERROR)
		errString = "CUBLAS_STATUS_LICENSE_ERROR";

	printf("\e[31m%s in file %s:%i - %s(...)\e[39m\n",
	       errString.c_str(), file, line, func);
	throw status;
}

}
