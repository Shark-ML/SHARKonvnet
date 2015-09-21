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
#ifndef SHARK_MODELS_CUDA_HELPERS_H
#define SHARK_MODELS_CUDA_HELPERS_H


#include "cudnn.h"
#include "cublas_v2.h"

namespace shark {

void cuda_status_check_warn(const char* file,
                            const char* func, const int line);

void cuda_status_check(const char* file,
                       const char* func, const int line);

void cuda_status_check(const cudnnStatus_t status, const char* file,
                       const char* func, const int line);

void cuda_status_check(const cublasStatus_t status, const char* file,
                       const char* func, const int line);

}
#endif
