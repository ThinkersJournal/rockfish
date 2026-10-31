// Written by Rockfish, not copied from AMD. AMD's CMake build generates
// rocblas-export.h; it is not in the repository. bindgen only needs the macro
// to resolve (to nothing), and no symbol visibility reaches the output.
#pragma once
#define ROCBLAS_EXPORT
#define ROCBLAS_NO_EXPORT
