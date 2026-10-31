// Written by Rockfish, not copied from AMD. AMD's CMake build generates
// miopen/config.h from config.h.in; only the backend selection matters to
// miopen.h. This selects the HIP backend, which is the one Rockfish binds.
#pragma once
#define MIOPEN_BACKEND_OPENCL 0
#define MIOPEN_BACKEND_HIP 1
