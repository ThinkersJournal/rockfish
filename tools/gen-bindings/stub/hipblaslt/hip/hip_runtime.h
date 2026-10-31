// Written by Rockfish, not copied from AMD. The real hip_runtime.h pulls in the
// device compiler headers; the host API needs only the runtime API and the
// library data types (hipDataType), which AMD's hip_runtime.h also includes.
#pragma once
#include <hip/hip_runtime_api.h>
#include <hip/library_types.h>
