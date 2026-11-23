// Written by Rockfish, not copied from AMD. AMD's CMake build generates
// rocsparse-export.h; it is not in the repository. bindgen only needs the macro
// to resolve (to nothing), and no symbol visibility reaches the output.
#pragma once
#define ROCSPARSE_EXPORT
#define ROCSPARSE_NO_EXPORT
