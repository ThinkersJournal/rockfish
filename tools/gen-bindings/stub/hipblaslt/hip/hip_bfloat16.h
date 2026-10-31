// Written by Rockfish, not copied from AMD. hipblaslt-types.h aliases
// hip_bfloat16; the bound functions do not use that alias and the generator
// keeps it out of the output. The layout matches AMD's (one 16-bit field).
#pragma once
typedef struct hip_bfloat16 { unsigned short data; } hip_bfloat16;
