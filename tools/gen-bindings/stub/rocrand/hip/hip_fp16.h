// Written by Rockfish, not copied from AMD. rocrand.h only needs `__half` to name
// the half-precision entry points, which this crate does not bind; a 16-bit opaque
// stand-in makes the header parse.
#pragma once
typedef struct { unsigned short data; } __half;
