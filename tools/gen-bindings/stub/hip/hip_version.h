// Written by Rockfish, not copied from AMD. AMD's build generates hip/hip_version.h;
// it is not in the repository. bindgen needs the include to resolve, and the
// generator allowlists no version macro, so these values never reach the output.
#pragma once
#define HIP_VERSION_MAJOR 7
#define HIP_VERSION_MINOR 2
#define HIP_VERSION_PATCH 0
#define HIP_VERSION_GITHASH "rockfish-stub"
#define HIP_VERSION_BUILD_ID 0
#define HIP_VERSION_BUILD_NAME ""
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
