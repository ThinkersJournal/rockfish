# rockfish-hip-emit

The `Hip` backend: lowers Unpopped's neutral `KernelPlan` and kernel-layer `KernelFn` to HIP C++ source text. Text only: no ROCm and no GPU are needed to build or test, and the emitted source has not been through a HIP compiler (UNVERIFIED). Anything it cannot spell exactly is refused by name with a typed `UnsupportedKernel` error; see the workspace `README.md` and `docs/PATENT-SWEEP.md` in the repository for what is declined and why. Not published (`publish = false`).

Licensed under MIT or Apache-2.0, at your option.
