//! Generates `generated.rs` for the `-sys` crates (HIP, hipRTC, rocBLAS, hipBLASLt) from the
//! vendored headers.
//!
//! bindgen produces the types and a plain `extern "C"` block; this tool removes
//! the block and replaces it with a table of optional function pointers filled
//! in through `rockfish_core::Library`, so a missing symbol is an error for that
//! call and never a link failure or a panic.

use std::path::{Path, PathBuf};

use quote::{format_ident, quote};
use syn::{Expr, ExprLit, FnArg, ForeignItem, Item, Lit, Pat, ReturnType};

struct Target {
    /// Crate directory under `crates/`.
    krate: &'static str,
    /// Directory the header is read from and `-I`-ed, relative to the repo root.
    include: &'static str,
    /// Header under `include`.
    header: &'static str,
    /// Stub directories (relative to `tools/gen-bindings/stub`) added with `-I`.
    stubs: &'static [&'static str],
    /// Header force-included before `header`, for headers that assume another was included first.
    preinclude: Option<&'static str>,
    /// Types provided by another `-sys` crate: (type, `use` path). Kept out of the output.
    reused: &'static [(&'static str, &'static str)],
    /// Parse the header as C++ (its API is `extern "C"`, but the file is not valid C).
    cxx: bool,
    /// Provenance written at the top of the generated file.
    origin: &'static str,
    /// Name of the generated table type.
    table: &'static str,
    /// Name of the `LibrarySpec` constant in the crate's `lib.rs`.
    spec: &'static str,
    /// Prefix used in error messages (the `LibrarySpec::logical_name`).
    library: &'static str,
    /// Functions to bind. Anything else in the header is ignored.
    functions: &'static [&'static str],
    /// bindgen regex for types and constants kept besides those the functions need.
    extra_types: &'static str,
}

const HIP_ORIGIN: &str = "vendor/hip (ROCm/rocm-systems rocm-7.2.4";

const ROCBLAS: Target = Target {
    krate: "rockfish-rocblas-sys",
    include: "vendor/rocblas/include/rocblas",
    header: "rocblas.h",
    stubs: &[".", "rocblas", "rocblas/internal"],
    preinclude: Some("hip/hip_runtime_api.h"),
    reused: &[("hipStream_t", "rockfish_hip_sys::hipStream_t")],
    cxx: false,
    origin: "vendor/rocblas (ROCm/rocm-libraries rocm-7.2.4",
    table: "RocblasApi",
    spec: "ROCBLAS",
    library: "ROCBLAS",
    extra_types: "^rocblas_.*$",
    functions: &[
        "rocblas_create_handle",
        "rocblas_destroy_handle",
        "rocblas_set_stream",
        "rocblas_get_stream",
        "rocblas_set_pointer_mode",
        "rocblas_get_pointer_mode",
        "rocblas_set_atomics_mode",
        "rocblas_get_atomics_mode",
        "rocblas_set_math_mode",
        "rocblas_get_math_mode",
        "rocblas_get_version_string",
        "rocblas_get_version_string_size",
        "rocblas_status_to_string",
        "rocblas_initialize",
        "rocblas_set_vector",
        "rocblas_get_vector",
        "rocblas_set_matrix",
        "rocblas_get_matrix",
        "rocblas_set_vector_async",
        "rocblas_get_vector_async",
        "rocblas_set_matrix_async",
        "rocblas_get_matrix_async",
        "rocblas_sscal",
        "rocblas_dscal",
        "rocblas_saxpy",
        "rocblas_daxpy",
        "rocblas_sdot",
        "rocblas_ddot",
        "rocblas_snrm2",
        "rocblas_dnrm2",
        "rocblas_sasum",
        "rocblas_dasum",
        "rocblas_scopy",
        "rocblas_dcopy",
        "rocblas_sswap",
        "rocblas_dswap",
        "rocblas_isamax",
        "rocblas_idamax",
        "rocblas_isamin",
        "rocblas_idamin",
        "rocblas_sgemv",
        "rocblas_dgemv",
        "rocblas_hgemm",
        "rocblas_sgemm",
        "rocblas_dgemm",
        "rocblas_hgemm_batched",
        "rocblas_hgemm_strided_batched",
        "rocblas_sgemm_batched",
        "rocblas_sgemm_strided_batched",
        "rocblas_dgemm_batched",
        "rocblas_dgemm_strided_batched",
        "rocblas_gemm_ex",
        "rocblas_gemm_batched_ex",
        "rocblas_gemm_strided_batched_ex",
        "rocblas_sgeam",
        "rocblas_dgeam",
    ],
};

const HIPBLASLT: Target = Target {
    krate: "rockfish-hipblaslt-sys",
    include: "vendor/hipblaslt/include",
    header: "hipblaslt/hipblaslt.h",
    stubs: &[".", "hipblaslt"],
    preinclude: None,
    reused: &[("hipStream_t", "rockfish_hip_sys::hipStream_t")],
    cxx: true,
    origin: "vendor/hipblaslt (ROCm/rocm-libraries rocm-7.2.4",
    table: "HipblasltApi",
    spec: "HIPBLASLT",
    library: "HIPBLASLT",
    extra_types: "^(hipblas[A-Za-z0-9]*_t|hipDataType)$",
    functions: &[
        "hipblasLtGetVersion",
        "hipblasLtGetGitRevision",
        "hipblasLtGetArchName",
        "hipblasLtCreate",
        "hipblasLtDestroy",
        "hipblasLtMatrixLayoutCreate",
        "hipblasLtMatrixLayoutDestroy",
        "hipblasLtMatrixLayoutSetAttribute",
        "hipblasLtMatrixLayoutGetAttribute",
        "hipblasLtMatmulDescCreate",
        "hipblasLtMatmulDescDestroy",
        "hipblasLtMatmulDescSetAttribute",
        "hipblasLtMatmulDescGetAttribute",
        "hipblasLtMatmulPreferenceCreate",
        "hipblasLtMatmulPreferenceDestroy",
        "hipblasLtMatmulPreferenceSetAttribute",
        "hipblasLtMatmulPreferenceGetAttribute",
        "hipblasLtMatmulAlgoGetHeuristic",
        "hipblasLtMatmul",
        "hipblasLtMatrixTransformDescCreate",
        "hipblasLtMatrixTransformDescDestroy",
        "hipblasLtMatrixTransformDescSetAttribute",
        "hipblasLtMatrixTransformDescGetAttribute",
        "hipblasLtMatrixTransform",
    ],
};

const HIP: Target = Target {
    krate: "rockfish-hip-sys",
    include: "vendor/hip/include",
    header: "hip/hip_runtime_api.h",
    stubs: &["."],
    preinclude: None,
    reused: &[],
    cxx: false,
    origin: HIP_ORIGIN,
    table: "HipApi",
    spec: "HIP",
    library: "HIP",
    extra_types: "^(hipError_t|hipMemcpyKind|hipDeviceAttribute_t|hipFunction_attribute|hipHostMalloc.*|hipStream.*|hipEvent.*|hipDeviceProp_tR0600|hipUUID)$",
    functions: &[
        "hipInit",
        "hipRuntimeGetVersion",
        "hipDriverGetVersion",
        "hipGetDeviceCount",
        "hipGetDevice",
        "hipSetDevice",
        "hipDeviceGet",
        "hipDeviceGetName",
        "hipDeviceGetAttribute",
        "hipGetDevicePropertiesR0600",
        "hipDeviceSynchronize",
        "hipDeviceReset",
        "hipDeviceCanAccessPeer",
        "hipDeviceEnablePeerAccess",
        "hipDeviceDisablePeerAccess",
        "hipDeviceGetStreamPriorityRange",
        "hipGetLastError",
        "hipPeekAtLastError",
        "hipGetErrorName",
        "hipGetErrorString",
        "hipMalloc",
        "hipFree",
        "hipMallocAsync",
        "hipFreeAsync",
        "hipHostMalloc",
        "hipHostFree",
        "hipMemGetInfo",
        "hipMemcpy",
        "hipMemcpyAsync",
        "hipMemcpyHtoD",
        "hipMemcpyDtoH",
        "hipMemcpyDtoD",
        "hipMemcpyHtoDAsync",
        "hipMemcpyDtoHAsync",
        "hipMemcpyDtoDAsync",
        "hipMemset",
        "hipMemsetAsync",
        "hipMemsetD8",
        "hipMemsetD32",
        "hipStreamCreate",
        "hipStreamCreateWithFlags",
        "hipStreamCreateWithPriority",
        "hipStreamDestroy",
        "hipStreamSynchronize",
        "hipStreamQuery",
        "hipStreamWaitEvent",
        "hipEventCreate",
        "hipEventCreateWithFlags",
        "hipEventDestroy",
        "hipEventRecord",
        "hipEventSynchronize",
        "hipEventQuery",
        "hipEventElapsedTime",
        "hipModuleLoadData",
        "hipModuleLoadDataEx",
        "hipModuleUnload",
        "hipModuleGetFunction",
        "hipModuleGetGlobal",
        "hipModuleLaunchKernel",
        "hipFuncGetAttribute",
    ],
};

const HIPRTC: Target = Target {
    krate: "rockfish-hiprtc-sys",
    include: "vendor/hip/include",
    header: "hip/hiprtc.h",
    stubs: &["."],
    preinclude: None,
    reused: &[],
    cxx: false,
    origin: HIP_ORIGIN,
    table: "HiprtcApi",
    spec: "HIPRTC",
    library: "HIPRTC",
    extra_types: "^(hiprtcResult|hiprtcProgram)$",
    functions: &[
        "hiprtcGetErrorString",
        "hiprtcVersion",
        "hiprtcCreateProgram",
        "hiprtcDestroyProgram",
        "hiprtcCompileProgram",
        "hiprtcGetProgramLogSize",
        "hiprtcGetProgramLog",
        "hiprtcGetCodeSize",
        "hiprtcGetCode",
        "hiprtcAddNameExpression",
        "hiprtcGetLoweredName",
    ],
};

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    let target = match which.as_str() {
        "hip" => &HIP,
        "hiprtc" => &HIPRTC,
        "rocblas" => &ROCBLAS,
        "hipblaslt" => &HIPBLASLT,
        other => panic!("usage: gen-bindings (hip|hiprtc|rocblas|hipblaslt); got `{other}`"),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = root
        .join("crates")
        .join(target.krate)
        .join("src/generated.rs");
    std::fs::write(&out, generate(&root, target)).expect("write generated.rs");
    println!("wrote {}", out.display());
}

fn generate(root: &Path, t: &Target) -> String {
    let include = root.join(t.include);
    let hip_include = root.join("vendor/hip/include");
    let stub = root.join("tools/gen-bindings/stub");
    let mut builder = bindgen::Builder::default()
        .header(include.join(t.header).to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .clang_arg(format!("-I{}", hip_include.display()))
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        .layout_tests(true)
        .allowlist_type(t.extra_types)
        .allowlist_var(t.extra_types);
    for s in t.stubs {
        builder = builder.clang_arg(format!("-I{}", stub.join(s).display()));
    }
    if t.cxx {
        builder = builder
            .clang_arg("-x")
            .clang_arg("c++")
            .clang_arg("-std=c++17");
    }
    if let Some(pre) = t.preinclude {
        builder = builder.clang_arg("-include").clang_arg(pre);
    }
    for (ty, _) in t.reused {
        builder = builder.blocklist_type(format!("^{ty}$"));
    }
    if !t.reused.is_empty() {
        // The pointee of a reused handle type is defined by the crate that owns it.
        builder = builder.blocklist_type("^ihip.*$");
    }
    for f in t.functions {
        builder = builder.allowlist_function(format!("^{f}$"));
    }
    let text = builder.generate().expect("bindgen failed").to_string();

    let mut file: syn::File = syn::parse_str(&text).expect("parse bindgen output");
    let mut decls = Vec::new();
    file.items.retain(|item| {
        if let Item::ForeignMod(block) = item {
            for foreign in &block.items {
                if let ForeignItem::Fn(f) = foreign {
                    decls.push(f.clone());
                }
            }
            return false;
        }
        true
    });

    let found: Vec<String> = decls.iter().map(|d| d.sig.ident.to_string()).collect();
    for wanted in t.functions {
        assert!(
            found.iter().any(|f| f == wanted),
            "`{wanted}` is not declared in {}",
            t.header
        );
    }
    assert_eq!(
        found.len(),
        t.functions.len(),
        "bindgen kept an unlisted function"
    );

    let table = format_ident!("{}", t.table);
    let spec = format_ident!("{}", t.spec);
    let library = t.library;
    let reuse_paths = t.reused.iter().map(|(_, path)| {
        let path: syn::Path = syn::parse_str(path).expect("reuse path parses");
        quote! { use #path; }
    });
    let mut aliases = Vec::new();
    let mut fields = Vec::new();
    let mut loads = Vec::new();
    let mut inits = Vec::new();
    let mut getters = Vec::new();
    let mut unresolved = Vec::new();
    for d in &decls {
        let name = &d.sig.ident;
        let name_str = name.to_string();
        let alias = format_ident!("{}_fn", name);
        let mut arg_names = Vec::new();
        let mut arg_types = Vec::new();
        for input in &d.sig.inputs {
            let FnArg::Typed(pt) = input else {
                panic!("method in extern block")
            };
            let Pat::Ident(id) = &*pt.pat else {
                panic!("pattern argument")
            };
            arg_names.push(id.ident.clone());
            arg_types.push((*pt.ty).clone());
        }
        let ret = match &d.sig.output {
            ReturnType::Default => quote! {},
            ReturnType::Type(arrow, ty) => quote! { #arrow #ty },
        };
        aliases.push(quote! {
            pub type #alias = unsafe extern "C" fn(#(#arg_names: #arg_types),*) #ret;
        });
        fields.push(quote! { #name: Option<#alias> });
        loads.push(quote! {
            // SAFETY: the alias is the signature bindgen read from the vendored header.
            let #name = unsafe { library.symbol::<#alias>(#name_str) }.ok();
        });
        inits.push(quote! { #name });
        unresolved.push(quote! {
            if self.#name.is_none() {
                missing.push(#name_str);
            }
        });
        getters.push(quote! {
            #[doc = concat!("The `", #name_str, "` entry point, or an error if the loaded library does not export it. The pointer is valid while this table is alive.")]
            pub fn #name(&self) -> Result<#alias, LoadError> {
                self.#name.ok_or_else(|| LoadError::SymbolMissing {
                    library: #library,
                    symbol: #name_str.to_owned(),
                    reason: "not exported by the loaded library".to_owned(),
                })
            }
        });
    }

    // First name per `hipError_t` code, so the safe layer can name a status
    // when the loaded library cannot (HIP only; hipRTC has no such constants).
    let mut status_names: Vec<(i32, String)> = Vec::new();
    for item in &file.items {
        let Item::Const(c) = item else { continue };
        let Some(short) = c.ident.to_string().strip_prefix("hipError_t_").map(str::to_owned) else {
            continue;
        };
        let Expr::Lit(ExprLit { lit: Lit::Int(value), .. }) = &*c.expr else {
            continue;
        };
        let code: i32 = value.base10_parse().expect("status code fits i32");
        if !status_names.iter().any(|(known, _)| *known == code) {
            status_names.push((code, short));
        }
    }
    status_names.sort();
    let status_codes = status_names.iter().map(|(c, _)| proc_macro2::Literal::i32_unsuffixed(*c));
    let status_strs = status_names.iter().map(|(_, n)| n);
    let error_names = if status_names.is_empty() {
        quote! {}
    } else {
        quote! {
            /// Name of each `hipError_t` status code in the header (the first
            /// name where several share a code), sorted by code.
            pub const ERROR_NAMES: &[(i32, &str)] = &[#((#status_codes, #status_strs)),*];
        }
    };

    let items = &file.items;
    let tokens = quote! {
        use crate::#spec as SPEC;
        use rockfish_core::{Library, LoadError};
        #(#reuse_paths)*

        #(#items)*

        #error_names

        #(#aliases)*

        /// Function pointers resolved from one loaded library.
        #[derive(Debug)]
        pub struct #table {
            library: Library,
            #(#fields,)*
        }

        impl #table {
            /// Search for and load the library, then resolve every entry point
            /// it exports. Entry points it lacks are reported when asked for.
            pub fn load() -> Result<Self, LoadError> {
                Ok(Self::from_library(Library::open(&SPEC)?))
            }

            /// Resolve the entry points from an already loaded library.
            pub fn from_library(library: Library) -> Self {
                #(#loads)*
                Self { library, #(#inits,)* }
            }

            /// The library the pointers came from.
            pub fn library(&self) -> &Library {
                &self.library
            }

            /// Names of the bound entry points the loaded library does not export.
            pub fn unresolved(&self) -> Vec<&'static str> {
                let mut missing = Vec::new();
                #(#unresolved)*
                missing
            }

            #(#getters)*
        }
    };

    let parsed: syn::File = syn::parse2(tokens).expect("generated code parses");
    format!(
        "// @generated by tools/gen-bindings from {},\n// see THIRD-PARTY-NOTICES.md). Do not edit; rerun the tool.\n\n{}",
        t.origin,
        prettyplease::unparse(&parsed)
    )
}
