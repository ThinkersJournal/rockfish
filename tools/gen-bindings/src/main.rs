//! Generates `generated.rs` for the HIP `-sys` crates from the vendored headers.
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
    /// Header under `vendor/hip/include`.
    header: &'static str,
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

const HIP: Target = Target {
    krate: "rockfish-hip-sys",
    header: "hip/hip_runtime_api.h",
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
    header: "hip/hiprtc.h",
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
        other => panic!("usage: gen-bindings (hip|hiprtc); got `{other}`"),
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
    let include = root.join("vendor/hip/include");
    let stub = root.join("tools/gen-bindings/stub");
    let mut builder = bindgen::Builder::default()
        .header(include.join(t.header).to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .clang_arg(format!("-I{}", stub.display()))
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        .layout_tests(true)
        .allowlist_type(t.extra_types)
        .allowlist_var(t.extra_types);
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
    let mut aliases = Vec::new();
    let mut fields = Vec::new();
    let mut loads = Vec::new();
    let mut inits = Vec::new();
    let mut getters = Vec::new();
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

            #(#getters)*
        }
    };

    let parsed: syn::File = syn::parse2(tokens).expect("generated code parses");
    format!(
        "// @generated by tools/gen-bindings from vendor/hip (ROCm/rocm-systems rocm-7.2.4,\n// see THIRD-PARTY-NOTICES.md). Do not edit; rerun the tool.\n\n{}",
        prettyplease::unparse(&parsed)
    )
}
