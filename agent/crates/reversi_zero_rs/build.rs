use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let Some(libtorch) = env::var_os("LIBTORCH").map(PathBuf::from) else {
        return;
    };
    let torch_cuda = libtorch.join("lib").join("libtorch_cuda.so");
    if !torch_cuda.is_file() {
        return;
    }

    // tch discovers libtorch_cuda and emits -ltorch_cuda, but GNU ld's
    // --as-needed can discard it because CUDA registers itself through static
    // initializers rather than a directly referenced symbol. Keep the library
    // on the Python extension so Device::cuda_if_available() sees CUDA.
    println!("cargo:rustc-link-arg-cdylib=-Wl,--push-state,--no-as-needed");
    println!("cargo:rustc-link-arg-cdylib={}", torch_cuda.display());
    println!("cargo:rustc-link-arg-cdylib=-Wl,--pop-state");
    println!("cargo:rustc-env=REVERSI_ZERO_CUDA_LINKED=1");
}
