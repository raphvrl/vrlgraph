use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    let Some(glslc) = which_glslc() else {
        println!("cargo:warning=glslc not found, skipping shader compilation");
        return;
    };

    let shaders = [("egui.vert.glsl", "vertex"), ("egui.frag.glsl", "fragment")];

    for (name, stage) in shaders {
        let src = shaders_dir.join(name);
        let stem = name.strip_suffix(".glsl").unwrap();
        let dst = out_dir.join(format!("{stem}.spv"));

        println!("cargo:rerun-if-changed={}", src.display());

        let status = Command::new(&glslc)
            .arg("--target-env=vulkan1.2")
            .arg(format!("-fshader-stage={stage}"))
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .status()
            .expect("failed to run glslc");

        assert!(status.success(), "glslc failed for {name}");
    }
}

fn which_glslc() -> Option<PathBuf> {
    for name in ["glslc", "glslc.exe"] {
        if Command::new(name).arg("--version").output().is_ok() {
            return Some(PathBuf::from(name));
        }
    }

    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let path = PathBuf::from(sdk).join("Bin").join("glslc.exe");
        if path.exists() {
            return Some(path);
        }
    }

    None
}
