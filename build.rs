use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let profile = std::env::var("PROFILE").unwrap();

    let shaders_out = manifest_dir
        .join("target")
        .join(&profile)
        .join("examples")
        .join("shaders");

    std::fs::create_dir_all(&shaders_out).expect("failed to create shader output directory");

    let examples_dir = manifest_dir.join("examples");

    // Rerun if a new example directory is added.
    println!("cargo:rerun-if-changed={}", examples_dir.display());

    for entry in std::fs::read_dir(&examples_dir).expect("failed to read examples dir") {
        let example_dir = entry.unwrap().path();
        if !example_dir.is_dir() {
            continue;
        }

        let shaders_src = example_dir.join("shaders");
        if !shaders_src.is_dir() {
            continue;
        }

        // Watch the directory itself so that adding a new .glsl file triggers a rebuild.
        println!("cargo:rerun-if-changed={}", shaders_src.display());

        for shader_entry in std::fs::read_dir(&shaders_src).unwrap() {
            let shader_path = shader_entry.unwrap().path();

            if shader_path.extension().and_then(|e| e.to_str()) != Some("glsl") {
                continue;
            }

            println!("cargo:rerun-if-changed={}", shader_path.display());

            let stem = shader_path.file_stem().unwrap().to_str().unwrap();
            let spv_out = shaders_out.join(format!("{stem}.spv"));

            let stage_ext = Path::new(stem)
                .extension()
                .and_then(|e: &std::ffi::OsStr| e.to_str())
                .unwrap_or_else(|| panic!("cannot infer stage from {}", shader_path.display()));

            let stage = match stage_ext {
                "vert" => "vertex",
                "frag" => "fragment",
                "comp" => "compute",
                "geom" => "geometry",
                "tesc" => "tesscontrol",
                "tese" => "tesseval",
                other => panic!(
                    "unknown shader stage extension '.{other}' in {}",
                    shader_path.display()
                ),
            };

            let status = Command::new("glslc")
                .arg("--target-env=vulkan1.2")
                .arg(format!("-fshader-stage={stage}"))
                .arg(&shader_path)
                .arg("-o")
                .arg(&spv_out)
                .status()
                .expect("glslc not found — install the Vulkan SDK and ensure it is in PATH");

            assert!(
                status.success(),
                "glslc failed for {}",
                shader_path.display()
            );
        }
    }
}
