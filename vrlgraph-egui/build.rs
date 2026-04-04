use std::path::{Path, PathBuf};
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

    compile_example_shaders(&manifest_dir, &glslc);
}

fn compile_example_shaders(manifest_dir: &Path, glslc: &Path) {
    let examples_dir = manifest_dir.join("examples");
    if !examples_dir.is_dir() {
        return;
    }

    let profile = std::env::var("PROFILE").unwrap();
    let shaders_out = manifest_dir
        .ancestors()
        .find(|p| p.join("Cargo.lock").exists())
        .unwrap_or(manifest_dir)
        .join("target")
        .join(&profile)
        .join("examples")
        .join("shaders");

    let mut needs_compile = false;

    for entry in std::fs::read_dir(&examples_dir).expect("failed to read examples dir") {
        let example_dir = entry.unwrap().path();
        if !example_dir.is_dir() {
            continue;
        }
        let shaders_src = example_dir.join("shaders");
        if !shaders_src.is_dir() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", shaders_src.display());
        for shader_entry in std::fs::read_dir(&shaders_src).unwrap() {
            let shader_path = shader_entry.unwrap().path();
            if shader_path.extension().and_then(|e| e.to_str()) != Some("glsl") {
                continue;
            }
            println!("cargo:rerun-if-changed={}", shader_path.display());
            needs_compile = true;
        }
    }

    if !needs_compile {
        return;
    }

    std::fs::create_dir_all(&shaders_out).expect("failed to create shader output directory");

    for entry in std::fs::read_dir(&examples_dir).unwrap() {
        let example_dir = entry.unwrap().path();
        if !example_dir.is_dir() {
            continue;
        }
        let shaders_src = example_dir.join("shaders");
        if !shaders_src.is_dir() {
            continue;
        }
        for shader_entry in std::fs::read_dir(&shaders_src).unwrap() {
            let shader_path = shader_entry.unwrap().path();
            if shader_path.extension().and_then(|e| e.to_str()) != Some("glsl") {
                continue;
            }
            let stem = shader_path.file_stem().unwrap().to_str().unwrap();
            let spv_out = shaders_out.join(format!("{stem}.spv"));
            let stage_ext = Path::new(stem)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_else(|| panic!("cannot infer stage from {}", shader_path.display()));
            let stage = match stage_ext {
                "vert" => "vertex",
                "frag" => "fragment",
                "comp" => "compute",
                other => panic!("unknown shader stage '.{other}' in {}", shader_path.display()),
            };
            let status = Command::new(glslc)
                .arg("--target-env=vulkan1.2")
                .arg(format!("-fshader-stage={stage}"))
                .arg(&shader_path)
                .arg("-o")
                .arg(&spv_out)
                .status()
                .expect("failed to run glslc");
            assert!(status.success(), "glslc failed for {}", shader_path.display());
        }
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
