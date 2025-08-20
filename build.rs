use std::{
    io,
    path::Path,
    process::{Command, Stdio},
};

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=./shaders/shader.slang");
    let shader_path = "./shaders/shader.slang";
    let out_dir = "./shaders/out";

    let output_path = Path::new(&out_dir).join("slang.spv");
    let output = Command::new("slangc")
        .arg(shader_path)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("spirv_1_4")
        .arg("-emit-spirv-directly")
        .arg("-fvk-use-entrypoint-name")
        .arg("-entry")
        .arg("vertMain")
        .arg("-entry")
        .arg("fragMain")
        .arg("-o")
        .arg(&output_path)
        .stderr(Stdio::inherit())
        .output()?;
    if !output.status.success() {
        eprintln!(
            "failed to compile shaders: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("build.rs finished");
    Ok(())
}
