use std::env;
use std::fs;
use time::{format_description, OffsetDateTime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = format_description::parse_borrowed::<2>("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=src/keccakf1600_x86-64.s");
    tonic_build::configure()
        .build_server(false)
        // .type_attribute(".", "#[derive(Debug)]")
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;
    // PoM mining kernel → PTX set (loaded at runtime into candle's CUDA context).
    // Build a fallback ladder so mixed/older rigs don't fail on a single too-new .target.
    println!("cargo:rerun-if-changed=cuda/pom_mine.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=POM_SM_LIST");
    println!("cargo:rerun-if-env-changed=POM_FATBIN_LEGACY");
    println!("cargo:rerun-if-env-changed=POM_FATBIN_NEXTGEN");
    println!("cargo:rerun-if-changed=cuda/pom_mine_legacy.fatbin");
    println!("cargo:rerun-if-changed=cuda/pom_mine_nextgen.fatbin");
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        let nvcc = env::var("NVCC").ok().unwrap_or_else(|| {
            let pinned = "/home/slash/cuda-12.2/bin/nvcc";
            if std::path::Path::new(pinned).exists() { pinned.to_string() } else { "nvcc".to_string() }
        });
        let sm_list = env::var("POM_SM_LIST").unwrap_or_else(|_| "90,89,86,80,75,70,61".to_string());
        let sms: Vec<String> = sm_list
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        assert!(!sms.is_empty(), "POM_SM_LIST resolved to an empty set");

        for sm in sms {
            let ptx = format!("{out_dir}/pom_mine_sm{sm}.ptx");
            let output = std::process::Command::new(&nvcc)
                .args(["-ptx", "-O3", &format!("-arch=sm_{sm}"), "cuda/pom_mine.cu", "-o", &ptx])
                .output()
                .unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run for sm_{sm}: {e}"));
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!(
                    "nvcc failed to compile cuda/pom_mine.cu for sm_{sm}:\n{}",
                    stderr
                );
            }
        }

        // Optional prebuilt fatbins: staged into OUT_DIR so runtime can embed/package them now,
        // then start loading them in a follow-up refactor.
        let legacy_src = env::var("POM_FATBIN_LEGACY")
            .unwrap_or_else(|_| "cuda/pom_mine_legacy.fatbin".to_string());
        let nextgen_src = env::var("POM_FATBIN_NEXTGEN")
            .unwrap_or_else(|_| "cuda/pom_mine_nextgen.fatbin".to_string());
        let legacy_dst = format!("{out_dir}/pom_mine_legacy.fatbin");
        let nextgen_dst = format!("{out_dir}/pom_mine_nextgen.fatbin");

        if std::path::Path::new(&legacy_src).exists() {
            fs::copy(&legacy_src, &legacy_dst).unwrap_or_else(|e| {
                panic!("failed copying POM_FATBIN_LEGACY from {legacy_src} to {legacy_dst}: {e}")
            });
            println!("cargo:warning=PoM legacy fatbin staged from {}", legacy_src);
        } else {
            fs::write(&legacy_dst, []).unwrap_or_else(|e| {
                panic!("failed creating empty legacy fatbin placeholder {legacy_dst}: {e}")
            });
        }

        if std::path::Path::new(&nextgen_src).exists() {
            fs::copy(&nextgen_src, &nextgen_dst).unwrap_or_else(|e| {
                panic!("failed copying POM_FATBIN_NEXTGEN from {nextgen_src} to {nextgen_dst}: {e}")
            });
            println!("cargo:warning=PoM nextgen fatbin staged from {}", nextgen_src);
        } else {
            fs::write(&nextgen_dst, []).unwrap_or_else(|e| {
                panic!("failed creating empty nextgen fatbin placeholder {nextgen_dst}: {e}")
            });
        }
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_arch == "x86_64" && target_os != "windows" && target_os != "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64.s").compile("libkeccak.a");
    }
    if target_arch == "x86_64" && target_os == "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64-osx.s").compile("libkeccak.a");
    }
    Ok(())
}
