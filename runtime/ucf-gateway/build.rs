fn main() {
    println!("cargo:rerun-if-changed=../../proto/ucf_gateway_v1.proto");
    prost_build::Config::new()
        .compile_protos(&["../../proto/ucf_gateway_v1.proto"], &["../../proto"])
        .expect("compile gateway proto");
}
