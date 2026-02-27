fn main() {
    println!("cargo:rerun-if-changed=../../proto/ucf_gateway_v1.proto");

    if std::env::var_os("PROTOC").is_none() {
        let protoc_path = protoc_bin_vendored::protoc_bin_path().expect("resolve vendored protoc");
        std::env::set_var("PROTOC", protoc_path);
    }

    prost_build::Config::new()
        .compile_protos(&["../../proto/ucf_gateway_v1.proto"], &["../../proto"])
        .expect("compile gateway proto");
}
