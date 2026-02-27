#![forbid(unsafe_code)]

#[cfg(feature = "gateway-smoke")]
mod smoke {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::thread;

    use tempfile::tempdir;
    use ucf_client::{parse_cli, run};
    use ucf_gateway::{run_tcp_once, GatewayConfig, GatewayService};

    #[test]
    fn submit_and_stream_work_against_local_gateway() {
        let td = tempdir().expect("tmp");
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 45092);

        let td_path = td.path().to_path_buf();
        let handle = thread::spawn(move || {
            let mut svc = GatewayService::new(GatewayConfig::for_tests(&td_path));
            run_tcp_once(&mut svc, addr.port()).expect("server run");
        });

        std::thread::sleep(std::time::Duration::from_millis(50));

        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/client/controlframe_min.json");

        let args = vec![
            "ucf-client".to_string(),
            "submit".to_string(),
            "--fixture".to_string(),
            fixture.display().to_string(),
            "--endpoint".to_string(),
            format!("tcp://{}", addr),
            "--auth".to_string(),
            "test-token".to_string(),
        ];
        let out = run(parse_cli(&args).expect("parse")).expect("run");
        assert!(out.contains("submit_ok"));

        handle.join().expect("join");
    }
}
