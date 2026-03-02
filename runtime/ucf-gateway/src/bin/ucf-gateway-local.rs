use std::path::PathBuf;

#[cfg(unix)]
use ucf_gateway::run_unix_once;
use ucf_gateway::{run_tcp_once, GatewayConfig, GatewayService, GatewayTransport};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workdir =
        PathBuf::from(std::env::var("UCF_WORKDIR").unwrap_or_else(|_| ".ucf".to_string()));
    std::fs::create_dir_all(&workdir)?;
    let cfg = GatewayConfig::from_env(&workdir)?;
    let mut service = GatewayService::new(cfg);
    let transport = GatewayTransport::default_v1();
    loop {
        match &transport {
            #[cfg(unix)]
            GatewayTransport::Unix(path) => run_unix_once(&mut service, path)?,
            GatewayTransport::TcpLocal(port) => run_tcp_once(&mut service, *port)?,
            #[allow(unreachable_patterns)]
            _ => run_tcp_once(&mut service, 44991)?,
        }
    }
}
