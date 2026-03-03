use std::path::PathBuf;

#[cfg(unix)]
use ucf_gateway::run_unix_once;
use ucf_gateway::{
    run_tcp_once, transport_from_env, GatewayConfig, GatewayService, GatewayTransport,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workdir =
        PathBuf::from(std::env::var("UCF_WORKDIR").unwrap_or_else(|_| ".ucf".to_string()));
    std::fs::create_dir_all(&workdir)?;
    let cfg = GatewayConfig::from_env(&workdir)?;
    let mut service = GatewayService::new(cfg);
    let transport = transport_from_env(&workdir)?;
    loop {
        match &transport {
            #[cfg(unix)]
            GatewayTransport::Unix(path) => run_unix_once(&mut service, path)?,
            #[cfg(not(unix))]
            GatewayTransport::Unix(path) => {
                return Err(format!(
                    "unix socket transport is not supported on this build: {}",
                    path.display()
                )
                .into());
            }
            GatewayTransport::TcpLocal(port) => run_tcp_once(&mut service, *port)?,
            GatewayTransport::Pipe(name) => {
                return Err(format!(
                    "named pipe transport is not implemented in this build: {name}"
                )
                .into());
            }
        }
    }
}
