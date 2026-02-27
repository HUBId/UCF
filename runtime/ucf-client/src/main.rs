#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match ucf_client::parse_cli(&args).and_then(ucf_client::run) {
        Ok(out) => println!("{out}"),
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}
