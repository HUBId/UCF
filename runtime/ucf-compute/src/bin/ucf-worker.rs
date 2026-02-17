fn main() {
    if let Err(err) = ucf_compute::worker_backend::run_worker() {
        eprintln!("ucf-worker error: {err}");
        std::process::exit(1);
    }
}
