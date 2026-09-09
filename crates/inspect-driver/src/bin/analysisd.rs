//! `crabbit-analysisd`: the resident analysis/compilation server
//! (docs/SERVERD-PLAN.md). Speaks the pliron-inspect server protocol over
//! stdio, and optionally over a loopback HTTP shim; answers pipeline runs,
//! per-pass IR, artifacts, and the `ir_*` language-feature queries.

use clap::Parser;
use crabbit_inspect_driver::analysis_hooks_factory;

#[derive(Parser)]
#[command(name = "crabbit-analysisd")]
#[command(about = "crabbit resident analysis server for pliron-inspect")]
struct Args {
    /// Worker threads for pipeline runs.
    #[arg(long, default_value_t = 4)]
    workers: usize,
    /// Additionally serve the protocol over HTTP on this loopback address
    /// (e.g. 127.0.0.1:8177).
    #[arg(long)]
    http: Option<std::net::SocketAddr>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    pliron_inspect_driver::run_server_stdio(analysis_hooks_factory(), args.workers, args.http)
}
