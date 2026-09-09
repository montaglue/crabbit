//! `crabbit-inspect-driver`: the one-shot stdio CLI driver for
//! pliron-inspect (render/pass requests on a single document). The
//! resident analysis server is the separate `crabbit-analysisd` binary —
//! separate thin binaries sharing this crate's library, by decision of
//! record (no mode-flag multiplexing).

use std::path::PathBuf;

use clap::Parser;
use crabbit_inspect_driver::CrabbitHooks;
use pliron_inspect_driver::run_stdio_driver;

#[derive(Parser)]
#[command(name = "crabbit-inspect-driver")]
#[command(about = "crabbit IR driver for pliron-inspect (one-shot CLI)")]
struct Args {
    /// Input IR file
    input: Option<PathBuf>,
    /// Removed: the resident server moved to `crabbit-analysisd`.
    #[arg(long, hide = true)]
    serve: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.serve {
        anyhow::bail!(
            "--serve moved to the dedicated `crabbit-analysisd` binary \
             (cargo run -p crabbit-inspect-driver --bin crabbit-analysisd)"
        );
    }
    run_stdio_driver(&CrabbitHooks::new(), args.input.as_deref())
}
