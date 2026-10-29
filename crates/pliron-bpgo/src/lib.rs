//! Backward profile-guided optimization over pliron traces.
//!
//! A compilation's trace holds the IR before its first pass and after every
//! pass. Backward PGO continues the compilation in reverse: a measurement of
//! the compiled program is attached to the ops of the last snapshot, and
//! then, from the last pass to the first, each pass's backward counterpart
//! ([BackwardPass]) reads the profile on its output and spreads it onto its
//! input, using the metadata its forward pass recorded on the ops. The
//! result is a profile at every level the backward passes reach.
//!
//! A profile is more than a cost: each op carries named measurements
//! ([ProfileAttr]) — `cycles` spent in it, how many times it ran
//! (`executions`), and what a pass attributes to its own decisions
//! (`regalloc.cycles`, `frame.cycles`, ...). How each one flows back is
//! the backward pass's business; [align] has the matching most of them
//! start from.
//!
//! [walk()] runs the backward passes over a [Trace](pliron_trace::Trace) and
//! returns the profile of every snapshot it reaches, keyed by the source
//! position of each op in that snapshot's text, so the trace's IR stays
//! exactly as the compiler printed it.

pub mod align;
pub mod profile;
pub mod walk;

pub use profile::{CYCLES, EXECUTIONS, ProfileAttr};
pub use walk::{BackwardPass, BackwardPasses, OpProfile, Stop, Walk, annotate, walk};
