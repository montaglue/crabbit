//! Matching a function's ops before and after a pass.

use std::collections::{HashMap, VecDeque};

use pliron::common_traits::Named;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;

use crate::profile;

/// The ops of a function before (input) and after (output) a pass,
/// matched up.
#[derive(Default, Debug)]
pub struct Alignment {
    /// `(input, output)`: the same op on both sides.
    pub pairs: Vec<(Ptr<Operation>, Ptr<Operation>)>,
    /// Output ops with no input counterpart: the pass created them.
    pub created: Vec<Ptr<Operation>>,
    /// Input ops with no output counterpart: the pass removed or replaced them.
    pub deleted: Vec<Ptr<Operation>>,
    /// `(deleted, created)` at the same spot of a block — between the same
    /// two matched ops: what the pass put in place of what it took out.
    pub replaced: Vec<Replacement>,
}

/// Deleted input ops and the created output ops in their place.
pub type Replacement = (Vec<Ptr<Operation>>, Vec<Ptr<Operation>>);

impl Alignment {
    /// The deleted ops that `created` stands in place of, if any.
    pub fn replaced_by(&self, created: Ptr<Operation>) -> &[Ptr<Operation>] {
        self.replaced
            .iter()
            .find(|(_, outs)| outs.contains(&created))
            .map_or(&[], |(ins, _)| ins.as_slice())
    }
}

/// Above this many key comparisons a block is matched greedily instead of
/// by a longest common subsequence.
const LCS_LIMIT: usize = 4_000_000;

/// Match the ops of the function `output` to those of `input` by `key`
/// (equal keys may be the same op): blocks with the same label first, in
/// order (a longest common subsequence of keys), then what is left over
/// across the whole function, in order. Only the function's own blocks are
/// matched; an op with regions of its own counts as one op.
pub fn align(
    ctx: &Context,
    input: Ptr<Operation>,
    output: Ptr<Operation>,
    key: &dyn Fn(&Context, Ptr<Operation>) -> String,
) -> Alignment {
    let in_blocks = blocks(ctx, input);
    let out_blocks = blocks(ctx, output);
    let in_by_label: HashMap<String, usize> = in_blocks
        .iter()
        .enumerate()
        .filter_map(|(i, (label, _))| label.clone().map(|l| (l, i)))
        .collect();

    let mut in_used: Vec<Vec<bool>> = in_blocks
        .iter()
        .map(|(_, ops)| vec![false; ops.len()])
        .collect();
    let mut out_left: Vec<Ptr<Operation>> = Vec::new();
    let mut alignment = Alignment::default();
    let mut gaps: Vec<Replacement> = Vec::new();

    for (label, out_ops) in &out_blocks {
        let Some(&bi) = label.as_ref().and_then(|l| in_by_label.get(l)) else {
            out_left.extend(out_ops);
            continue;
        };
        let in_ops = &in_blocks[bi].1;
        let in_keys: Vec<String> = in_ops.iter().map(|&op| key(ctx, op)).collect();
        let out_keys: Vec<String> = out_ops.iter().map(|&op| key(ctx, op)).collect();
        let matched = matching(&in_keys, &out_keys);
        let mut out_matched = vec![false; out_ops.len()];
        let (mut prev_i, mut prev_o) = (0, 0);
        for &(i, o) in matched
            .iter()
            .chain(std::iter::once(&(in_ops.len(), out_ops.len())))
        {
            gaps.push((in_ops[prev_i..i].to_vec(), out_ops[prev_o..o].to_vec()));
            if i < in_ops.len() {
                in_used[bi][i] = true;
                out_matched[o] = true;
                alignment.pairs.push((in_ops[i], out_ops[o]));
            }
            (prev_i, prev_o) = (i + 1, o + 1);
        }
        out_left.extend(
            out_ops
                .iter()
                .zip(out_matched)
                .filter(|(_, m)| !m)
                .map(|(&op, _)| op),
        );
    }

    // Leftovers across the function: equal keys, in order.
    let mut in_left: HashMap<String, VecDeque<Ptr<Operation>>> = HashMap::new();
    let mut in_left_order: Vec<Ptr<Operation>> = Vec::new();
    for (bi, (_, ops)) in in_blocks.iter().enumerate() {
        for (i, &op) in ops.iter().enumerate() {
            if !in_used[bi][i] {
                in_left.entry(key(ctx, op)).or_default().push_back(op);
                in_left_order.push(op);
            }
        }
    }
    let mut taken = std::collections::HashSet::new();
    for op in out_left {
        match in_left.get_mut(&key(ctx, op)).and_then(VecDeque::pop_front) {
            Some(input_op) => {
                taken.insert(input_op);
                alignment.pairs.push((input_op, op));
            }
            None => alignment.created.push(op),
        }
    }
    alignment.deleted = in_left_order
        .into_iter()
        .filter(|op| !taken.contains(op))
        .collect();
    let created: std::collections::HashSet<_> = alignment.created.iter().copied().collect();
    let deleted: std::collections::HashSet<_> = alignment.deleted.iter().copied().collect();
    alignment.replaced = gaps
        .into_iter()
        .map(|(ins, outs)| {
            (
                ins.into_iter()
                    .filter(|op| deleted.contains(op))
                    .collect::<Vec<_>>(),
                outs.into_iter()
                    .filter(|op| created.contains(op))
                    .collect::<Vec<_>>(),
            )
        })
        .filter(|(ins, outs)| !ins.is_empty() && !outs.is_empty())
        .collect();
    alignment
}

/// Each matched pair passes its whole profile from the output op to the
/// input op.
pub fn transfer(ctx: &Context, alignment: &Alignment) {
    for &(input, output) in &alignment.pairs {
        if let Some(p) = profile::get(ctx, output) {
            profile::add_all(ctx, input, &p, 1.0);
        }
    }
}

/// The function's blocks, with their labels and ops.
fn blocks(ctx: &Context, func: Ptr<Operation>) -> Vec<(Option<String>, Vec<Ptr<Operation>>)> {
    let regions: Vec<_> = func.deref(ctx).regions().collect();
    let mut out = Vec::new();
    for region in regions {
        for block in region.deref(ctx).iter(ctx) {
            let label = block
                .deref(ctx)
                .given_name(ctx)
                .map(|name| name.to_string());
            out.push((label, block.deref(ctx).iter(ctx).collect()));
        }
    }
    out
}

/// Pairs `(i, o)` of equal keys, in order: a longest common subsequence,
/// or a greedy in-order match for very long blocks.
fn matching(a: &[String], b: &[String]) -> Vec<(usize, usize)> {
    if a.len().saturating_mul(b.len()) > LCS_LIMIT {
        let mut pairs = Vec::new();
        let mut j = 0;
        for (i, key) in a.iter().enumerate() {
            if let Some(off) = b[j..].iter().position(|k| k == key) {
                pairs.push((i, j + off));
                j += off + 1;
            }
        }
        return pairs;
    }
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![0u32; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[at(i, j)] = if a[i] == b[j] {
                dp[at(i + 1, j + 1)] + 1
            } else {
                dp[at(i + 1, j)].max(dp[at(i, j + 1)])
            };
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if dp[at(i + 1, j)] >= dp[at(i, j + 1)] {
            i += 1;
        } else {
            j += 1;
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::matching;

    fn keys(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    #[test]
    fn lcs_keeps_order_and_skips_insertions() {
        let pairs = matching(&keys("a b c d"), &keys("a x b c y d"));
        assert_eq!(pairs, vec![(0, 0), (1, 2), (2, 3), (3, 5)]);
    }
}
