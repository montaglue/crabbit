use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::context::{Context, Ptr};
use crate::graph::walkers::{
    IRNode, WALKCONFIG_PREORDER_FORWARD,
    interruptible::{self, walk_advance, walk_break},
};
use crate::ir::{op::Op, operation::Operation};
use crate::result::{Error, STAIRResult};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PassOptions {
    options: HashMap<String, OptionValue>,
}

impl PassOptions {
    pub fn new(options: HashMap<String, OptionValue>) -> Self {
        Self { options }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionValue {
    Bool(bool),
    Num(String),
    Str(String),
    Ident(String),
    List(Vec<OptionValue>),
}

pub trait Pass {
    fn name(&self) -> &str;
    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>>;
}

pub type PassObj = Arc<dyn Pass>;

pub trait OperationPass {
    type OpType: Op + 'static;
    fn name(&self) -> &str;
    fn run_on_operation(&self, op: Self::OpType, ctx: &mut Context) -> STAIRResult<()>;
}

impl<U: OperationPass> Pass for U {
    fn name(&self) -> &str {
        <U as OperationPass>::name(self)
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        let mut state = OperationPassState { pass: self };
        if let ControlFlow::Break(err) = interruptible::mutable::walk_op(
            ctx,
            &mut state,
            &WALKCONFIG_PREORDER_FORWARD,
            root,
            run_operation_pass::<U>,
        ) {
            return Err(err);
        }
        Ok(root)
    }
}

struct OperationPassState<'a, U: OperationPass> {
    pass: &'a U,
}

fn run_operation_pass<U: OperationPass>(
    ctx: &mut Context,
    state: &mut OperationPassState<'_, U>,
    node: IRNode,
) -> interruptible::WalkResult<Error> {
    let IRNode::Operation(op_ptr) = node else {
        return walk_advance();
    };
    if let Some(op) = Operation::get_op::<U::OpType>(op_ptr, ctx) {
        if let Err(err) = state.pass.run_on_operation(op, ctx) {
            return walk_break(err);
        }
    }
    walk_advance()
}
