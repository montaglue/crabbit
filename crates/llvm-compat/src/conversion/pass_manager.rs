use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{PassObj, PassOptions},
    result::STAIRResult,
};

pub type AfterPassCallback = Box<dyn FnMut(&Context, Ptr<Operation>, &str)>;

#[derive(Default)]
pub struct PassManager {
    pub after_pass: Option<AfterPassCallback>,
}

pub struct PreparedPass {
    pub pass: PassObj,
    pub options: PassOptions,
}

impl PreparedPass {
    pub fn new(pass: PassObj) -> Self {
        Self {
            pass: pass,
            options: PassOptions::default(),
        }
    }
}

impl From<PassObj> for PreparedPass {
    fn from(value: PassObj) -> Self {
        PreparedPass::new(value)
    }
}

#[derive(Default)]
pub struct Pipeline {
    pub passes: Vec<PreparedPass>,
}

impl Pipeline {
    pub fn new(passes: Vec<PassObj>) -> Self {
        Self {
            passes: passes.into_iter().map(|p| PreparedPass::new(p)).collect(),
        }
    }

    pub fn add_pass(mut self, pass: impl Into<PreparedPass>) -> Self {
        let pass = pass.into();
        self.passes.push(pass);
        self
    }

    pub fn add_pipeline(mut self, pipeline: Pipeline) -> Self {
        self.passes.extend(pipeline.passes.into_iter());
        self
    }

    pub fn names(&self) -> Vec<String> {
        self.passes
            .iter()
            .map(|p| p.pass.name().to_string())
            .collect()
    }
}

impl<T: Into<PreparedPass>> From<Vec<T>> for Pipeline {
    fn from(value: Vec<T>) -> Self {
        Pipeline {
            passes: value.into_iter().map(Into::into).collect(),
        }
    }
}

impl PassManager {
    pub fn run(
        &mut self,
        pipeline: Pipeline,
        ctx: &mut Context,
        mut root: Ptr<Operation>,
    ) -> STAIRResult<Ptr<Operation>> {
        for pass in pipeline.passes {
            let options = pass.options;
            let pass = pass.pass;
            let pass_name = pass.name().to_string();
            root = pass.run(root, ctx, options)?;
            if let Some(after_pass) = &mut self.after_pass {
                after_pass(ctx, root, &pass_name);
            }
        }
        Ok(root)
    }

    pub fn without_callback() -> Self {
        Self::default()
    }

    pub fn with_after_pass(after_pass: AfterPassCallback) -> Self {
        Self {
            after_pass: Some(after_pass),
        }
    }
}

impl PassManager {
    pub fn run_owned_context(
        &self,
        passes: Vec<PreparedPass>,
        mut ctx: Context,
        mut root: Ptr<Operation>,
    ) -> STAIRResult<Ptr<Operation>> {
        for pass in passes {
            let options = pass.options;
            let pass = pass.pass;
            root = pass.run(root, &mut ctx, options)?;
        }
        Ok(root)
    }
}
