use std::collections::{BTreeMap, HashMap};

use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::op_interfaces::{
            BinaryFixup, BinarySerializableOpInterface, BinarySerializationContext,
        },
        aarch64::ops::{self as aarch64_ops, FuncOp},
        builtin::op_interfaces::{OneRegionInterface, SymbolOpInterface},
    },
    input_error_noloc,
    ir::{basic_block::BasicBlock, op::op_cast, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::{
    attrs::{ATTR_KEY_AARCH64_ENCODED, ATTR_KEY_AARCH64_FIXUPS, ATTR_KEY_AARCH64_MODULE_LITERALS},
    error::Aarch64DarwinErr,
    frontend::module_op,
    util::{cast_operation, set_bytes_attr, set_fixups_attr},
};

pub struct Aarch64EncodePass;

impl Pass for Aarch64EncodePass {
    fn name(&self) -> &str {
        "aarch64-encode"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let funcs: Vec<_> = body.deref(ctx).iter(ctx).collect();

        let mut offsets = HashMap::<String, u64>::new();
        let mut literals = BTreeMap::<String, Vec<u8>>::new();
        let mut next_offset = 0u64;
        for op in &funcs {
            let Some(func) = cast_operation::<FuncOp>(ctx, *op) else {
                continue;
            };
            offsets.insert(func.get_symbol_name(ctx).to_string(), next_offset);
            collect_literals(ctx, func, &mut literals);
            next_offset += code_size(ctx, func);
        }
        let mut literal_offsets = HashMap::<String, u64>::new();
        let mut literal_bytes = Vec::new();
        let mut fixups = Vec::<BinaryFixup>::new();
        for (label, bytes) in &literals {
            literal_offsets.insert(label.clone(), next_offset + literal_bytes.len() as u64);
            literal_bytes.extend_from_slice(bytes);
        }

        let mut function_offset = 0u64;
        for op in funcs {
            let Some(func) = cast_operation::<FuncOp>(ctx, op) else {
                continue;
            };
            let mut bytes = Vec::new();
            let block_offsets = block_offsets(ctx, func, function_offset);
            let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
            for block in blocks {
                let insts: Vec<_> = block.deref(ctx).iter(ctx).collect();
                for inst_op in insts {
                    if !aarch64_ops::is_instruction(ctx, inst_op) {
                        continue;
                    }
                    let pc = function_offset + bytes.len() as u64;
                    let op = Operation::get_op_dyn(inst_op, ctx);
                    let serializable = op_cast::<dyn BinarySerializableOpInterface>(&*op)
                        .ok_or_else(|| {
                            input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(format!(
                                "non-serializable AArch64 operation `{}` reached binary emission",
                                Operation::get_opid(inst_op, ctx)
                            )))
                        })?;
                    let refs = BinarySerializationContext {
                        function_offsets: &offsets,
                        block_offsets: &block_offsets,
                        literal_offsets: &literal_offsets,
                    };
                    let encoded = serializable.encode_binary(ctx, pc, &refs)?;
                    fixups.extend(encoded.fixups);
                    bytes.extend_from_slice(&encoded.bytes);
                }
            }
            set_bytes_attr(op, ctx, ATTR_KEY_AARCH64_ENCODED.as_str(), bytes.clone());
            function_offset += bytes.len() as u64;
        }
        set_bytes_attr(
            root,
            ctx,
            ATTR_KEY_AARCH64_MODULE_LITERALS.as_str(),
            literal_bytes,
        );
        set_fixups_attr(root, ctx, ATTR_KEY_AARCH64_FIXUPS.as_str(), fixups);
        Ok(changed())
    }
}

fn collect_literals(ctx: &Context, func: FuncOp, literals: &mut BTreeMap<String, Vec<u8>>) {
    for block in func.get_region(ctx).deref(ctx).iter(ctx) {
        for op in block.deref(ctx).iter(ctx) {
            if !aarch64_ops::is_instruction(ctx, op) {
                continue;
            }
            let op_obj = Operation::get_op_dyn(op, ctx);
            if let Some(serializable) = op_cast::<dyn BinarySerializableOpInterface>(&*op_obj)
                && let Some((label, bytes)) = serializable.literal(ctx)
            {
                literals.entry(label).or_insert(bytes);
            }
        }
    }
}

fn code_size(ctx: &Context, func: FuncOp) -> u64 {
    func.get_region(ctx)
        .deref(ctx)
        .iter(ctx)
        .map(|block| {
            block
                .deref(ctx)
                .iter(ctx)
                .filter_map(|op| {
                    let op_obj = Operation::get_op_dyn(op, ctx);
                    op_cast::<dyn BinarySerializableOpInterface>(&*op_obj)
                        .map(|serializable| serializable.byte_len(ctx))
                })
                .sum::<u64>()
        })
        .sum()
}

fn block_offsets(
    ctx: &Context,
    func: FuncOp,
    function_offset: u64,
) -> HashMap<Ptr<BasicBlock>, u64> {
    let mut offsets = HashMap::new();
    let mut next_offset = function_offset;
    for block in func.get_region(ctx).deref(ctx).iter(ctx) {
        offsets.insert(block, next_offset);
        next_offset += block
            .deref(ctx)
            .iter(ctx)
            .filter_map(|op| {
                let op_obj = Operation::get_op_dyn(op, ctx);
                op_cast::<dyn BinarySerializableOpInterface>(&*op_obj)
                    .map(|serializable| serializable.byte_len(ctx))
            })
            .sum::<u64>();
    }
    offsets
}

#[cfg(test)]
mod tests {
    use llvm_compat::ll::LinkageAttr;
    use crate::{
        context::Context,
        dialects::{
            aarch64,
            aarch64::op_interfaces::{BinaryFixup, FixupKind},
            aarch64::registers::Register,
            builtin::{self, op_interfaces::OneRegionInterface},
            macho,
        },
        ir::{basic_block::BasicBlock, op::Op},
        linked_list::ContainsLinkedList,
        conversion::pass::{AnalysisManager, Pass},
    };

    use super::{
        super::{
            attrs::{
                ATTR_KEY_AARCH64_ENCODED, ATTR_KEY_AARCH64_FIXUPS,
                ATTR_KEY_AARCH64_MODULE_LITERALS,
            },
            util::{get_bytes_attr, get_fixups_attr},
        },
        Aarch64EncodePass,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        macho::register(&mut ctx);
        ctx
    }

    fn bytes(hex: &str) -> Vec<u8> {
        llvm_compat::ll::BytesAttr::parse_str(&format!("0x{hex}"))
            .unwrap()
            .0
    }

    #[test]
    fn records_function_offsets_literals_blocks_and_fixups() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let ignored = builtin::ops::ModuleOp::new(&mut ctx, "ignored".try_into().unwrap());
        ignored.get_operation().insert_at_back(body, &ctx);

        let caller = aarch64::ops::FuncOp::new(&mut ctx, "caller".try_into().unwrap(), LinkageAttr::External);
        caller.get_operation().insert_at_back(body, &ctx);
        let entry = caller.entry_block(&ctx);
        let exit = BasicBlock::new(&mut ctx, Some("exit".try_into().unwrap()), vec![]);
        exit.insert_at_back(caller.get_operation().deref(&ctx).get_region(0), &ctx);
        aarch64::ops::mov_imm(&mut ctx, Register::gpr(0), 1).insert_at_back(entry, &ctx);
        aarch64::ops::call(&mut ctx, "puts".try_into().unwrap()).insert_at_back(entry, &ctx);
        aarch64::ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        aarch64::ops::adr_literal(&mut ctx, Register::gpr(1), "lit0", vec![0xaa, 0xbb, 0xcc, 0xdd])
            .insert_at_back(entry, &ctx);
        aarch64::ops::adr_literal(&mut ctx, Register::gpr(2), "lit1", vec![0x01, 0x02, 0x03, 0x04])
            .insert_at_back(entry, &ctx);
        aarch64::ops::b(&mut ctx, exit).insert_at_back(entry, &ctx);
        aarch64::ops::ret(&mut ctx).insert_at_back(exit, &ctx);

        let callee = aarch64::ops::FuncOp::new(&mut ctx, "callee".try_into().unwrap(), LinkageAttr::External);
        callee.get_operation().insert_at_back(body, &ctx);
        aarch64::ops::ret(&mut ctx).insert_at_back(callee.entry_block(&ctx), &ctx);
        let third = aarch64::ops::FuncOp::new(&mut ctx, "third".try_into().unwrap(), LinkageAttr::External);
        third.get_operation().insert_at_back(body, &ctx);
        aarch64::ops::call(&mut ctx, "caller".try_into().unwrap())
            .insert_at_back(third.entry_block(&ctx), &ctx);

        Aarch64EncodePass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        assert_eq!(
            get_bytes_attr(
                caller.get_operation(),
                &ctx,
                ATTR_KEY_AARCH64_ENCODED.as_str()
            ),
            Some(bytes(
                "200080d20000009405000094c1000010c200001001000014c0035fd6"
            ))
        );
        assert_eq!(
            get_bytes_attr(
                callee.get_operation(),
                &ctx,
                ATTR_KEY_AARCH64_ENCODED.as_str()
            ),
            Some(bytes("c0035fd6"))
        );
        assert_eq!(
            get_bytes_attr(
                third.get_operation(),
                &ctx,
                ATTR_KEY_AARCH64_ENCODED.as_str()
            ),
            Some(bytes("f8ffff97"))
        );
        assert_eq!(
            get_bytes_attr(
                module.get_operation(),
                &ctx,
                ATTR_KEY_AARCH64_MODULE_LITERALS.as_str()
            ),
            Some(bytes("aabbccdd01020304"))
        );
        assert_eq!(
            get_fixups_attr(
                module.get_operation(),
                &ctx,
                ATTR_KEY_AARCH64_FIXUPS.as_str()
            ),
            Some(vec![BinaryFixup {
                offset: 4,
                symbol: "puts".to_string(),
                kind: FixupKind::Call26,
            }])
        );
    }
}
