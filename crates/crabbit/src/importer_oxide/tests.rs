#![cfg(test)]
#[cfg(test)]
use super::*;
use super::*;

#[test]
fn abi_field_collection_flattens_simple_structs() {
    let mut ctx = create_context();
    let ptr_ty: TypeHandle = llvm::types::PointerType::get(&mut ctx, 0).into();
    let i32_ty: TypeHandle = IntegerType::get(&mut ctx, 32, Signedness::Signed).into();
    let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Unsigned).into();
    let nested_ty: TypeHandle =
        llvm::types::StructType::get_unnamed(&mut ctx, vec![i32_ty, ptr_ty]).into();
    let aggregate_ty: TypeHandle =
        llvm::types::StructType::get_unnamed(&mut ctx, vec![ptr_ty, i64_ty, nested_ty]).into();

    let mut fields = Vec::new();
    collect_simple_abi_fields(&ctx, aggregate_ty, Vec::new(), &mut fields).unwrap();

    let indices = fields
        .iter()
        .map(|(indices, _)| indices.clone())
        .collect::<Vec<_>>();
    assert_eq!(indices, vec![vec![0], vec![1], vec![2, 0], vec![2, 1]]);
    assert_eq!(fields[0].1, ptr_ty);
    assert_eq!(fields[1].1, i64_ty);
    assert_eq!(fields[2].1, i32_ty);
    assert_eq!(fields[3].1, ptr_ty);
}
