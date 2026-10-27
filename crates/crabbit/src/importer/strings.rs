use super::*;

pub(super) fn literal_string_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    _body: &Body<'tcx>,
    constant: &ConstOperand<'tcx>,
) -> Option<String> {
    if !matches!(
        constant.const_.ty().kind(),
        rustc_middle::ty::TyKind::Ref(_, inner, _) if matches!(inner.kind(), rustc_middle::ty::TyKind::Str)
    ) {
        return None;
    }

    if let Ok(value) = constant
        .const_
        .eval(tcx, _body.typing_env(tcx), constant.span)
        && let Some(bytes) = value.try_get_slice_bytes_for_diagnostics(tcx)
        && let Ok(value) = std::str::from_utf8(bytes)
    {
        return Some(value.to_string());
    }

    let debug = format!("{:?}", constant.const_);
    if let Some(value) = parse_debug_string_const(&debug) {
        return Some(value);
    }

    let snippet = tcx.sess.source_map().span_to_snippet(constant.span).ok()?;
    parse_rust_string_literal(snippet.trim())
}

/// Evaluate a (monomorphized) `&str` constant that is not a source literal,
/// e.g. a promoted `type_name` string, into its UTF-8 contents.
pub(super) fn evaluated_str_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
) -> Option<String> {
    let value = constant.eval(tcx, typing_env, span).ok()?;
    let bytes = value.try_get_slice_bytes_for_diagnostics(tcx)?;
    std::str::from_utf8(bytes).ok().map(str::to_string)
}

pub(super) fn literal_byte_string_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: &ConstOperand<'tcx>,
) -> Option<Vec<u8>> {
    let snippet = tcx.sess.source_map().span_to_snippet(constant.span).ok()?;
    parse_rust_byte_string_literal(snippet.trim())
}

pub(super) fn evaluated_byte_string_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    len: u64,
) -> Option<Vec<u8>> {
    let value = constant.eval(tcx, typing_env, span).ok()?;
    match value {
        rustc_mir::ConstValue::Scalar(ptr) => {
            let ptr = ptr
                .to_pointer(&tcx)
                .discard_err()?
                .into_pointer_or_addr()
                .ok()?;
            let (provenance, offset) = ptr.prov_and_relative_offset();
            allocation_bytes(tcx, provenance.alloc_id(), offset, len)
        }
        rustc_mir::ConstValue::Slice { alloc_id, meta } => {
            allocation_bytes(tcx, alloc_id, Size::ZERO, meta)
        }
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            let alloc = tcx.global_alloc(alloc_id);
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = alloc else {
                return None;
            };
            let ptr_size = tcx.data_layout.pointer_size();
            let ptr = alloc
                .inner()
                .read_scalar(
                    &tcx,
                    rustc_mir::interpret::alloc_range(offset, ptr_size),
                    true,
                )
                .ok()?
                .to_pointer(&tcx)
                .discard_err()?
                .into_pointer_or_addr()
                .ok()?;
            let (provenance, offset) = ptr.prov_and_relative_offset();
            allocation_bytes(tcx, provenance.alloc_id(), offset, len)
        }
        rustc_mir::ConstValue::ZeroSized => Some(Vec::new()),
    }
}

pub(super) fn allocation_bytes<'tcx>(
    tcx: TyCtxt<'tcx>,
    alloc_id: rustc_mir::interpret::AllocId,
    offset: Size,
    len: u64,
) -> Option<Vec<u8>> {
    let alloc = tcx.global_alloc(alloc_id);
    let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = alloc else {
        return None;
    };
    let range = rustc_mir::interpret::alloc_range(offset, Size::from_bytes(len));
    alloc
        .inner()
        .get_bytes_strip_provenance(&tcx, range)
        .ok()
        .map(|bytes| bytes.to_vec())
}

pub(super) fn parse_debug_string_const(debug: &str) -> Option<String> {
    let start = debug.find("const \"")? + "const ".len();
    parse_rust_string_literal(&debug[start..])
}

pub(super) fn parse_rust_string_literal(input: &str) -> Option<String> {
    let mut chars = input.chars();
    if chars.next()? != '"' {
        return None;
    }

    let mut out = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '0' => out.push('\0'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

pub(super) fn parse_rust_byte_string_literal(input: &str) -> Option<Vec<u8>> {
    let mut chars = input.chars();
    if chars.next()? != 'b' || chars.next()? != '"' {
        return None;
    }

    let mut out = Vec::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            match ch {
                'n' => out.push(b'\n'),
                'r' => out.push(b'\r'),
                't' => out.push(b'\t'),
                '0' => out.push(b'\0'),
                '\\' => out.push(b'\\'),
                '"' => out.push(b'"'),
                '\'' => out.push(b'\''),
                other if other.is_ascii() => out.push(other as u8),
                _ => return None,
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else if ch.is_ascii() {
            out.push(ch as u8);
        } else {
            return None;
        }
    }
    None
}
