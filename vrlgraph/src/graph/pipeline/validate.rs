use spirq::prelude::*;

#[derive(Debug, Clone)]
pub(crate) struct ReflectedPushConstants {
    pub total_size: usize,
    pub members: Vec<ReflectedMember>,
}

#[derive(Debug, Clone)]
pub(crate) struct ReflectedMember {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

pub(crate) fn reflect_push_constants(spv: &[u32]) -> Option<ReflectedPushConstants> {
    let entry_points = ReflectConfig::new().spv(spv).reflect().ok()?;

    for entry in &entry_points {
        for var in &entry.vars {
            if let Variable::PushConstant { ty, .. } = var
                && let Type::Struct(struct_ty) = ty {
                    let members: Vec<ReflectedMember> = struct_ty
                        .members
                        .iter()
                        .map(|m| ReflectedMember {
                            name: m.name.clone().unwrap_or_else(|| "?".into()),
                            offset: m.offset.unwrap_or(0),
                            size: m.ty.nbyte().unwrap_or(0),
                        })
                        .collect();

                    let total_size = members.iter().map(|m| m.offset + m.size).max().unwrap_or(0);

                    return Some(ReflectedPushConstants {
                        total_size,
                        members,
                    });
                }
        }
    }

    None
}

pub(crate) fn validate_push_constants(
    reflected: &ReflectedPushConstants,
    rust_size: usize,
    rust_type: &str,
) {
    if rust_size == reflected.total_size {
        return;
    }

    let mut detail = String::new();
    for m in &reflected.members {
        detail.push_str(&format!(
            "\n    [offset {:>3}, size {:>3}] {}",
            m.offset, m.size, m.name
        ));
    }

    tracing::warn!(
        "push_constants size mismatch:\n  \
         Rust type `{rust_type}` PADDED_SIZE = {rust_size}\n  \
         SPIR-V push_constant block size = {}{detail}",
        reflected.total_size,
    );
}
