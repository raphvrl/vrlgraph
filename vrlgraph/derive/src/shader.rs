use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitInt, Type, TypeArray, TypePath};

struct TypeLayout {
    align: usize,
    size: usize,
}

pub fn impl_shader_type(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;

    let named_fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "`ShaderType` requires a struct with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "`ShaderType` can only be derived for structs",
            ));
        }
    };

    let mut offset: usize = 0;
    let mut max_align: usize = 0;
    let mut write_stmts: Vec<TokenStream> = Vec::new();
    let mut field_names: Vec<syn::Ident> = Vec::new();
    let mut field_types: Vec<Type> = Vec::new();

    for field in named_fields.iter() {
        let field_name = field.ident.as_ref().unwrap();
        let field_ty = &field.ty;

        let tl = resolve_type_layout(field_ty, &field.attrs)?;

        let aligned_offset = round_up(offset, tl.align);
        let end = aligned_offset + tl.size;

        write_stmts.push(quote! {
            <#field_ty as ::vrlgraph::ShaderType>::write_padded(
                &self.#field_name,
                &mut dst[#aligned_offset..#end],
            );
        });

        field_names.push(field_name.clone());
        field_types.push(field_ty.clone());

        offset = end;
        if tl.align > max_align {
            max_align = tl.align;
        }
    }

    let padded_size = round_up(offset, max_align);

    Ok(quote! {
        impl Clone for #name {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl Copy for #name {}

        impl ::vrlgraph::ShaderType for #name {
            const PADDED_SIZE: usize = #padded_size;

            fn write_padded(&self, dst: &mut [u8]) {
                #(#write_stmts)*
            }
        }
    })
}

fn resolve_type_layout(
    ty: &Type,
    attrs: &[syn::Attribute],
) -> syn::Result<TypeLayout> {
    if let Some(tl) = parse_align_attr(attrs)? {
        return Ok(tl);
    }

    resolve_type_layout_inner(ty)
}

fn resolve_type_layout_inner(ty: &Type) -> syn::Result<TypeLayout> {
    match ty {
        Type::Path(tp) => resolve_path_layout(tp),
        Type::Array(ta) => resolve_array_layout(ta),
        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported type for `ShaderType` — use `#[align(N)]` to specify layout manually",
        )),
    }
}

fn resolve_path_layout(tp: &TypePath) -> syn::Result<TypeLayout> {
    let ident = type_path_ident(tp);
    match ident.as_deref() {
        Some("f32" | "u32" | "i32") => Ok(TypeLayout { align: 4, size: 4 }),
        Some("f64" | "u64" | "i64") => Ok(TypeLayout { align: 8, size: 8 }),
        Some("u16" | "i16") => Ok(TypeLayout { align: 2, size: 2 }),
        Some("bool") => Ok(TypeLayout { align: 4, size: 4 }),

        Some("Vec2" | "UVec2" | "IVec2") => Ok(TypeLayout { align: 4, size: 8 }),
        Some("Vec3" | "UVec3" | "IVec3") => Ok(TypeLayout { align: 4, size: 12 }),
        Some("Vec3A") => Ok(TypeLayout { align: 4, size: 16 }),
        Some("Vec4" | "UVec4" | "IVec4") => Ok(TypeLayout { align: 4, size: 16 }),

        Some("BVec2") => Ok(TypeLayout { align: 4, size: 8 }),
        Some("BVec3") => Ok(TypeLayout { align: 4, size: 12 }),
        Some("BVec4") => Ok(TypeLayout { align: 4, size: 16 }),

        Some("Mat2") => Ok(TypeLayout { align: 4, size: 16 }),
        Some("Mat3") => Ok(TypeLayout { align: 4, size: 36 }),
        Some("Mat4") => Ok(TypeLayout { align: 4, size: 64 }),

        Some("DVec2") => Ok(TypeLayout { align: 8, size: 16 }),
        Some("DVec3") => Ok(TypeLayout { align: 8, size: 24 }),
        Some("DVec4") => Ok(TypeLayout { align: 8, size: 32 }),

        Some("DMat2") => Ok(TypeLayout { align: 8, size: 32 }),
        Some("DMat3") => Ok(TypeLayout { align: 8, size: 72 }),
        Some("DMat4") => Ok(TypeLayout { align: 8, size: 128 }),

        Some("U64Vec2" | "I64Vec2") => Ok(TypeLayout { align: 8, size: 16 }),
        Some("U64Vec3" | "I64Vec3") => Ok(TypeLayout { align: 8, size: 24 }),
        Some("U64Vec4" | "I64Vec4") => Ok(TypeLayout { align: 8, size: 32 }),

        Some("U16Vec2" | "I16Vec2") => Ok(TypeLayout { align: 2, size: 4 }),
        Some("U16Vec3" | "I16Vec3") => Ok(TypeLayout { align: 2, size: 6 }),
        Some("U16Vec4" | "I16Vec4") => Ok(TypeLayout { align: 2, size: 8 }),

        _ => Err(syn::Error::new_spanned(
            tp,
            format!(
                "unknown type `{}` for `ShaderType` — use `#[align(N)]` to specify layout manually",
                tp.path
                    .segments
                    .last()
                    .map_or("?", |s| s.ident.to_string().leak())
            ),
        )),
    }
}

fn resolve_array_layout(ta: &TypeArray) -> syn::Result<TypeLayout> {
    let len = parse_array_len(&ta.len)?;
    let inner = resolve_type_layout_inner(&ta.elem)?;
    Ok(TypeLayout {
        align: inner.align,
        size: inner.size * len,
    })
}

fn parse_align_attr(attrs: &[syn::Attribute]) -> syn::Result<Option<TypeLayout>> {
    for attr in attrs {
        if attr.path().is_ident("align") {
            let lit: LitInt = attr.parse_args()?;
            let align: usize = lit.base10_parse()?;
            if !align.is_power_of_two() {
                return Err(syn::Error::new_spanned(
                    lit,
                    "alignment must be a power of two",
                ));
            }
            return Ok(Some(TypeLayout { align, size: align }));
        }
    }
    Ok(None)
}

fn type_path_ident(tp: &TypePath) -> Option<String> {
    tp.path.segments.last().map(|s| s.ident.to_string())
}

fn parse_array_len(expr: &syn::Expr) -> syn::Result<usize> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(lit),
            ..
        }) => lit.base10_parse(),
        _ => Err(syn::Error::new_spanned(
            expr,
            "expected integer literal for array length",
        )),
    }
}

const fn round_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}
