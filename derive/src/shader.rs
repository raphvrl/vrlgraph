use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitInt, Type, TypeArray, TypePath};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Std140,
    Std430,
}

struct TypeLayout {
    align: usize,
    size: usize,
}

pub fn impl_shader_type(input: DeriveInput, layout: Layout) -> syn::Result<TokenStream> {
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

        let tl = resolve_type_layout(field_ty, &field.attrs, layout)?;

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

    if layout == Layout::Std140 {
        max_align = round_up(max_align, 16);
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
    layout: Layout,
) -> syn::Result<TypeLayout> {
    if let Some(tl) = parse_align_attr(attrs, layout)? {
        return Ok(tl);
    }

    match ty {
        Type::Path(tp) => resolve_path_layout(tp, layout),
        Type::Array(ta) => resolve_array_layout(ta, layout),
        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported type for `ShaderType` — use `#[align(N)]` to specify layout manually",
        )),
    }
}

fn resolve_path_layout(tp: &TypePath, _layout: Layout) -> syn::Result<TypeLayout> {
    let ident = type_path_ident(tp);
    match ident.as_deref() {
        Some("f32" | "u32" | "i32") => Ok(TypeLayout { align: 4, size: 4 }),
        Some("u64") => Ok(TypeLayout { align: 8, size: 8 }),
        Some("Vec2") => Ok(TypeLayout { align: 8, size: 8 }),
        Some("Vec3") => Ok(TypeLayout {
            align: 16,
            size: 12,
        }),
        Some("Vec3A") => Ok(TypeLayout {
            align: 16,
            size: 16,
        }),
        Some("Vec4") => Ok(TypeLayout {
            align: 16,
            size: 16,
        }),
        Some("UVec2") => Ok(TypeLayout { align: 8, size: 8 }),
        Some("UVec3") => Ok(TypeLayout {
            align: 16,
            size: 12,
        }),
        Some("UVec4") => Ok(TypeLayout {
            align: 16,
            size: 16,
        }),
        Some("IVec2") => Ok(TypeLayout { align: 8, size: 8 }),
        Some("IVec3") => Ok(TypeLayout {
            align: 16,
            size: 12,
        }),
        Some("IVec4") => Ok(TypeLayout {
            align: 16,
            size: 16,
        }),
        Some("Mat3") => Ok(TypeLayout {
            align: 16,
            size: 48,
        }),
        Some("Mat4") => Ok(TypeLayout {
            align: 16,
            size: 64,
        }),
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

fn resolve_array_layout(ta: &TypeArray, layout: Layout) -> syn::Result<TypeLayout> {
    let len = parse_array_len(&ta.len)?;
    let elem = &*ta.elem;

    match elem {
        Type::Path(tp) => {
            let inner = resolve_path_layout(tp, layout)?;
            match len {
                2 => Ok(TypeLayout {
                    align: inner.size * 2,
                    size: inner.size * 2,
                }),
                3 => Ok(TypeLayout {
                    align: 16,
                    size: inner.size * 3,
                }),
                4 => Ok(TypeLayout {
                    align: inner.size * 4,
                    size: inner.size * 4,
                }),
                _ => Err(syn::Error::new_spanned(
                    ta,
                    "unsupported array length for `ShaderType` — use `#[align(N)]`",
                )),
            }
        }
        Type::Array(inner_arr) => {
            let inner_len = parse_array_len(&inner_arr.len)?;
            let inner_elem = &*inner_arr.elem;

            let scalar_size = match inner_elem {
                Type::Path(tp) => resolve_path_layout(tp, layout)?.size,
                _ => {
                    return Err(syn::Error::new_spanned(
                        inner_elem,
                        "unsupported nested array element type",
                    ));
                }
            };

            if scalar_size == 4 && inner_len == 4 && len == 4 {
                return Ok(TypeLayout {
                    align: 16,
                    size: 64,
                });
            }
            if scalar_size == 4 && inner_len == 4 && len == 3 {
                return Ok(TypeLayout {
                    align: 16,
                    size: 48,
                });
            }

            Err(syn::Error::new_spanned(
                ta,
                "unsupported matrix type — use `[[f32; 4]; 4]` for mat4 or `[[f32; 4]; 3]` for mat3",
            ))
        }
        _ => Err(syn::Error::new_spanned(
            ta,
            "unsupported array element type for `ShaderType`",
        )),
    }
}

fn parse_align_attr(attrs: &[syn::Attribute], _layout: Layout) -> syn::Result<Option<TypeLayout>> {
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
