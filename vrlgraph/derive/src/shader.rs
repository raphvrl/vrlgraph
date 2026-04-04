use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, LitInt};

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

    let mut const_decls: Vec<TokenStream> = Vec::new();
    let mut write_stmts: Vec<TokenStream> = Vec::new();
    let mut align_exprs: Vec<TokenStream> = Vec::new();

    for (i, field) in named_fields.iter().enumerate() {
        let field_name = field.ident.as_ref().unwrap();
        let field_ty = &field.ty;
        let offset_ident = format_ident!("__OFF_{}", i);
        let end_ident = format_ident!("__END_{}", i);

        let (align_expr, size_expr) = match parse_align_override(&field.attrs)? {
            Some(val) => (quote! { #val }, quote! { #val }),
            None => (
                quote! { <#field_ty as ::vrlgraph::ShaderType>::SCALAR_ALIGN },
                quote! { <#field_ty as ::vrlgraph::ShaderType>::PADDED_SIZE },
            ),
        };

        let prev_end = if i == 0 {
            quote! { 0usize }
        } else {
            let prev = format_ident!("__END_{}", i - 1);
            quote! { #prev }
        };

        const_decls.push(quote! {
            const #offset_ident: usize = ::vrlgraph::round_up(#prev_end, #align_expr);
            const #end_ident: usize = #offset_ident + #size_expr;
        });

        write_stmts.push(quote! {
            <#field_ty as ::vrlgraph::ShaderType>::write_padded(
                &self.#field_name,
                &mut dst[#offset_ident..#end_ident],
            );
        });

        align_exprs.push(align_expr);
    }

    let max_align_expr = align_exprs.iter().rev().fold(
        quote! { 1usize },
        |acc, a| quote! { const_max(#a, #acc) },
    );

    let last_end = if named_fields.is_empty() {
        quote! { 0usize }
    } else {
        let last = format_ident!("__END_{}", named_fields.len() - 1);
        quote! { #last }
    };

    Ok(quote! {
        impl Clone for #name {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl Copy for #name {}

        impl ::vrlgraph::ShaderType for #name {
            const SCALAR_ALIGN: usize = {
                const fn const_max(a: usize, b: usize) -> usize {
                    if a > b { a } else { b }
                }
                #max_align_expr
            };

            const PADDED_SIZE: usize = {
                const fn const_max(a: usize, b: usize) -> usize {
                    if a > b { a } else { b }
                }
                #(#const_decls)*
                ::vrlgraph::round_up(#last_end, #max_align_expr)
            };

            fn write_padded(&self, dst: &mut [u8]) {
                #(#const_decls)*
                #(#write_stmts)*
            }
        }
    })
}

fn parse_align_override(attrs: &[syn::Attribute]) -> syn::Result<Option<usize>> {
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
            return Ok(Some(align));
        }
    }
    Ok(None)
}
