use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Ident, parse_macro_input};

/// Derives [`VertexInput`] for a `#[repr(C)]` struct.
///
/// Each field must implement [`VertexAttribute`] or carry a `#[format(FORMAT)]`
/// attribute to override the inferred Vulkan format.
///
/// # Attributes
///
/// - `#[vertex_input(rate = instance)]` on the struct — generates
///   `VK_VERTEX_INPUT_RATE_INSTANCE` instead of the default `VERTEX`.
/// - `#[format(R32G32B32_SFLOAT)]` on a field — overrides the format inferred
///   from the field type.
///
/// # Example
///
/// ```rust,ignore
/// #[repr(C)]
/// #[derive(Clone, Copy, Pod, Zeroable, VertexInput)]
/// struct Vertex {
///     pos: [f32; 3],
///     uv:  [f32; 2],
/// }
/// ```
#[proc_macro_derive(VertexInput, attributes(format, vertex_input))]
pub fn derive_vertex_input(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_vertex_input(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn impl_vertex_input(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;

    let input_rate = parse_input_rate(&input.attrs)?;

    let named_fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "`VertexInput` requires a struct with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "`VertexInput` can only be derived for structs",
            ));
        }
    };

    let attribute_entries: Vec<TokenStream2> = named_fields
        .iter()
        .enumerate()
        .map(|(location, field)| {
            let field_name = &field.ident;
            let field_ty = &field.ty;
            let location = location as u32;

            let format_expr = match parse_format_attr(&field.attrs)? {
                Some(fmt) => quote! { ::vrlgraph::ash::vk::Format::#fmt },
                None => quote! { <#field_ty as ::vrlgraph::VertexAttribute>::FORMAT },
            };

            Ok(quote! {
                ::vrlgraph::ash::vk::VertexInputAttributeDescription {
                    location: #location,
                    binding: 0,
                    format: #format_expr,
                    offset: ::std::mem::offset_of!(#name, #field_name) as u32,
                }
            })
        })
        .collect::<syn::Result<_>>()?;

    Ok(quote! {
        impl ::vrlgraph::VertexInput for #name {
            const BINDINGS: &'static [::vrlgraph::ash::vk::VertexInputBindingDescription] = &[
                ::vrlgraph::ash::vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: ::std::mem::size_of::<#name>() as u32,
                    input_rate: #input_rate,
                },
            ];
            const ATTRIBUTES: &'static [::vrlgraph::ash::vk::VertexInputAttributeDescription] = &[
                #(#attribute_entries),*
            ];
        }
    })
}

fn parse_input_rate(attrs: &[syn::Attribute]) -> syn::Result<TokenStream2> {
    for attr in attrs {
        if attr.path().is_ident("vertex_input") {
            let mut rate: Option<Ident> = None;
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rate") {
                    rate = Some(meta.value()?.parse()?);
                    Ok(())
                } else {
                    Err(meta.error("unknown `vertex_input` key — expected `rate`"))
                }
            })?;
            if let Some(r) = rate {
                return match r.to_string().as_str() {
                    "vertex" => Ok(quote! { ::vrlgraph::ash::vk::VertexInputRate::VERTEX }),
                    "instance" => Ok(quote! { ::vrlgraph::ash::vk::VertexInputRate::INSTANCE }),
                    _ => Err(syn::Error::new_spanned(
                        r,
                        "expected `vertex` or `instance`",
                    )),
                };
            }
        }
    }
    Ok(quote! { ::vrlgraph::ash::vk::VertexInputRate::VERTEX })
}

fn parse_format_attr(attrs: &[syn::Attribute]) -> syn::Result<Option<Ident>> {
    for attr in attrs {
        if attr.path().is_ident("format") {
            return Ok(Some(attr.parse_args()?));
        }
    }
    Ok(None)
}
