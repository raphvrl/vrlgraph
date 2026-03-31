mod shader;

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

/// Derives [`ShaderType`] for a struct, generating `Clone`, `Copy`, and a
/// [`write_padded`](ShaderType::write_padded) implementation that serializes
/// the struct to GPU-compatible padded bytes.
///
/// The struct itself is **not modified** — no hidden padding fields are
/// inserted. Instead, padding is applied at serialization time when calling
/// methods like [`Cmd::push_shader`] or [`Graph::write_shader`].
///
/// # Layout selection
///
/// - `#[derive(ShaderType)]` — std140 (default, suitable for uniform buffers)
/// - `#[shader_type(std430)]` on the struct — std430 (suitable for storage
///   buffers)
///
/// # Supported field types
///
/// **Scalars:** `f32`, `u32`, `i32`, `u64`
///
/// **Vectors (as arrays):** `[f32; 2]`, `[f32; 3]`, `[f32; 4]` (same for
/// `u32`/`i32`)
///
/// **Matrices:** `[[f32; 4]; 4]` (mat4), `[[f32; 4]; 3]` (mat3)
///
/// **glam types** (with the `glam` feature): `Vec2`, `Vec3`, `Vec3A`, `Vec4`,
/// `UVec2`–`UVec4`, `IVec2`–`IVec4`, `Mat3`, `Mat4`
///
/// For unsupported types, annotate the field with `#[align(N)]` where `N` is
/// the required alignment (must be a power of two).
///
/// # Example
///
/// ```rust,ignore
/// use vrlgraph::ShaderType;
///
/// #[derive(ShaderType)]
/// struct Camera {
///     view: [[f32; 4]; 4],
///     proj: [[f32; 4]; 4],
///     position: [f32; 3],
/// }
///
/// let cam = Camera { view, proj, position: [0.0, 1.0, 0.0] };
/// cmd.push_shader(&cam);
/// ```
#[proc_macro_derive(ShaderType, attributes(shader_type, align))]
pub fn derive_shader_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let layout = parse_shader_type_layout(&input.attrs)
        .unwrap_or(shader::Layout::Std140);

    shader::impl_shader_type(input, layout)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn parse_shader_type_layout(attrs: &[syn::Attribute]) -> Option<shader::Layout> {
    for attr in attrs {
        if attr.path().is_ident("shader_type") {
            let mut layout = None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("std140") {
                    layout = Some(shader::Layout::Std140);
                } else if meta.path.is_ident("std430") {
                    layout = Some(shader::Layout::Std430);
                }
                Ok(())
            });
            return layout;
        }
    }
    None
}
