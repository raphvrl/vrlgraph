#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 0, binding = 2) uniform sampler samplers[];

layout(push_constant) uniform PC {
    vec2 screen_size;
    uint texture_index;
    uint sampler_index;
} pc;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 out_color;

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 low = srgb / 12.92;
    vec3 high = pow((srgb + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, cutoff);
}

void main() {
    vec4 tex_color = texture(
        sampler2D(textures[pc.texture_index], samplers[pc.sampler_index]),
        in_uv
    );
    vec4 color = in_color;
    color.rgb = srgb_to_linear(color.rgb);
    out_color = color * tex_color;
    out_color.rgb *= out_color.a;
}
