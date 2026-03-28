#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 0, binding = 2) uniform sampler samplers[];

layout(push_constant) uniform Params {
    uint sampled_idx;
    uint sampler_idx;
} u;

void main() {
    out_color = texture(sampler2D(textures[u.sampled_idx], samplers[u.sampler_idx]), v_uv);
}
