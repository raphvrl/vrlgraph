#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 4) uniform texture2DArray array_textures[];
layout(set = 0, binding = 2) uniform sampler samplers[];

layout(push_constant) uniform Params {
    uint array_idx;
    uint sampler_idx;
} u;

void main() {
    int layer = (v_uv.x < 0.5) ? 0 : 1;
    vec2 local_uv = vec2(fract(v_uv.x * 2.0), v_uv.y);

    out_color = texture(
        sampler2DArray(array_textures[u.array_idx], samplers[u.sampler_idx]),
        vec3(local_uv, float(layer))
    );
}
