#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430) readonly buffer Transform {
    mat4 matrix;
};

layout(buffer_reference, std430) readonly buffer Colors {
    vec4 data[];
};

layout(push_constant) uniform PC {
    uint64_t transform_addr;
    uint64_t colors_addr;
} pc;

layout(location = 0) in vec2 in_pos;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 pos = (Transform(pc.transform_addr).matrix * vec4(in_pos, 0.0, 1.0)).xy;
    gl_Position = vec4(pos, 0.0, 1.0);
    out_color = Colors(pc.colors_addr).data[gl_VertexIndex];
}
