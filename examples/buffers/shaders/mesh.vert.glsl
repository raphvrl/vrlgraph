#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430) readonly buffer Transform {
    float angle;
    float scale;
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
    Transform t = Transform(pc.transform_addr);
    Colors c = Colors(pc.colors_addr);

    float cs = cos(t.angle);
    float sn = sin(t.angle);
    vec2 pos = vec2(cs * in_pos.x - sn * in_pos.y,
            sn * in_pos.x + cs * in_pos.y) * t.scale;

    gl_Position = vec4(pos, 0.0, 1.0);
    out_color = c.data[gl_VertexIndex];
}
