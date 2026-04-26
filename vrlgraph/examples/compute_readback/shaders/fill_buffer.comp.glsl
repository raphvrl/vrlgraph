#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 64) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Counts {
    uint values[];
};

layout(push_constant) uniform PC {
    uint64_t addr;
    uint count;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    Counts c = Counts(pc.addr);
    c.values[i] = i * 2u;
}
