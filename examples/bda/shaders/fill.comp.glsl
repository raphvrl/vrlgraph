#version 450
#extension GL_EXT_buffer_reference              : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = 8, local_size_y = 8) in;

layout(buffer_reference, std430) readonly buffer Palette {
    vec4 colors[8];
};

layout(set = 0, binding = 1, rgba8) uniform writeonly image2D storage_images[];

layout(push_constant) uniform PC {
    uint width;
    uint height;
    uint storage_idx;
    uint _pad;
    uint64_t palette_addr;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= int(pc.width) || coord.y >= int(pc.height)) return;

    Palette palette = Palette(pc.palette_addr);

    float t = float(coord.x) / float(pc.width);
    int band = clamp(int(t * 8.0), 0, 7);
    float blend = fract(t * 8.0);

    vec4 color = mix(palette.colors[band], palette.colors[min(band + 1, 7)], blend);

    imageStore(storage_images[pc.storage_idx], coord, color);
}
