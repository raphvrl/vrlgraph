#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, rgba8) uniform writeonly image2D out_image;

layout(push_constant) uniform Params {
    uint width;
    uint height;
} u;

float edge(vec2 a, vec2 b, vec2 p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= int(u.width) || coord.y >= int(u.height)) return;

    vec2 uv = vec2(coord) / vec2(u.width, u.height);

    vec2 a = vec2(0.5, 0.1);
    vec2 b = vec2(0.9, 0.9);
    vec2 c = vec2(0.1, 0.9);

    bool inside = edge(a, b, uv) >= 0.0
               && edge(b, c, uv) >= 0.0
               && edge(c, a, uv) >= 0.0;

    vec4 color = inside
        ? vec4(1.0, 0.6, 0.1, 1.0)
        : vec4(0.08, 0.08, 0.08, 1.0);

    imageStore(out_image, coord, color);
}
