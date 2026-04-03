#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in flat uint v_view;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Params {
    float time;
} u;

void main() {
    vec3 left_tint  = vec3(0.9, 0.3, 0.2);
    vec3 right_tint = vec3(0.2, 0.3, 0.9);
    vec3 base = (v_view == 0) ? left_tint : right_tint;

    vec2 p = v_uv - 0.5;
    float d = length(p);
    float angle = atan(p.y, p.x) + u.time;

    float ring = smoothstep(0.22, 0.24, d) - smoothstep(0.26, 0.28, d);
    float ring2 = smoothstep(0.34, 0.36, d) - smoothstep(0.38, 0.40, d);
    float spoke = step(0.5, fract(angle * 3.0 / 3.14159));

    float pattern = max(ring, ring2 * spoke);
    out_color = vec4(mix(base * 0.15, base, pattern + 0.25), 1.0);
}
