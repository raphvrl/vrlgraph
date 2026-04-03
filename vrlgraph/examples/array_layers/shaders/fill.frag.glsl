#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Params {
    vec4 color;
    uint layer;
} u;

void main() {
    vec2 p = v_uv - 0.5;
    float pattern;

    switch (u.layer) {
        case 0:
            pattern = step(length(p), 0.3);
            break;
        case 1:
            pattern = step(0.5, fract(v_uv.y * 8.0));
            break;
        case 2:
            pattern = step(0.5, fract(v_uv.x * 8.0));
            break;
        default:
            pattern = mod(floor(v_uv.x * 8.0) + floor(v_uv.y * 8.0), 2.0);
            break;
    }

    vec3 base = u.color.rgb;
    vec3 dark = base * 0.3;
    out_color = vec4(mix(dark, base, pattern), 1.0);
}
