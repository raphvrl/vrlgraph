#version 450
#extension GL_EXT_multiview : require

layout(location = 0) out vec2 v_uv;
layout(location = 1) out flat uint v_view;

layout(push_constant) uniform Params {
    float time;
} u;

void main() {
    vec2 pos[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );

    float eye_offset = (gl_ViewIndex == 0) ? -0.05 : 0.05;

    vec2 p = pos[gl_VertexIndex];
    gl_Position = vec4(p.x + eye_offset, p.y, 0.0, 1.0);
    v_uv = p * 0.5 + 0.5;
    v_view = gl_ViewIndex;
}
