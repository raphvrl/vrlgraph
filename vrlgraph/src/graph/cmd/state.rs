use ash::vk;
use smallvec::SmallVec;

use super::Cmd;
use crate::resource::Pipeline;
use crate::types::{ColorWriteMask, CompareOp, CullMode, FrontFace, PolygonMode, Topology};

impl<'a> Cmd<'a> {
    /// Binds a graphics pipeline. Dynamic rasterizer state is **not** reset;
    /// values persist across binds (OpenGL-like model). Call
    /// [`reset_dynamic_state`](Cmd::reset_dynamic_state) to restore defaults.
    pub fn bind_graphics_pipeline(&mut self, handle: Pipeline) {
        let pipe = self
            .frame_ctx()
            .pool
            .get_pipeline(handle.0)
            .expect("pipeline handle stale — destroyed before frame end");
        let raw_pipe = pipe.pipeline;
        let layout = pipe.layout;
        #[cfg(debug_assertions)]
        let reflected = pipe.reflected_pc.clone();
        unsafe {
            self.device
                .cmd_bind_pipeline(self.raw, vk::PipelineBindPoint::GRAPHICS, raw_pipe)
        };
        self.bound_layout = Some(layout);
        self.bound_bind_point = vk::PipelineBindPoint::GRAPHICS;
        #[cfg(debug_assertions)]
        {
            self.reflected_pc = reflected;
            self.pc_mismatch_warned.set(false);
        }
    }

    /// Resets all dynamic rasterizer state to defaults. Called once at the
    /// beginning of each pass. Defaults:
    /// - Rasterizer discard: off
    /// - Depth bias: off (constant = 0, clamp = 0, slope = 0)
    /// - Primitive restart: off
    /// - Cull mode: none
    /// - Front face: counter-clockwise
    /// - Topology: triangle list
    /// - Depth test/write: off
    /// - Depth compare: less-or-equal
    /// - Depth clamp: off
    /// - Polygon mode: fill
    /// - Blending: disabled (RGBA write mask, 1 attachment)
    pub fn reset_dynamic_state(&self) {
        self.set_rasterizer_discard_enable(false);
        self.set_depth_bias_enable(false);
        self.set_depth_bias(0.0, 0.0, 0.0);
        self.set_primitive_restart_enable(false);

        self.set_cull_mode(CullMode::NONE);
        self.set_front_face(FrontFace::CounterClockwise);
        self.set_primitive_topology(Topology::TriangleList);
        self.set_depth_test_enable(false);
        self.set_depth_write_enable(false);
        self.set_depth_compare_op(CompareOp::LessOrEqual);

        self.set_depth_clamp_enable(false);
        self.set_polygon_mode(PolygonMode::Fill);
        self.set_default_blend_state(1);
    }

    /// Binds a compute pipeline. Always call this before [`dispatch`](Cmd::dispatch).
    pub fn bind_compute_pipeline(&mut self, handle: Pipeline) {
        let pipe = self
            .frame_ctx()
            .pool
            .get_pipeline(handle.0)
            .expect("pipeline handle stale — destroyed before frame end");
        let raw_pipe = pipe.pipeline;
        let layout = pipe.layout;
        #[cfg(debug_assertions)]
        let reflected = pipe.reflected_pc.clone();
        unsafe {
            self.device
                .cmd_bind_pipeline(self.raw, vk::PipelineBindPoint::COMPUTE, raw_pipe)
        };
        self.bound_layout = Some(layout);
        self.bound_bind_point = vk::PipelineBindPoint::COMPUTE;
        #[cfg(debug_assertions)]
        {
            self.reflected_pc = reflected;
            self.pc_mismatch_warned.set(false);
        }
    }

    /// Sets the viewport. Use [`set_viewport_scissor`](Cmd::set_viewport_scissor)
    /// instead when the viewport and scissor cover the full surface.
    pub fn set_viewport(&self, viewport: vk::Viewport) {
        unsafe {
            self.device
                .cmd_set_viewport_with_count(self.raw, &[viewport])
        };
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&self, scissor: vk::Rect2D) {
        unsafe { self.device.cmd_set_scissor_with_count(self.raw, &[scissor]) };
    }

    /// Sets the viewport and scissor to cover the full extent. Depth range is
    /// `[0.0, 1.0]`. This is the right call for most full-screen passes.
    pub fn set_viewport_scissor(&self, extent: vk::Extent2D) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        unsafe {
            self.device
                .cmd_set_viewport_with_count(self.raw, &[viewport]);
            self.device.cmd_set_scissor_with_count(self.raw, &[scissor]);
        }
    }

    pub fn set_cull_mode(&self, mode: CullMode) {
        unsafe { self.device.cmd_set_cull_mode(self.raw, mode.into()) };
    }

    pub fn set_front_face(&self, face: FrontFace) {
        unsafe { self.device.cmd_set_front_face(self.raw, face.into()) };
    }

    pub fn set_primitive_topology(&self, topology: Topology) {
        unsafe {
            self.device
                .cmd_set_primitive_topology(self.raw, topology.into())
        };
    }

    pub fn set_depth_test_enable(&self, enable: bool) {
        unsafe { self.device.cmd_set_depth_test_enable(self.raw, enable) };
    }

    pub fn set_depth_write_enable(&self, enable: bool) {
        unsafe { self.device.cmd_set_depth_write_enable(self.raw, enable) };
    }

    pub fn set_depth_compare_op(&self, op: CompareOp) {
        unsafe { self.device.cmd_set_depth_compare_op(self.raw, op.into()) };
    }

    pub fn set_depth_clamp_enable(&self, enable: bool) {
        unsafe { self.ext_ds3.cmd_set_depth_clamp_enable(self.raw, enable) };
    }

    pub fn set_polygon_mode(&self, mode: PolygonMode) {
        unsafe { self.ext_ds3.cmd_set_polygon_mode(self.raw, mode.into()) };
    }

    pub fn set_color_blend_enable(&self, first: u32, enables: &[vk::Bool32]) {
        unsafe {
            self.ext_ds3
                .cmd_set_color_blend_enable(self.raw, first, enables)
        };
    }

    pub fn set_color_blend_equation(&self, first: u32, equations: &[vk::ColorBlendEquationEXT]) {
        unsafe {
            self.ext_ds3
                .cmd_set_color_blend_equation(self.raw, first, equations)
        };
    }

    pub fn set_color_write_mask(&self, first: u32, masks: &[ColorWriteMask]) {
        let raw: SmallVec<[vk::ColorComponentFlags; 4]> =
            masks.iter().map(|m| (*m).into()).collect();
        unsafe { self.ext_ds3.cmd_set_color_write_mask(self.raw, first, &raw) };
    }

    pub fn set_rasterizer_discard_enable(&self, enable: bool) {
        unsafe {
            self.device
                .cmd_set_rasterizer_discard_enable(self.raw, enable)
        };
    }

    pub fn set_depth_bias_enable(&self, enable: bool) {
        unsafe { self.device.cmd_set_depth_bias_enable(self.raw, enable) };
    }

    pub fn set_depth_bias(&self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        unsafe {
            self.device
                .cmd_set_depth_bias(self.raw, constant_factor, clamp, slope_factor)
        };
    }

    pub fn set_primitive_restart_enable(&self, enable: bool) {
        unsafe {
            self.device
                .cmd_set_primitive_restart_enable(self.raw, enable)
        };
    }

    /// Disables blending and sets the write mask to RGBA for `count` color
    /// attachments. Call this after binding a graphics pipeline when no custom
    /// blend state is needed.
    pub fn set_default_blend_state(&self, count: u32) {
        let enables: SmallVec<[vk::Bool32; 4]> = smallvec::smallvec![vk::FALSE; count as usize];
        let masks: SmallVec<[ColorWriteMask; 4]> =
            smallvec::smallvec![ColorWriteMask::RGBA; count as usize];
        self.set_color_blend_enable(0, &enables);
        self.set_color_write_mask(0, &masks);
    }
}
