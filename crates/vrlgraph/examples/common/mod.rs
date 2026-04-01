use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use vrlgraph::prelude::*;

pub trait Example: Sized {
    fn init(window: Window) -> Result<Self, GraphError>;
    fn draw(&mut self) -> Result<(), GraphError>;
    fn resize(&mut self, width: u32, height: u32);
    fn window(&self) -> &Window;

    fn window_attributes() -> WindowAttributes {
        Window::default_attributes()
    }
}

struct App<E: Example> {
    state: Option<E>,
}

impl<E: Example> App<E> {
    fn new() -> Self {
        Self { state: None }
    }
}

impl<E: Example> ApplicationHandler for App<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(E::window_attributes()).unwrap();

        match E::init(window) {
            Ok(state) => self.state = Some(state),
            Err(e) => {
                tracing::error!("init error: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => match state.draw() {
                Ok(()) => {}
                Err(GraphError::SwapchainOutOfDate) => {
                    let size = state.window().inner_size();
                    state.resize(size.width, size.height);
                }
                Err(e) => {
                    tracing::error!("draw error: {e}");
                    event_loop.exit();
                }
            },
            _ => {}
        }
    }
}

pub fn run<E: Example>() {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::<E>::new();
    event_loop.run_app(&mut app).unwrap();
}
