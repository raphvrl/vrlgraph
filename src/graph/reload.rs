use rustc_hash::FxHashSet;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use ash::vk;
use notify::{EventKind, RecursiveMode, Watcher};
use smallvec::SmallVec;

use crate::resource::ShaderModuleHandle;

#[derive(Clone)]
pub(crate) struct PipelineDesc {
    pub kind: PipelineKind,
}

#[derive(Clone)]
pub(crate) enum PipelineKind {
    Graphics {
        vertex: ShaderModuleHandle,
        fragment: ShaderModuleHandle,
        color_formats: Vec<vk::Format>,
        depth_format: Option<vk::Format>,
        vertex_bindings: Vec<vk::VertexInputBindingDescription>,
        vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
    },
    Compute {
        shader: ShaderModuleHandle,
    },
}

impl PipelineDesc {
    pub fn shader_module_handles(&self) -> SmallVec<[ShaderModuleHandle; 2]> {
        match &self.kind {
            PipelineKind::Graphics {
                vertex, fragment, ..
            } => {
                smallvec::smallvec![*vertex, *fragment]
            }
            PipelineKind::Compute { shader } => smallvec::smallvec![*shader],
        }
    }
}

pub(crate) struct ShaderWatcher {
    _watcher: notify::PollWatcher,
    rx: mpsc::Receiver<PathBuf>,
    watched: FxHashSet<PathBuf>,
}

impl ShaderWatcher {
    pub fn new() -> Result<Self, notify::Error> {
        let (tx, rx) = mpsc::channel();

        let watcher = notify::PollWatcher::new(
            move |res: notify::Result<notify::Event>| {
                if let Ok(event) = res
                    && matches!(event.kind, EventKind::Modify(_))
                {
                    for path in event.paths {
                        let _ = tx.send(path);
                    }
                }
            },
            notify::Config::default().with_poll_interval(Duration::from_millis(500)),
        )?;

        Ok(Self {
            _watcher: watcher,
            rx,
            watched: FxHashSet::default(),
        })
    }

    pub fn watch(&mut self, path: &std::path::Path) {
        if let Ok(canonical) = path.canonicalize()
            && self.watched.insert(canonical.clone())
            && let Err(e) = self._watcher.watch(&canonical, RecursiveMode::NonRecursive)
        {
            tracing::warn!(
                "shader watcher: could not watch {}: {e}",
                canonical.display()
            );
        }
    }

    pub fn drain_changed(&self) -> Vec<PathBuf> {
        self.rx.try_iter().collect()
    }
}

impl Default for ShaderWatcher {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| {
            tracing::warn!("shader watcher disabled: {e}");
            let (_tx, rx) = mpsc::channel();
            let watcher = notify::PollWatcher::new(|_| {}, notify::Config::default())
                .expect("PollWatcher creation must not fail");
            Self {
                _watcher: watcher,
                rx,
                watched: FxHashSet::default(),
            }
        })
    }
}
