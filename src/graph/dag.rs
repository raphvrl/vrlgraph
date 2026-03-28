use std::collections::{HashMap, HashSet};

use petgraph::algo::toposort;
use petgraph::graph::DiGraph;
use petgraph::visit::{Bfs, Reversed};

use super::pass::RecordedPass;
use crate::resource::BufferHandle;

#[derive(Debug)]
pub(super) struct CycleError {
    pub pass_name: &'static str,
}

pub(super) fn sort_and_cull_passes(
    passes: Vec<RecordedPass>,
    live_images: &HashSet<u32>,
) -> Result<Vec<RecordedPass>, CycleError> {
    if passes.is_empty() {
        return Ok(passes);
    }

    let n = passes.len();

    let mut image_writers: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut buffer_writers: HashMap<BufferHandle, Vec<usize>> = HashMap::new();

    for (i, pass) in passes.iter().enumerate() {
        for w in &pass.writes {
            image_writers.entry(w.image.0).or_default().push(i);
        }
        for w in &pass.buffer_writes {
            buffer_writers.entry(w.handle).or_default().push(i);
        }
    }

    let mut graph: DiGraph<usize, ()> = DiGraph::with_capacity(n, n * 2);
    let nodes: Vec<_> = (0..n).map(|i| graph.add_node(i)).collect();

    for (reader_idx, pass) in passes.iter().enumerate() {
        for r in &pass.reads {
            if let Some(writers) = image_writers.get(&r.image.0) {
                for &writer_idx in writers {
                    if writer_idx != reader_idx {
                        graph.add_edge(nodes[writer_idx], nodes[reader_idx], ());
                    }
                }
            }
        }
        for r in &pass.buffer_reads {
            if let Some(writers) = buffer_writers.get(&r.handle) {
                for &writer_idx in writers {
                    if writer_idx != reader_idx {
                        graph.add_edge(nodes[writer_idx], nodes[reader_idx], ());
                    }
                }
            }
        }
    }

    let mut live = vec![false; n];

    for (i, pass) in passes.iter().enumerate() {
        let is_root = pass.writes.iter().any(|w| live_images.contains(&w.image.0))
            || !pass.buffer_writes.is_empty();

        if !is_root {
            continue;
        }

        let rev = Reversed(&graph);
        let mut bfs = Bfs::new(&rev, nodes[i]);
        while let Some(node) = bfs.next(&rev) {
            live[graph[node]] = true;
        }
    }

    let sorted = toposort(&graph, None).map_err(|cycle| CycleError {
        pass_name: passes[graph[cycle.node_id()]].name,
    })?;

    let mut slots: Vec<Option<RecordedPass>> = passes.into_iter().map(Some).collect();

    let result = sorted
        .into_iter()
        .filter(|&node| live[graph[node]])
        .map(|node| {
            slots[graph[node]]
                .take()
                .expect("pass consumed twice — internal DAG bug")
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ash::vk;

    use super::*;
    use crate::graph::access::LoadOp;
    use crate::graph::image::GraphImage;
    use crate::graph::pass::PassAccess;

    fn img_access(id: u32) -> PassAccess {
        PassAccess {
            image: GraphImage(id),
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::empty(),
            access: vk::AccessFlags2::empty(),
            is_color: false,
            is_depth: false,
            load_op: LoadOp::Auto,
            layer: None,
        }
    }

    fn make_pass(name: &'static str, write_ids: &[u32], read_ids: &[u32]) -> RecordedPass {
        RecordedPass {
            name,
            reads: read_ids.iter().copied().map(img_access).collect(),
            writes: write_ids.iter().copied().map(img_access).collect(),
            buffer_reads: vec![],
            buffer_writes: vec![],
            view_mask: 0,
            execute: Box::new(|_, _| {}),
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(
            sort_and_cull_passes(vec![], &HashSet::new())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn live_pass_is_kept() {
        let result =
            sort_and_cull_passes(vec![make_pass("a", &[0], &[])], &HashSet::from([0u32])).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "a");
    }

    #[test]
    fn dead_pass_is_culled() {
        let result =
            sort_and_cull_passes(vec![make_pass("dead", &[0], &[])], &HashSet::new()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn dependency_order_is_respected() {
        let producer = make_pass("producer", &[0], &[]);
        let consumer = make_pass("consumer", &[1], &[0]);
        let result =
            sort_and_cull_passes(vec![consumer, producer], &HashSet::from([1u32])).unwrap();
        assert_eq!(result[0].name, "producer");
        assert_eq!(result[1].name, "consumer");
    }
}
