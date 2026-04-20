use std::marker::PhantomData;

/// Marker type for 2D sampled images (binding 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sampled;
/// Marker type for storage images (binding 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Storage;
/// Marker type for cubemap sampled images (binding 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cubemap;
/// Marker type for 2D array sampled images (binding 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Array2D;

/// Type-safe index into the global bindless descriptor table.
///
/// The phantom type `K` encodes which binding the index belongs to. Returned
/// by `Cmd::sampled_index` and friends as an internal representation;
/// use those methods directly to get the `u32` for push constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindlessIndex<K>(u32, PhantomData<K>);

impl<K> BindlessIndex<K> {
    pub(crate) fn new(index: u32) -> Self {
        Self(index, PhantomData)
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

impl<K> From<BindlessIndex<K>> for u32 {
    fn from(idx: BindlessIndex<K>) -> u32 {
        idx.0
    }
}

/// A sampler registered in the global bindless table.
///
/// Returned by [`Graph::create_sampler`](crate::graph::Graph::create_sampler). Pass it to
/// [`Cmd::sampler_index`](crate::graph::Cmd::sampler_index) inside a
/// pass closure to get the `u32` index for push constants. Pass it to
/// [`Graph::destroy_sampler`](crate::graph::Graph::destroy_sampler) to release it.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub(crate) handle: crate::resource::SamplerHandle,
    index: u32,
}

impl Sampler {
    pub(crate) fn new(handle: crate::resource::SamplerHandle, index: u32) -> Self {
        Self { handle, index }
    }

    pub fn raw(&self) -> u32 {
        self.index
    }
}
