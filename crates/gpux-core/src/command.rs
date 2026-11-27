use metal::{CommandBufferRef, ComputeCommandEncoderRef, DeviceRef, MTLSize};
use parking_lot::Mutex;
use std::time::Instant;

use crate::buffer::GpuBuffer;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;
use crate::profile::KernelProfiler;

struct CommandQueueState {
    current_cb: Option<metal::CommandBuffer>,
    /// Persistent compute encoder — reused across enqueue() calls to eliminate
    /// per-dispatch encoder create/destroy overhead (~40µs each).
    current_encoder: Option<metal::ComputeCommandEncoder>,
    dispatch_count: usize,
}

pub struct CommandQueue {
    queue: metal::CommandQueue,
    state: Mutex<CommandQueueState>,
}

impl CommandQueue {
    pub fn new(device: &DeviceRef) -> crate::Result<Self> {
        let queue = device.new_command_queue();
        Ok(Self {
            queue,
            state: Mutex::new(CommandQueueState {
                current_cb: None,
                current_encoder: None,
                dispatch_count: 0,
            }),
        })
    }

    /// Synchronous encode: creates a command buffer, runs the closure, commits, and waits.
    /// Use for operations that need immediate results (weight loading, debugging).
    pub fn encode<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnOnce(&mut CommandEncoder) -> crate::Result<()>,
    {
        // Flush any pending async work first
        self.synchronize()?;

        let command_buffer = self.queue.new_command_buffer();
        let metal_enc = command_buffer.new_compute_command_encoder();
        {
            let mut encoder = CommandEncoder { encoder: metal_enc, pending: Vec::new() };
            f(&mut encoder)?;
        }
        metal_enc.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status == metal::MTLCommandBufferStatus::Error {
            return Err(GpuError::Execution("command buffer error".into()));
        }
        Ok(())
    }

    /// Async enqueue: dispatches GPU work without blocking.
    /// Work is batched into a shared command buffer with a persistent compute encoder.
    /// The encoder is reused across calls — no create/destroy overhead per dispatch.
    ///
    /// When `KernelProfiler::global().is_enabled()`, each enqueue is committed in its
    /// own CB and synchronously waited; the elapsed time is recorded against the
    /// kernel name(s) used during the closure. This serializes execution (slow!) but
    /// gives precise per-kernel GPU time. Use only for profiling.
    pub fn enqueue<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnOnce(&mut CommandEncoder) -> crate::Result<()>,
    {
        if KernelProfiler::global().is_enabled() {
            return self.enqueue_profiled(f);
        }

        let mut state = self.state.lock();

        // Ensure command buffer exists
        if state.current_cb.is_none() {
            state.current_cb = Some(self.queue.new_command_buffer().to_owned());
        }

        // Ensure compute encoder exists — reused across dispatches
        if state.current_encoder.is_none() {
            let cb = state.current_cb.as_deref().unwrap();
            state.current_encoder = Some(cb.new_compute_command_encoder().to_owned());
        }

        let result = {
            let enc_ref = state.current_encoder.as_deref().unwrap();
            let mut encoder = CommandEncoder { encoder: enc_ref, pending: Vec::new() };
            f(&mut encoder)
        };
        state.dispatch_count += 1;

        // Auto-flush at 2048 dispatches to avoid oversized command buffers.
        if state.dispatch_count >= 2048 {
            if let Some(encoder) = state.current_encoder.take() {
                encoder.end_encoding();
            }
            if let Some(cb) = state.current_cb.take() {
                cb.commit();
                state.dispatch_count = 0;
            }
        }

        result
    }

    fn enqueue_profiled<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnOnce(&mut CommandEncoder) -> crate::Result<()>,
    {
        // Flush anything that was queued normally before profiling started.
        self.synchronize()?;

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let mut names: Vec<String> = Vec::new();
        {
            let mut encoder = CommandEncoder { encoder: enc, pending: Vec::new() };
            f(&mut encoder)?;
            names = encoder.pending;
        }
        enc.end_encoding();

        let start = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let elapsed = start.elapsed();

        if cb.status() == metal::MTLCommandBufferStatus::Error {
            return Err(GpuError::Execution("command buffer error".into()));
        }

        let key = if names.is_empty() {
            "(no-dispatch)".to_string()
        } else if names.len() == 1 {
            names.into_iter().next().unwrap()
        } else {
            names.join("+")
        };
        KernelProfiler::global().record(&key, elapsed);

        Ok(())
    }

    /// Flush all pending GPU work and wait for completion.
    /// Call before reading GPU buffer contents back to CPU.
    pub fn synchronize(&self) -> crate::Result<()> {
        let mut state = self.state.lock();
        // End the persistent encoder first
        if let Some(encoder) = state.current_encoder.take() {
            encoder.end_encoding();
        }
        if let Some(cb) = state.current_cb.take() {
            let count = state.dispatch_count;
            cb.commit();
            cb.wait_until_completed();
            if cb.status() == metal::MTLCommandBufferStatus::Error {
                return Err(GpuError::Execution("command buffer error".into()));
            }
            if std::env::var("GPUX_DEBUG_DISPATCH").is_ok() {
                eprintln!("  [gpu] synced {} dispatches", count);
            }
            state.dispatch_count = 0;
        }
        Ok(())
    }

    pub fn metal_queue(&self) -> &metal::CommandQueue {
        &self.queue
    }
}

unsafe impl Send for CommandQueue {}
unsafe impl Sync for CommandQueue {}

/// Lightweight wrapper around a Metal compute encoder reference.
/// The encoder is NOT created/destroyed per dispatch — it's reused
/// from the CommandQueue's persistent encoder.
///
/// `pending` accumulates the kernel names dispatched during the current
/// `enqueue` closure; the profiler reads it at the boundary.
pub struct CommandEncoder<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    pending: Vec<String>,
}

impl<'a> CommandEncoder<'a> {
    fn note_dispatch(&mut self, pipeline: &ComputePipeline) {
        if KernelProfiler::global().is_enabled() {
            self.pending.push(pipeline.function_name().to_string());
        }
    }

    pub fn dispatch(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        grid_size: [u64; 3],
        threadgroup_size: [u64; 3],
    ) -> crate::Result<()> {
        self.encoder.set_compute_pipeline_state(pipeline.metal_pipeline());

        for (i, buf) in buffers.iter().enumerate() {
            self.encoder.set_buffer(i as u64, Some(buf.metal_buffer()), buf.metal_offset() as u64);
        }

        let grid = MTLSize::new(grid_size[0], grid_size[1], grid_size[2]);
        let threadgroup = MTLSize::new(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]);

        self.encoder.dispatch_thread_groups(grid, threadgroup);
        self.note_dispatch(pipeline);

        Ok(())
    }

    pub fn dispatch_threads(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        params: &[&GpuBuffer],
        total_threads: [u64; 3],
        threadgroup_size: [u64; 3],
    ) -> crate::Result<()> {
        self.encoder.set_compute_pipeline_state(pipeline.metal_pipeline());

        let mut idx = 0u64;
        for buf in buffers.iter() {
            self.encoder.set_buffer(idx, Some(buf.metal_buffer()), buf.metal_offset() as u64);
            idx += 1;
        }
        for buf in params.iter() {
            self.encoder.set_buffer(idx, Some(buf.metal_buffer()), buf.metal_offset() as u64);
            idx += 1;
        }

        let grid = MTLSize::new(total_threads[0], total_threads[1], total_threads[2]);
        let threadgroup = MTLSize::new(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]);

        self.encoder.dispatch_threads(grid, threadgroup);
        self.note_dispatch(pipeline);

        Ok(())
    }

    pub fn dispatch_1d(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        count: u64,
    ) -> crate::Result<()> {
        let tg = 256u64.min(count);
        self.dispatch_threads(pipeline, buffers, &[], [count, 1, 1], [tg, 1, 1])
    }

    /// Dispatch with inline params + threadgroup memory binding.
    /// `tgmem_sizes[i]` is the byte length to bind at threadgroup memory index `i`.
    /// Used by tiled kernels (e.g. simdgroup_matrix GEMM) that need shared scratch.
    pub fn dispatch_inline_tgmem(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        inline_params: &[&[u8]],
        tgmem_sizes: &[usize],
        grid_size: [u64; 3],
        threadgroup_size: [u64; 3],
    ) -> crate::Result<()> {
        self.encoder.set_compute_pipeline_state(pipeline.metal_pipeline());

        for (i, buf) in buffers.iter().enumerate() {
            self.encoder.set_buffer(i as u64, Some(buf.metal_buffer()), buf.metal_offset() as u64);
        }
        let base = buffers.len() as u64;
        for (j, bytes) in inline_params.iter().enumerate() {
            self.encoder.set_bytes(
                base + j as u64,
                bytes.len() as u64,
                bytes.as_ptr() as *const _,
            );
        }
        for (i, &size) in tgmem_sizes.iter().enumerate() {
            self.encoder.set_threadgroup_memory_length(i as u64, size as u64);
        }

        let grid = MTLSize::new(grid_size[0], grid_size[1], grid_size[2]);
        let threadgroup = MTLSize::new(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]);
        self.encoder.dispatch_thread_groups(grid, threadgroup);
        self.note_dispatch(pipeline);
        Ok(())
    }

    /// Dispatch with inline parameter bytes appended after `buffers`.
    /// Uses Metal's `setBytes` for the inline params — no buffer allocation,
    /// no atomic bump on the scratch allocator. Faster than passing dims/scale
    /// as separate scratch buffers when the data is ≤4 KB.
    ///
    /// Buffer indices: `buffers[i]` -> binding `i`, `inline_params[j]` -> binding `buffers.len() + j`.
    pub fn dispatch_inline(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        inline_params: &[&[u8]],
        grid_size: [u64; 3],
        threadgroup_size: [u64; 3],
    ) -> crate::Result<()> {
        self.encoder.set_compute_pipeline_state(pipeline.metal_pipeline());

        for (i, buf) in buffers.iter().enumerate() {
            self.encoder.set_buffer(i as u64, Some(buf.metal_buffer()), buf.metal_offset() as u64);
        }
        let base = buffers.len() as u64;
        for (j, bytes) in inline_params.iter().enumerate() {
            self.encoder.set_bytes(
                base + j as u64,
                bytes.len() as u64,
                bytes.as_ptr() as *const _,
            );
        }

        let grid = MTLSize::new(grid_size[0], grid_size[1], grid_size[2]);
        let threadgroup = MTLSize::new(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]);
        self.encoder.dispatch_thread_groups(grid, threadgroup);
        self.note_dispatch(pipeline);
        Ok(())
    }
}
