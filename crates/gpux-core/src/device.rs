use metal::{Device, DeviceRef};

use crate::buffer::{GpuBuffer, ScratchAllocator};
use crate::command::CommandQueue;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

pub struct GpuDevice {
    device: Device,
    scratch: Option<ScratchAllocator>,
}

impl GpuDevice {
    pub fn new() -> crate::Result<Self> {
        let device = Device::system_default().ok_or(GpuError::NoDevice)?;
        log::info!(
            "GPU device: {} (unified memory: {}, recommended max working set: {} MB)",
            device.name(),
            device.has_unified_memory(),
            device.recommended_max_working_set_size() / 1024 / 1024,
        );
        Ok(Self { device, scratch: None })
    }

    /// Enable scratch allocator for decode. Call once before inference.
    /// Capacity in bytes (e.g., 16 * 1024 * 1024 for 16 MB).
    pub fn enable_scratch(&mut self, capacity: usize) {
        self.scratch = Some(ScratchAllocator::new(&self.device, capacity));
        log::info!("Scratch allocator enabled: {} MB", capacity / 1024 / 1024);
    }

    /// Activate scratch allocator for decode. Call after model/cache loading.
    pub fn activate_scratch(&self) {
        if let Some(ref scratch) = self.scratch {
            scratch.set_active(true);
        }
    }

    /// Reset scratch allocator. Call after synchronize() between tokens.
    pub fn reset_scratch(&self) {
        if let Some(ref scratch) = self.scratch {
            scratch.reset();
        }
    }

    pub fn scratch_used(&self) -> usize {
        self.scratch.as_ref().map(|s| s.used()).unwrap_or(0)
    }

    pub fn create_command_queue(&self) -> crate::Result<CommandQueue> {
        CommandQueue::new(&self.device)
    }

    pub fn create_buffer(&self, size: usize) -> crate::Result<GpuBuffer> {
        // Try scratch first for small-to-medium allocations
        if let Some(ref scratch) = self.scratch {
            if let Some(buf) = scratch.alloc(size) {
                return Ok(buf);
            }
        }
        GpuBuffer::new(&self.device, size)
    }

    /// Allocate a buffer WITHOUT zero-initialization.
    /// Use for kernel output buffers that will be fully written.
    pub fn create_buffer_uninit(&self, size: usize) -> crate::Result<GpuBuffer> {
        if let Some(ref scratch) = self.scratch {
            if let Some(buf) = scratch.alloc_uninit(size) {
                return Ok(buf);
            }
        }
        GpuBuffer::new(&self.device, size)
    }

    /// Allocate a buffer that persists across scratch resets (for weights, KV cache, etc.)
    pub fn create_buffer_persistent(&self, size: usize) -> crate::Result<GpuBuffer> {
        GpuBuffer::new(&self.device, size)
    }

    pub fn create_buffer_with_data<T: bytemuck::Pod>(&self, data: &[T]) -> crate::Result<GpuBuffer> {
        if let Some(ref scratch) = self.scratch {
            if let Some(buf) = scratch.alloc_with_data(data) {
                return Ok(buf);
            }
        }
        GpuBuffer::with_data(&self.device, data)
    }

    pub fn create_pipeline(&self, source: &str, function_name: &str) -> crate::Result<ComputePipeline> {
        ComputePipeline::new(&self.device, source, function_name)
    }

    pub fn metal_device(&self) -> &DeviceRef {
        &self.device
    }

    pub fn name(&self) -> &str {
        self.device.name()
    }

    pub fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    pub fn max_buffer_length(&self) -> u64 {
        self.device.max_buffer_length()
    }

    pub fn max_threads_per_threadgroup(&self) -> u64 {
        self.device.max_threads_per_threadgroup().width
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("name", &self.device.name())
            .field("unified_memory", &self.device.has_unified_memory())
            .finish()
    }
}
