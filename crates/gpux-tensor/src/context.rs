use std::sync::Arc;

use gpux_core::{CommandQueue, GpuDevice};
use gpux_shaders::ShaderRegistry;

pub struct GpuContext {
    pub device: GpuDevice,
    pub queue: CommandQueue,
    pub shaders: ShaderRegistry,
}

impl GpuContext {
    pub fn new() -> gpux_core::Result<Arc<Self>> {
        let mut device = GpuDevice::new()?;
        // 16 MB scratch for decode intermediates — eliminates Metal buffer allocation overhead
        device.enable_scratch(16 * 1024 * 1024);
        let queue = device.create_command_queue()?;
        let shaders = ShaderRegistry::new(&device)?;
        Ok(Arc::new(Self {
            device,
            queue,
            shaders,
        }))
    }
}
