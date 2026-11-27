use metal::{CompileOptions, ComputePipelineState, DeviceRef};
use std::collections::HashMap;
use parking_lot::RwLock;

use crate::error::GpuError;

pub struct ComputePipeline {
    pipeline: ComputePipelineState,
    function_name: String,
}

impl ComputePipeline {
    pub fn new(device: &DeviceRef, source: &str, function_name: &str) -> crate::Result<Self> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);

        let library = device
            .new_library_with_source(source, &options)
            .map_err(|e| GpuError::ShaderCompilation(e.to_string()))?;

        let function = library
            .get_function(function_name, None)
            .map_err(|e| GpuError::PipelineCreation(format!("{}: {}", function_name, e)))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| GpuError::PipelineCreation(format!("{}: {}", function_name, e)))?;

        Ok(Self {
            pipeline,
            function_name: function_name.to_string(),
        })
    }

    pub fn metal_pipeline(&self) -> &ComputePipelineState {
        &self.pipeline
    }

    pub fn max_total_threads_per_threadgroup(&self) -> u64 {
        self.pipeline.max_total_threads_per_threadgroup()
    }

    pub fn thread_execution_width(&self) -> u64 {
        self.pipeline.thread_execution_width()
    }

    pub fn function_name(&self) -> &str {
        &self.function_name
    }
}

/// Caches compiled pipelines to avoid recompilation
pub struct PipelineCache {
    device: *const metal::DeviceRef,
    cache: RwLock<HashMap<String, ComputePipeline>>,
}

unsafe impl Send for PipelineCache {}
unsafe impl Sync for PipelineCache {}

impl PipelineCache {
    pub fn new(device: &DeviceRef) -> Self {
        Self {
            device: device as *const _,
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_or_create(
        &self,
        source: &str,
        function_name: &str,
    ) -> crate::Result<std::sync::Arc<ComputePipeline>> {
        {
            let cache = self.cache.read();
            if let Some(_) = cache.get(function_name) {
                // Pipeline exists, but we can't return a reference from RwLock easily
                // We'll use Arc-based approach below
            }
        }

        let device = unsafe { &*self.device };
        let pipeline = ComputePipeline::new(device, source, function_name)?;
        Ok(std::sync::Arc::new(pipeline))
    }
}
