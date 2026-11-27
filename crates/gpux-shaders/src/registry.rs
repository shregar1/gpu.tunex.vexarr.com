use std::collections::HashMap;
use std::sync::Arc;

use gpux_core::{ComputePipeline, GpuDevice, GpuError};

use crate::kernels;

pub struct ShaderRegistry {
    pipelines: HashMap<String, Arc<ComputePipeline>>,
}

impl ShaderRegistry {
    pub fn new(device: &GpuDevice) -> gpux_core::Result<Self> {
        let mut pipelines = HashMap::new();

        for shader in kernels::all_shaders() {
            for &func_name in shader.functions {
                log::debug!("Compiling shader function: {}", func_name);
                let pipeline = device.create_pipeline(shader.source, func_name)?;
                pipelines.insert(func_name.to_string(), Arc::new(pipeline));
            }
        }

        log::info!("Compiled {} shader functions", pipelines.len());
        Ok(Self { pipelines })
    }

    pub fn get(&self, name: &str) -> gpux_core::Result<&Arc<ComputePipeline>> {
        self.pipelines
            .get(name)
            .ok_or_else(|| GpuError::KernelNotFound(name.to_string()))
    }

    pub fn list_kernels(&self) -> Vec<&str> {
        self.pipelines.keys().map(|s| s.as_str()).collect()
    }
}
