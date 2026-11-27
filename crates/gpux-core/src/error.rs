use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("no Metal-compatible GPU device found")]
    NoDevice,
    #[error("failed to create command queue")]
    CommandQueueCreation,
    #[error("failed to create buffer of size {size}")]
    BufferCreation { size: usize },
    #[error("failed to compile Metal shader: {0}")]
    ShaderCompilation(String),
    #[error("failed to create compute pipeline for function '{0}'")]
    PipelineCreation(String),
    #[error("command buffer execution failed: {0}")]
    Execution(String),
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("kernel not found: {0}")]
    KernelNotFound(String),
    #[error("{0}")]
    Other(String),
}
