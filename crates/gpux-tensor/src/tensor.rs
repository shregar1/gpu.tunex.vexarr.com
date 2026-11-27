use std::sync::Arc;

use gpux_core::GpuBuffer;

use crate::context::GpuContext;
use crate::shape::Shape;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q8_0,
    Q4_0,
    /// Q4_K_M (k-quants): super-block of 256 elements with sub-block scales.
    /// 144 bytes per super-block: half d + half dmin + 12 bytes packed 6-bit
    /// scales/mins for 8 sub-blocks + 128 bytes of 4-bit quants.
    /// Effective ~4.5 bits/elem; better quality than Q4_0 at similar bandwidth.
    Q4_K,
}

impl DType {
    pub fn byte_size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K => {
                panic!("quantized dtypes: use bytes_for_elements()")
            }
        }
    }

    pub fn bytes_for_elements(&self, n: usize) -> usize {
        match self {
            DType::F32 => n * 4,
            DType::F16 => n * 2,
            DType::Q8_0 => (n / 32) * 34,   // half scale + 32 int8
            DType::Q4_0 => (n / 32) * 18,   // half scale + 16 packed int4
            DType::Q4_K => (n / 256) * 144, // half d + half dmin + 12 bytes scales + 128 bytes qs
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Q8_0 | DType::Q4_0 | DType::Q4_K)
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub buffer: Arc<GpuBuffer>,
    pub shape: Shape,
    pub ctx: Arc<GpuContext>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(buffer: GpuBuffer, shape: Shape, ctx: Arc<GpuContext>) -> Self {
        Self {
            buffer: Arc::new(buffer),
            shape,
            ctx,
            dtype: DType::F32,
        }
    }

    pub fn new_with_dtype(buffer: GpuBuffer, shape: Shape, ctx: Arc<GpuContext>, dtype: DType) -> Self {
        Self {
            buffer: Arc::new(buffer),
            shape,
            ctx,
            dtype,
        }
    }

    pub fn zeros(ctx: &Arc<GpuContext>, shape: impl Into<Shape>) -> gpux_core::Result<Self> {
        let shape = shape.into();
        let buffer = ctx.device.create_buffer(shape.byte_size())?;
        Ok(Self::new(buffer, shape, ctx.clone()))
    }

    /// Allocate an uninitialized tensor. Use only when the kernel will write every element.
    pub fn uninit(ctx: &Arc<GpuContext>, shape: impl Into<Shape>) -> gpux_core::Result<Self> {
        let shape = shape.into();
        let buffer = ctx.device.create_buffer_uninit(shape.byte_size())?;
        Ok(Self::new(buffer, shape, ctx.clone()))
    }

    /// Allocate a persistent zero tensor that survives scratch resets (for KV cache, etc.)
    pub fn zeros_persistent(ctx: &Arc<GpuContext>, shape: impl Into<Shape>) -> gpux_core::Result<Self> {
        let shape = shape.into();
        let buffer = ctx.device.create_buffer_persistent(shape.byte_size())?;
        Ok(Self::new(buffer, shape, ctx.clone()))
    }

    pub fn ones(ctx: &Arc<GpuContext>, shape: impl Into<Shape>) -> gpux_core::Result<Self> {
        let shape = shape.into();
        let data = vec![1.0f32; shape.numel()];
        let buffer = ctx.device.create_buffer_with_data(&data)?;
        Ok(Self::new(buffer, shape, ctx.clone()))
    }

    pub fn from_data(ctx: &Arc<GpuContext>, shape: impl Into<Shape>, data: &[f32]) -> gpux_core::Result<Self> {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "data length must match shape");
        let buffer = ctx.device.create_buffer_with_data(data)?;
        Ok(Self::new(buffer, shape, ctx.clone()))
    }

    pub fn from_uint_data(ctx: &Arc<GpuContext>, shape: impl Into<Shape>, data: &[u32]) -> gpux_core::Result<Self> {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "data length must match shape");
        let buffer = ctx.device.create_buffer_with_data(data)?;
        Ok(Self::new(buffer, shape, ctx.clone()))
    }

    pub fn from_raw_bytes(ctx: &Arc<GpuContext>, shape: impl Into<Shape>, data: &[u8], dtype: DType) -> gpux_core::Result<Self> {
        let shape = shape.into();
        let expected_bytes = shape.numel() * dtype.byte_size();
        assert_eq!(data.len(), expected_bytes, "raw bytes length must match shape * dtype");
        let buffer = ctx.device.create_buffer_with_data(data)?;
        Ok(Self::new_with_dtype(buffer, shape, ctx.clone(), dtype))
    }

    pub fn scalar(ctx: &Arc<GpuContext>, value: f32) -> gpux_core::Result<Self> {
        let buffer = ctx.device.create_buffer_with_data(&[value])?;
        Ok(Self::new(buffer, Shape::scalar(), ctx.clone()))
    }

    pub fn randn(ctx: &Arc<GpuContext>, shape: impl Into<Shape>, scale: f32) -> gpux_core::Result<Self> {
        use rand::Rng;
        let shape = shape.into();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| {
                let u1: f32 = rng.gen::<f32>().max(1e-7);
                let u2: f32 = rng.gen();
                scale * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        Self::from_data(ctx, shape, &data)
    }

    pub fn kaiming_uniform(ctx: &Arc<GpuContext>, shape: impl Into<Shape>) -> gpux_core::Result<Self> {
        use rand::Rng;
        let shape = shape.into();
        let fan_in = if shape.ndim() >= 2 { shape.dims()[1] } else { shape.dims()[0] };
        let bound = (1.0 / fan_in as f32).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| rng.gen_range(-bound..bound))
            .collect();
        Self::from_data(ctx, shape, &data)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.ctx.queue.synchronize().expect("GPU sync failed");
        self.buffer.read_to_vec()
    }

    pub fn to_uint_vec(&self) -> Vec<u32> {
        self.ctx.queue.synchronize().expect("GPU sync failed");
        self.buffer.read_to_vec()
    }

    pub fn gpu_buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Self {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape.numel(),
            new_shape.numel(),
            "reshape: numel mismatch {} vs {}",
            self.shape.numel(),
            new_shape.numel()
        );
        Self {
            buffer: self.buffer.clone(),
            shape: new_shape,
            ctx: self.ctx.clone(),
            dtype: self.dtype,
        }
    }

    pub fn to_f16(&self) -> gpux_core::Result<Self> {
        if self.dtype == DType::F16 {
            return Ok(self.clone());
        }
        let n = self.numel();
        let out_buf = self.ctx.device.create_buffer(n * 2)?;
        let out = Self::new_with_dtype(out_buf, self.shape.clone(), self.ctx.clone(), DType::F16);
        let pipeline = self.ctx.shaders.get("cast_f32_to_f16")?;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n as u64)
        })?;
        Ok(out)
    }

    pub fn to_f32(&self) -> gpux_core::Result<Self> {
        if self.dtype == DType::F32 {
            return Ok(self.clone());
        }
        let n = self.numel();
        let out_buf = self.ctx.device.create_buffer(n * 4)?;
        let out = Self::new(out_buf, self.shape.clone(), self.ctx.clone());
        let pipeline = self.ctx.shaders.get("cast_f16_to_f32")?;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n as u64)
        })?;
        Ok(out)
    }

    pub fn is_f16(&self) -> bool {
        self.dtype == DType::F16
    }

    pub fn zeros_like(&self) -> gpux_core::Result<Self> {
        let byte_size = self.numel() * self.dtype.byte_size();
        let buffer = self.ctx.device.create_buffer(byte_size)?;
        Ok(Self::new_with_dtype(buffer, self.shape.clone(), self.ctx.clone(), self.dtype))
    }

    fn dims_buffer(&self, dims: &[u32]) -> gpux_core::Result<GpuBuffer> {
        self.ctx.device.create_buffer_with_data(dims)
    }

    /// Zero-copy view: narrow the last dimension of a "vector-like" tensor.
    /// Valid when the product of all dims except the last is 1 (e.g., [1, D] or [D]).
    /// Returns a view sharing the same underlying buffer with an adjusted offset.
    pub fn narrow_last_dim(&self, start: usize, len: usize) -> Self {
        let last = self.shape.last_dim();
        assert!(
            start + len <= last,
            "narrow_last_dim: start {} + len {} > last_dim {}",
            start, len, last
        );
        let batch = self.shape.numel() / last;
        assert_eq!(
            batch, 1,
            "narrow_last_dim zero-copy requires batch=1, got {}",
            batch
        );
        let byte_start = start * self.dtype.byte_size();
        let byte_len = len * self.dtype.byte_size();
        let new_buf = self.buffer.narrow(byte_start, byte_len);
        let mut new_dims = self.shape.dims().to_vec();
        *new_dims.last_mut().unwrap() = len;
        Self {
            buffer: std::sync::Arc::new(new_buf),
            shape: Shape::new(new_dims),
            ctx: self.ctx.clone(),
            dtype: self.dtype,
        }
    }

    /// Zero-copy view: narrow dimension 0 (always contiguous for row-major layout).
    /// E.g., [16, 1, 64] narrowed at (dim=0, start=14, len=2) → [2, 1, 64].
    pub fn narrow_first_dim(&self, start: usize, len: usize) -> Self {
        let first = self.shape.dims()[0];
        assert!(
            start + len <= first,
            "narrow_first_dim: start {} + len {} > dim0 {}",
            start, len, first
        );
        let chunk: usize = self.shape.dims()[1..].iter().product();
        let byte_start = start * chunk * self.dtype.byte_size();
        let byte_len = len * chunk * self.dtype.byte_size();
        let new_buf = self.buffer.narrow(byte_start, byte_len);
        let mut new_dims = self.shape.dims().to_vec();
        new_dims[0] = len;
        Self {
            buffer: std::sync::Arc::new(new_buf),
            shape: Shape::new(new_dims),
            ctx: self.ctx.clone(),
            dtype: self.dtype,
        }
    }
}

