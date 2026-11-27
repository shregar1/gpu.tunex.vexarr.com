

use crate::shape::Shape;
use crate::tensor::{DType, Tensor};

const TILE_SIZE: u64 = 32;

fn use_simd_matvec() -> bool {
    static SIMD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SIMD.get_or_init(|| {
        std::env::var("GPUX_SIMD").map(|v| v != "0").unwrap_or(true)
    })
}

fn threadgroup_1d(n: u64) -> ([u64; 3], [u64; 3]) {
    let tg = 256u64.min(n);
    let grid = (n + tg - 1) / tg;
    ([grid, 1, 1], [tg, 1, 1])
}

fn threadgroup_2d(width: u64, height: u64) -> ([u64; 3], [u64; 3]) {
    let tg_x = TILE_SIZE.min(width);
    let tg_y = TILE_SIZE.min(height);
    let grid_x = (width + tg_x - 1) / tg_x;
    let grid_y = (height + tg_y - 1) / tg_y;
    ([grid_x, grid_y, 1], [tg_x, tg_y, 1])
}

impl Tensor {
    // ===== Elementwise ops =====

    pub fn add(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        assert_eq!(self.shape(), other.shape(), "shapes must match for add");
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("add_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &other.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn sub(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        assert_eq!(self.shape(), other.shape(), "shapes must match for sub");
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("sub_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &other.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn mul(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        assert_eq!(self.shape(), other.shape(), "shapes must match for mul");
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("mul_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &other.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn mul_scalar(&self, scalar: f32) -> gpux_core::Result<Tensor> {
        let scalar_buf = self.ctx.device.create_buffer_with_data(&[scalar])?;
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("mul_scalar_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &scalar_buf, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn neg(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("neg_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn exp(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("exp_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn log(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("log_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn relu(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("relu_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn gelu(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("gelu_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn silu(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("silu_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn tanh_(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("tanh_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    // ===== Fused ops =====

    pub fn silu_mul(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        assert_eq!(self.shape(), other.shape(), "shapes must match for silu_mul");
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("silu_mul_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &other.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    pub fn fused_gate_up_silu(&self, gate_weight: &Tensor, up_weight: &Tensor) -> gpux_core::Result<Tensor> {
        let n = gate_weight.shape().dims()[0] as u32;
        let k = gate_weight.shape().last_dim() as u32;
        let m = (self.shape.numel() / k as usize) as u32;
        let out = Tensor::uninit(&self.ctx, Shape::new(vec![m as usize, n as usize]))?;

        // Q4_0 M-batched fast path for spec-decode verify (M=2..8). Single dispatch
        // produces silu(gate@x)·(up@x) for all M rows, sharing weight reads across M.
        // Dramatically faster than the unfused {gate, up, silu_mul} fallback for verify.
        const Q4_M_MAX: u32 = 8;
        if m > 1 && m <= Q4_M_MAX && gate_weight.dtype == DType::Q4_0 {
            let pipeline = self.ctx.shaders.get("fused_gate_up_silu_q4_0_m_f32")?;
            let dims = [m, n, k, 0u32];
            let rows_per_tg: u64 = 32;
            let tg_threads: u64 = 256;
            let num_tgs = (n as u64 + rows_per_tg - 1) / rows_per_tg;
            self.ctx.queue.enqueue(|enc| {
                enc.dispatch_inline(pipeline,
                    &[&self.buffer, &gate_weight.buffer, &up_weight.buffer, &out.buffer],
                    &[bytemuck::cast_slice(&dims)],
                    [num_tgs, 1, 1], [tg_threads, 1, 1])
            })?;
            return Ok(out);
        }

        // M=1 decode-only fast paths below.
        // Q4_0/Q8_0 use 32-rows-per-TG layout (8 SG/TG × 4 rows/SG, 256 threads/TG)
        //   — halves the TG count vs 16-rows-per-TG to reduce launch overhead.
        // F16/F32 use 4-rows-per-TG layout (memory-bound, smaller TGs OK).
        assert_eq!(m, 1, "M>1 fused_gate_up_silu only supported for Q4_0 with M\u{2264}8; \
                         caller should fall back to separate gate/up/silu_mul");
        let dims = [n, k];
        let (kernel, rows_per_tg, tg_threads) = match gate_weight.dtype {
            DType::F16  => ("fused_gate_up_silu_f16w_f32", 4u64, 128u64),
            DType::Q4_0 => ("fused_gate_up_silu_q4_0_f32", 32u64, 256u64),
            DType::Q8_0 => ("fused_gate_up_silu_q8_0_f32", 32u64, 256u64),
            DType::F32  => ("fused_gate_up_silu_f32", 4u64, 128u64),
            // Q4_K hasn't got a fused gate_up_silu kernel yet; SwiGLU::forward
            // detects this at the dtype level and routes to separate
            // gate/up matvecs + silu_mul, so we shouldn't be reached for Q4_K.
            DType::Q4_K => panic!("fused_gate_up_silu has no Q4_K variant; SwiGLU should have used the unfused path"),
        };
        let pipeline = self.ctx.shaders.get(kernel)?;
        let num_tgs = (n as u64 + rows_per_tg - 1) / rows_per_tg;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_inline(pipeline,
                &[&self.buffer, &gate_weight.buffer, &up_weight.buffer, &out.buffer],
                &[bytemuck::cast_slice(&dims)],
                [num_tgs, 1, 1], [tg_threads, 1, 1])
        })?;
        Ok(out)
    }

    pub fn add_rms_norm(&self, other: &Tensor, gamma: &Tensor, eps: f32)
        -> gpux_core::Result<(Tensor, Tensor)>
    {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let residual_out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let normed_out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = [rows, cols];
        let eps_arr = [eps];
        let f16_gamma = gamma.dtype == DType::F16;
        let kernel = if f16_gamma { "rmsnorm_residual_f16w_f32" } else { "rmsnorm_residual_f32" };
        let pipeline = self.ctx.shaders.get(kernel)?;
        // SIMD-cooperative kernel: 1 threadgroup per row, 128 threads (4 SG) per TG.
        // The f16-weight variant has been converted; the f32 fallback keeps the legacy
        // 1-thread-per-row layout (used only when gamma is f32, i.e. training).
        let (grid, tg) = if f16_gamma {
            ([rows as u64, 1, 1], [128u64, 1, 1])
        } else {
            threadgroup_1d(rows as u64)
        };

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_inline(
                pipeline,
                &[&self.buffer, &other.buffer, &gamma.buffer,
                  &residual_out.buffer, &normed_out.buffer],
                &[bytemuck::cast_slice(&dims), bytemuck::cast_slice(&eps_arr)],
                grid,
                tg,
            )
        })?;
        Ok((residual_out, normed_out))
    }

    // ===== Matrix multiplication =====

    pub fn matmul(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        assert!(self.ndim() >= 2 && other.ndim() >= 2, "matmul requires 2D+ tensors");
        let m = self.shape.dims()[self.ndim() - 2] as u32;
        let k = self.shape.last_dim() as u32;
        let n = other.shape.last_dim() as u32;
        assert_eq!(k, other.shape.dims()[other.ndim() - 2] as u32, "inner dims must match");

        let out_shape = self.shape.matmul_shape(other.shape());
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[m, n, k])?;
        let pipeline = self.ctx.shaders.get("matmul_f32")?;

        let grid_x = ((n as u64) + TILE_SIZE - 1) / TILE_SIZE;
        let grid_y = ((m as u64) + TILE_SIZE - 1) / TILE_SIZE;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &other.buffer, &out.buffer, &dims],
                [grid_x, grid_y, 1],
                [TILE_SIZE, TILE_SIZE, 1],
            )
        })?;
        Ok(out)
    }

    pub fn matmul_at_b(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        let m = self.shape.last_dim() as u32;
        let k = self.shape.dims()[self.ndim() - 2] as u32;
        let n = other.shape.last_dim() as u32;

        let out_shape = Shape::new(vec![m as usize, n as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[m, n, k])?;
        let pipeline = self.ctx.shaders.get("matmul_at_b_f32")?;

        let grid_x = ((n as u64) + TILE_SIZE - 1) / TILE_SIZE;
        let grid_y = ((m as u64) + TILE_SIZE - 1) / TILE_SIZE;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &other.buffer, &out.buffer, &dims],
                [grid_x, grid_y, 1],
                [TILE_SIZE, TILE_SIZE, 1],
            )
        })?;
        Ok(out)
    }

    pub fn matmul_a_bt(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        // Route quantized weights to the quantized matmul path.
        if other.dtype.is_quantized() {
            return self.matmul_a_bt_quantized(other);
        }

        let m = self.shape.dims()[self.ndim() - 2] as u32;
        let k = self.shape.last_dim() as u32;
        let n = other.shape.dims()[other.ndim() - 2] as u32;
        let f16_weights = other.dtype == DType::F16;

        let out_shape = Shape::new(vec![m as usize, n as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;

        if m == 1 {
            let dims = [n, k];
            if use_simd_matvec() {
                let kernel = if f16_weights { "matvec_a_bt_f16w_f32" } else { "matvec_a_bt_f32" };
                let pipeline = self.ctx.shaders.get(kernel)?;
                let rows_per_tg = 16u64;
                let tg_threads = 128u64;
                let num_tgs = (n as u64 + rows_per_tg - 1) / rows_per_tg;
                self.ctx.queue.enqueue(|enc| {
                    enc.dispatch_inline(pipeline, &[&self.buffer, &other.buffer, &out.buffer],
                        &[bytemuck::cast_slice(&dims)],
                        [num_tgs, 1, 1], [tg_threads, 1, 1])
                })?;
            } else if f16_weights {
                let dims_buf = self.ctx.device.create_buffer_with_data(&dims)?;
                let pipeline = self.ctx.shaders.get("matvec_scalar_f16w_f32")?;
                self.ctx.queue.enqueue(|enc| {
                    enc.dispatch_1d(pipeline, &[&self.buffer, &other.buffer, &out.buffer, &dims_buf], n as u64)
                })?;
            } else {
                let dims3 = [m, n, k];
                let pipeline = self.ctx.shaders.get("matmul_a_bt_f32")?;
                let grid_x = ((n as u64) + TILE_SIZE - 1) / TILE_SIZE;
                let grid_y = 1u64;
                self.ctx.queue.enqueue(|enc| {
                    enc.dispatch_inline(
                        pipeline,
                        &[&self.buffer, &other.buffer, &out.buffer],
                        &[bytemuck::cast_slice(&dims3)],
                        [grid_x, grid_y, 1],
                        [TILE_SIZE, TILE_SIZE, 1],
                    )
                })?;
            }
        } else {
            let dims = [m, n, k];
            let kernel = if f16_weights { "matmul_a_bt_f16w_f32" } else { "matmul_a_bt_f32" };
            let pipeline = self.ctx.shaders.get(kernel)?;
            let grid_x = ((n as u64) + TILE_SIZE - 1) / TILE_SIZE;
            let grid_y = ((m as u64) + TILE_SIZE - 1) / TILE_SIZE;
            self.ctx.queue.enqueue(|enc| {
                enc.dispatch_inline(
                    pipeline,
                    &[&self.buffer, &other.buffer, &out.buffer],
                    &[bytemuck::cast_slice(&dims)],
                    [grid_x, grid_y, 1],
                    [TILE_SIZE, TILE_SIZE, 1],
                )
            })?;
        }
        Ok(out)
    }

    /// Fused matmul + bias add for M=1 (decode). Uses SIMD matvec with bias folded in.
    /// Falls back to matmul_a_bt + broadcast_add_bias for M>1.
    pub fn matmul_a_bt_bias(&self, weight: &Tensor, bias: &Tensor) -> gpux_core::Result<Tensor> {
        // Route quantized to the quantized fused-bias path.
        if weight.dtype.is_quantized() {
            return self.matmul_a_bt_bias_quantized(weight, bias);
        }

        let m = self.shape.dims()[self.ndim() - 2] as u32;
        let k = self.shape.last_dim() as u32;
        let n = weight.shape.dims()[weight.ndim() - 2] as u32;
        let f16_weights = weight.dtype == DType::F16;

        if m == 1 && use_simd_matvec() {
            let out_shape = Shape::new(vec![1, n as usize]);
            let out = Tensor::uninit(&self.ctx, out_shape)?;
            let dims = [n, k];
            let kernel = if f16_weights { "matvec_a_bt_bias_f16w_f32" } else { "matvec_a_bt_bias_f32" };
            let pipeline = self.ctx.shaders.get(kernel)?;
            let rows_per_tg = 16u64;
            let tg_threads = 128u64;
            let num_tgs = (n as u64 + rows_per_tg - 1) / rows_per_tg;
            self.ctx.queue.enqueue(|enc| {
                enc.dispatch_inline(pipeline,
                    &[&self.buffer, &weight.buffer, &bias.buffer, &out.buffer],
                    &[bytemuck::cast_slice(&dims)],
                    [num_tgs, 1, 1], [tg_threads, 1, 1])
            })?;
            return Ok(out);
        }

        // Fallback: separate matmul + bias
        let out = self.matmul_a_bt(weight)?;
        out.broadcast_add_bias(bias)
    }

    // ===== Batched matrix multiplication =====

    pub fn batched_matmul(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        let b = self.shape.dims()[0] as u32;
        let m = self.shape.dims()[1] as u32;
        let k = self.shape.dims()[2] as u32;
        let n = other.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![b as usize, m as usize, n as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[b, m, n, k])?;
        let pipeline = self.ctx.shaders.get("batched_matmul_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &other.buffer, &out.buffer, &dims],
                [
                    ((n as u64) + TILE_SIZE - 1) / TILE_SIZE,
                    ((m as u64) + TILE_SIZE - 1) / TILE_SIZE,
                    b as u64,
                ],
                [TILE_SIZE, TILE_SIZE, 1],
            )
        })?;
        Ok(out)
    }

    pub fn batched_matmul_a_bt(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        let b = self.shape.dims()[0] as u32;
        let m = self.shape.dims()[1] as u32;
        let k = self.shape.dims()[2] as u32;
        let n = other.shape.dims()[1] as u32;

        let out_shape = Shape::new(vec![b as usize, m as usize, n as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[b, m, n, k])?;
        let pipeline = self.ctx.shaders.get("batched_matmul_a_bt_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &other.buffer, &out.buffer, &dims],
                [
                    ((n as u64) + TILE_SIZE - 1) / TILE_SIZE,
                    ((m as u64) + TILE_SIZE - 1) / TILE_SIZE,
                    b as u64,
                ],
                [TILE_SIZE, TILE_SIZE, 1],
            )
        })?;
        Ok(out)
    }

    pub fn batched_matmul_at_b(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        let b = self.shape.dims()[0] as u32;
        let k = self.shape.dims()[1] as u32;
        let m = self.shape.dims()[2] as u32;
        let n = other.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![b as usize, m as usize, n as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[b, m, n, k])?;
        let pipeline = self.ctx.shaders.get("batched_matmul_at_b_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &other.buffer, &out.buffer, &dims],
                [
                    ((n as u64) + TILE_SIZE - 1) / TILE_SIZE,
                    ((m as u64) + TILE_SIZE - 1) / TILE_SIZE,
                    b as u64,
                ],
                [TILE_SIZE, TILE_SIZE, 1],
            )
        })?;
        Ok(out)
    }

    // ===== Reduction ops =====

    pub fn sum_all(&self) -> gpux_core::Result<Tensor> {
        let out = Tensor::zeros(&self.ctx, Shape::scalar())?;
        let n_buf = self.ctx.device.create_buffer_with_data(&[self.numel() as u32])?;
        let pipeline = self.ctx.shaders.get("sum_all_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &n_buf], [1, 1, 1], [256, 1, 1])
        })?;
        Ok(out)
    }

    pub fn sum_rows(&self) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, Shape::new(vec![rows as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let pipeline = self.ctx.shaders.get("sum_rows_f32")?;
        let (grid, tg) = threadgroup_1d(rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    pub fn sum_cols(&self) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, Shape::new(vec![cols as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let pipeline = self.ctx.shaders.get("sum_cols_f32")?;
        let (grid, tg) = threadgroup_1d(cols as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    pub fn mean_rows(&self) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, Shape::new(vec![rows as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let pipeline = self.ctx.shaders.get("mean_rows_f32")?;
        let (grid, tg) = threadgroup_1d(rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    pub fn argmax_rows(&self) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out_size = rows as usize * std::mem::size_of::<u32>();
        let mut out_buf = self.ctx.device.create_buffer(out_size)?;
        out_buf.fill_zero();
        let out = Tensor::new(out_buf, Shape::new(vec![rows as usize]), self.ctx.clone());
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let pipeline = self.ctx.shaders.get("argmax_rows_f32")?;
        let (grid, tg) = threadgroup_1d(rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    // ===== Softmax =====

    pub fn softmax(&self) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let pipeline = self.ctx.shaders.get("softmax_f32")?;
        let (grid, tg) = threadgroup_1d(rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    pub fn softmax_backward(&self, softmax_output: &Tensor) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let pipeline = self.ctx.shaders.get("softmax_backward_f32")?;
        let (grid, tg) = threadgroup_1d(rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &softmax_output.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    // ===== Cross entropy =====

    pub fn cross_entropy_loss(&self, targets: &Tensor) -> gpux_core::Result<(Tensor, Tensor)> {
        let batch = self.shape.dims()[0] as u32;
        let vocab = self.shape.dims()[1] as u32;

        let losses = Tensor::uninit(&self.ctx, Shape::new(vec![batch as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[batch, vocab])?;
        let pipeline = self.ctx.shaders.get("cross_entropy_forward_f32")?;
        let (grid, tg) = threadgroup_1d(batch as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &targets.buffer, &losses.buffer, &dims], grid, tg)
        })?;

        let softmax_out = self.softmax()?;
        Ok((losses.sum_all()?.mul_scalar(1.0 / batch as f32)?, softmax_out))
    }

    // ===== LayerNorm =====

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> gpux_core::Result<(Tensor, Tensor, Tensor)> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let mean_out = Tensor::uninit(&self.ctx, Shape::new(vec![rows as usize]))?;
        let rstd_out = Tensor::uninit(&self.ctx, Shape::new(vec![rows as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let eps_buf = self.ctx.device.create_buffer_with_data(&[eps])?;
        let pipeline = self.ctx.shaders.get("layernorm_forward_f32")?;
        let (grid, tg) = threadgroup_1d(rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[
                    &self.buffer, &gamma.buffer, &beta.buffer,
                    &out.buffer, &mean_out.buffer, &rstd_out.buffer,
                    &dims, &eps_buf,
                ],
                grid,
                tg,
            )
        })?;
        Ok((out, mean_out, rstd_out))
    }

    // ===== RMSNorm =====

    pub fn rms_norm(&self, gamma: &Tensor, eps: f32) -> gpux_core::Result<(Tensor, Tensor)> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let rstd_out = Tensor::uninit(&self.ctx, Shape::new(vec![rows as usize]))?;
        let dims = [rows, cols];
        let eps_arr = [eps];
        let f16_gamma = gamma.dtype == DType::F16;
        let kernel = if f16_gamma { "rmsnorm_forward_f16w_f32" } else { "rmsnorm_forward_f32" };
        let pipeline = self.ctx.shaders.get(kernel)?;
        // SIMD-cooperative kernel: 1 TG per row, 128 threads (4 SG) per TG.
        // f32 fallback keeps the legacy 1-thread-per-row layout (training only).
        let (grid, tg) = if f16_gamma {
            ([rows as u64, 1, 1], [128u64, 1, 1])
        } else {
            threadgroup_1d(rows as u64)
        };

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_inline(
                pipeline,
                &[&self.buffer, &gamma.buffer, &out.buffer, &rstd_out.buffer],
                &[bytemuck::cast_slice(&dims), bytemuck::cast_slice(&eps_arr)],
                grid,
                tg,
            )
        })?;
        Ok((out, rstd_out))
    }

    // ===== RoPE =====

    pub fn rope(&self, inv_freq: &Tensor) -> gpux_core::Result<Tensor> {
        let bh = self.shape.dims()[0] as u32;
        let t = self.shape.dims()[1] as u32;
        let d = self.shape.dims()[2] as u32;

        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = self.ctx.device.create_buffer_with_data(&[bh, t, d])?;
        let pipeline = self.ctx.shaders.get("rope_fused_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &out.buffer, &dims, &inv_freq.buffer],
                [
                    ((d as u64 / 2) + 7) / 8,
                    ((t as u64) + 7) / 8,
                    bh as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    // ===== Transpose =====

    pub fn transpose_2d(&self) -> gpux_core::Result<Tensor> {
        assert_eq!(self.ndim(), 2, "transpose_2d requires 2D tensor");
        let m = self.shape.dims()[0] as u32;
        let n = self.shape.dims()[1] as u32;
        let out_shape = Shape::new(vec![n as usize, m as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[m, n])?;
        let pipeline = self.ctx.shaders.get("transpose_2d_f32")?;
        let (grid, tg) = threadgroup_2d(n as u64, m as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    // ===== Broadcast add bias =====

    pub fn broadcast_add_bias(&self, bias: &Tensor) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols = self.shape.cols() as u32;
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;
        let pipeline = self.ctx.shaders.get("broadcast_add_bias_f32")?;
        let (grid, tg) = threadgroup_2d(cols as u64, rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &bias.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    // ===== Accumulate (in-place add) =====

    pub fn accumulate(&self, other: &Tensor) -> gpux_core::Result<()> {
        let pipeline = self.ctx.shaders.get("accumulate_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&other.buffer, &self.buffer], n)
        })?;
        Ok(())
    }

    // ===== Embedding =====

    pub fn embedding_lookup(&self, indices: &Tensor) -> gpux_core::Result<Tensor> {
        let total_tokens = indices.numel() as u32;
        let dim = self.shape.dims()[1] as u32;
        let mut out_shape_dims = indices.shape.dims().to_vec();
        out_shape_dims.push(dim as usize);
        let out = Tensor::uninit(&self.ctx, Shape::new(out_shape_dims))?;
        let dims = [total_tokens, dim];
        let kernel = match self.dtype {
            DType::F16  => "embedding_forward_f16w_f32",
            DType::Q4_0 => "embedding_forward_q4_0_f32",
            DType::Q8_0 => "embedding_forward_q8_0_f32",
            // Q4_K embedding lookup not implemented; the example's quantize
            // selection keeps the embedding at f16 in this case.
            DType::Q4_K => panic!("Q4_K embedding lookup not implemented; \
                                   keep embed_tokens at f16 (skip in quantize loop)"),
            DType::F32  => "embedding_forward_f32",
        };
        let pipeline = self.ctx.shaders.get(kernel)?;
        let (grid, tg) = threadgroup_2d(dim as u64, total_tokens as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_inline(pipeline, &[&indices.buffer, &self.buffer, &out.buffer],
                &[bytemuck::cast_slice(&dims)], grid, tg)
        })?;
        Ok(out)
    }

    // ===== Attention helpers =====

    pub fn reshape_for_attention(&self, batch: usize, seq_len: usize, n_heads: usize, head_dim: usize) -> gpux_core::Result<Tensor> {
        // When seq_len=1 (decode), memory layout [B,1,H*D] and [B*H,1,D] are identical — zero-copy
        if seq_len == 1 {
            return Ok(self.reshape(vec![batch * n_heads, 1, head_dim]));
        }

        let out_shape = Shape::new(vec![batch * n_heads, seq_len, head_dim]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[
            batch as u32, seq_len as u32, n_heads as u32, head_dim as u32,
        ])?;
        let pipeline = self.ctx.shaders.get("reshape_for_attention_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &out.buffer, &dims],
                [
                    ((head_dim as u64) + 7) / 8,
                    ((seq_len as u64) + 7) / 8,
                    (batch * n_heads) as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    pub fn reshape_from_attention(&self, batch: usize, seq_len: usize, n_heads: usize, head_dim: usize) -> gpux_core::Result<Tensor> {
        // When seq_len=1 (decode), memory layout [B*H,1,D] and [B,1,H*D] are identical — zero-copy
        if seq_len == 1 {
            return Ok(self.reshape(vec![batch, 1, n_heads * head_dim]));
        }

        let out_shape = Shape::new(vec![batch, seq_len, n_heads * head_dim]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[
            batch as u32, seq_len as u32, n_heads as u32, head_dim as u32,
        ])?;
        let pipeline = self.ctx.shaders.get("reshape_from_attention_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &out.buffer, &dims],
                [
                    ((head_dim as u64) + 7) / 8,
                    ((seq_len as u64) + 7) / 8,
                    (batch * n_heads) as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    pub fn attention_scores(&self, keys: &Tensor, scale: f32) -> gpux_core::Result<Tensor> {
        let bh = self.shape.dims()[0] as u32;
        let t = self.shape.dims()[1] as u32;
        let d = self.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![bh as usize, t as usize, t as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[bh, t, d])?;
        let scale_buf = self.ctx.device.create_buffer_with_data(&[scale])?;
        let pipeline = self.ctx.shaders.get("attention_scores_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &keys.buffer, &out.buffer, &dims, &scale_buf],
                [
                    ((t as u64) + 7) / 8,
                    ((t as u64) + 7) / 8,
                    bh as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    pub fn attention_weighted_values(&self, values: &Tensor) -> gpux_core::Result<Tensor> {
        let bh = self.shape.dims()[0] as u32;
        let t = self.shape.dims()[1] as u32;
        let d = values.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![bh as usize, t as usize, d as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[bh, t, d])?;
        let pipeline = self.ctx.shaders.get("attention_weighted_values_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &values.buffer, &out.buffer, &dims],
                [
                    ((d as u64) + 7) / 8,
                    ((t as u64) + 7) / 8,
                    bh as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    // ===== GQA attention =====

    pub fn gqa_attention_scores(&self, keys: &Tensor, scale: f32, n_rep: usize) -> gpux_core::Result<Tensor> {
        let bhq = self.shape.dims()[0] as u32;
        let t = self.shape.dims()[1] as u32;
        let d = self.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![bhq as usize, t as usize, t as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[bhq, t, d, n_rep as u32])?;
        let scale_buf = self.ctx.device.create_buffer_with_data(&[scale])?;
        let pipeline = self.ctx.shaders.get("gqa_attention_scores_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &keys.buffer, &out.buffer, &dims, &scale_buf],
                [
                    ((t as u64) + 7) / 8,
                    ((t as u64) + 7) / 8,
                    bhq as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    pub fn gqa_attention_weighted_values(&self, values: &Tensor, n_rep: usize) -> gpux_core::Result<Tensor> {
        let bhq = self.shape.dims()[0] as u32;
        let t = self.shape.dims()[1] as u32;
        let d = values.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![bhq as usize, t as usize, d as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[bhq, t, d, n_rep as u32])?;
        let pipeline = self.ctx.shaders.get("gqa_attention_weighted_values_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &values.buffer, &out.buffer, &dims],
                [
                    ((d as u64) + 7) / 8,
                    ((t as u64) + 7) / 8,
                    bhq as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    // ===== Position embedding =====

    pub fn add_position_embedding(&self, pos_weight: &Tensor) -> gpux_core::Result<Tensor> {
        let batch = self.shape.dims()[0] as u32;
        let seq_len = self.shape.dims()[1] as u32;
        let dim = self.shape.dims()[2] as u32;
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = self.ctx.device.create_buffer_with_data(&[batch, seq_len, dim])?;
        let pipeline = self.ctx.shaders.get("add_position_embedding_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &pos_weight.buffer, &out.buffer, &dims],
                [
                    ((dim as u64) + 7) / 8,
                    ((seq_len as u64) + 7) / 8,
                    batch as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    // ===== Optimizer ops =====

    pub fn adamw_step(
        &self,
        grad: &Tensor,
        m: &Tensor,
        v: &Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> gpux_core::Result<()> {
        let beta1_t = beta1.powi(step as i32);
        let beta2_t = beta2.powi(step as i32);
        let n_as_float = f32::from_bits(self.numel() as u32);
        let config = self.ctx.device.create_buffer_with_data(&[
            lr, beta1, beta2, eps, weight_decay, beta1_t, beta2_t, n_as_float,
        ])?;
        let pipeline = self.ctx.shaders.get("adamw_step_f32")?;
        let n = self.numel() as u64;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(
                pipeline,
                &[&self.buffer, &grad.buffer, &m.buffer, &v.buffer, &config],
                n,
            )
        })?;
        Ok(())
    }

    pub fn clip_grad_norm(&self, max_norm: f32) -> gpux_core::Result<()> {
        let data = self.to_vec();
        let total_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let clip_coeff = (max_norm / (total_norm + 1e-6)).min(1.0);
        if clip_coeff < 1.0 {
            let coeff_buf = self.ctx.device.create_buffer_with_data(&[clip_coeff])?;
            let pipeline = self.ctx.shaders.get("clip_grad_norm_f32")?;
            let n = self.numel() as u64;
            self.ctx.queue.enqueue(|enc| {
                enc.dispatch_1d(pipeline, &[&self.buffer, &coeff_buf], n)
            })?;
        }
        Ok(())
    }

    // ===== Cross entropy backward =====

    pub fn cross_entropy_backward(
        &self,
        targets: &Tensor,
        grad_scale: f32,
    ) -> gpux_core::Result<Tensor> {
        let batch = self.shape.dims()[0] as u32;
        let vocab = self.shape.dims()[1] as u32;
        let grad = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = self.ctx.device.create_buffer_with_data(&[batch, vocab])?;
        let scale_buf = self.ctx.device.create_buffer_with_data(&[grad_scale])?;
        let pipeline = self.ctx.shaders.get("cross_entropy_backward_f32")?;
        let (grid, tg) = threadgroup_2d(vocab as u64, batch as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &targets.buffer, &grad.buffer, &dims, &scale_buf],
                grid,
                tg,
            )
        })?;
        Ok(grad)
    }

    // ===== LayerNorm backward =====

    pub fn layernorm_backward(
        &self,
        input: &Tensor,
        gamma: &Tensor,
        mean: &Tensor,
        rstd: &Tensor,
    ) -> gpux_core::Result<(Tensor, Tensor, Tensor)> {
        let rows = input.shape.rows() as u32;
        let cols = input.shape.cols() as u32;

        let grad_input = Tensor::uninit(&self.ctx, input.shape.clone())?;
        let grad_gamma = Tensor::uninit(&self.ctx, gamma.shape.clone())?;
        let grad_beta = Tensor::uninit(&self.ctx, gamma.shape.clone())?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols])?;

        // grad_input
        let pipeline = self.ctx.shaders.get("layernorm_backward_f32")?;
        let (grid, tg) = threadgroup_1d(rows as u64);
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[
                    &self.buffer, &input.buffer, &gamma.buffer,
                    &mean.buffer, &rstd.buffer, &grad_input.buffer, &dims,
                ],
                grid,
                tg,
            )
        })?;

        // grad_gamma and grad_beta
        let pipeline2 = self.ctx.shaders.get("layernorm_grad_gamma_beta_f32")?;
        let (grid2, tg2) = threadgroup_1d(cols as u64);
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline2,
                &[
                    &self.buffer, &input.buffer, &mean.buffer, &rstd.buffer,
                    &grad_gamma.buffer, &grad_beta.buffer, &dims,
                ],
                grid2,
                tg2,
            )
        })?;

        Ok((grad_input, grad_gamma, grad_beta))
    }

    // ===== GELU backward =====

    pub fn gelu_backward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let pipeline = self.ctx.shaders.get("gelu_backward_f32")?;
        let n = self.numel() as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_1d(pipeline, &[&self.buffer, &input.buffer, &out.buffer], n)
        })?;
        Ok(out)
    }

    // ===== Embedding backward =====

    pub fn embedding_backward(&self, indices: &Tensor, vocab_size: usize) -> gpux_core::Result<Tensor> {
        let total_tokens = indices.numel() as u32;
        let dim = self.shape.last_dim() as u32;
        let grad_weight = Tensor::uninit(&self.ctx, Shape::new(vec![vocab_size, dim as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[total_tokens, dim])?;
        let pipeline = self.ctx.shaders.get("embedding_backward_f32")?;
        let (grid, tg) = threadgroup_2d(dim as u64, total_tokens as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&indices.buffer, &self.buffer, &grad_weight.buffer, &dims],
                grid,
                tg,
            )
        })?;
        Ok(grad_weight)
    }

    // ===== Slice and concat =====

    pub fn slice_cols(&self, start: usize, end: usize) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let total_cols = self.shape.cols() as u32;
        let out_cols = (end - start) as u32;

        let out = Tensor::uninit(&self.ctx, Shape::new(vec![rows as usize, out_cols as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, total_cols, start as u32, end as u32])?;
        let pipeline = self.ctx.shaders.get("slice_cols_f32")?;
        let (grid, tg) = threadgroup_2d(out_cols as u64, rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    pub fn concat_cols(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        let rows = self.shape.rows() as u32;
        let cols_a = self.shape.cols() as u32;
        let cols_b = other.shape.cols() as u32;
        let total = cols_a + cols_b;

        let out = Tensor::uninit(&self.ctx, Shape::new(vec![rows as usize, total as usize]))?;
        let dims = self.ctx.device.create_buffer_with_data(&[rows, cols_a, cols_b])?;
        let pipeline = self.ctx.shaders.get("concat_cols_f32")?;
        let (grid, tg) = threadgroup_2d(total as u64, rows as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&self.buffer, &other.buffer, &out.buffer, &dims], grid, tg)
        })?;
        Ok(out)
    }

    pub fn split_qkv(&self, embed_dim: usize) -> gpux_core::Result<(Tensor, Tensor, Tensor)> {
        let q = self.slice_cols(0, embed_dim)?;
        let k = self.slice_cols(embed_dim, 2 * embed_dim)?;
        let v = self.slice_cols(2 * embed_dim, 3 * embed_dim)?;
        Ok((q, k, v))
    }

    pub fn concat_qkv(q: &Tensor, k: &Tensor, v: &Tensor) -> gpux_core::Result<Tensor> {
        let qk = q.concat_cols(k)?;
        qk.concat_cols(v)
    }

    // ===== KV Cache ops =====

    pub fn rope_with_offset(&self, inv_freq: &Tensor, offset: usize) -> gpux_core::Result<Tensor> {
        let bh = self.shape.dims()[0] as u32;
        let t = self.shape.dims()[1] as u32;
        let d = self.shape.dims()[2] as u32;

        let out = Tensor::uninit(&self.ctx, self.shape.clone())?;
        let dims = [bh, t, d, offset as u32];
        let pipeline = self.ctx.shaders.get("rope_with_offset_fused_f32")?;

        // Note: rope_with_offset_fused_f32 expects dims at index 2, inv_freq at index 3.
        // We can't trivially use dispatch_inline because params aren't at the tail —
        // bind dims as a regular small buffer and inv_freq as a buffer.
        // (This kernel could be re-ordered later for setBytes.)
        let dims_buf = self.ctx.device.create_buffer_with_data(&dims)?;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &out.buffer, &dims_buf, &inv_freq.buffer],
                [
                    ((d as u64 / 2) + 7) / 8,
                    ((t as u64) + 7) / 8,
                    bh as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    pub fn kv_cache_append(&self, cache: &Tensor, cache_pos: usize, max_seq: usize) -> gpux_core::Result<()> {
        let bhkv = self.shape.dims()[0] as u32;
        let new_tok = self.shape.dims()[1] as u32;
        let d = self.shape.dims()[2] as u32;

        let dims = self.ctx.device.create_buffer_with_data(&[
            bhkv, max_seq as u32, d, cache_pos as u32, new_tok,
        ])?;
        let pipeline = self.ctx.shaders.get("kv_cache_append_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &cache.buffer, &dims],
                [
                    ((d as u64) + 7) / 8,
                    ((new_tok as u64) + 7) / 8,
                    bhkv as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(())
    }

    /// Fused RoPE + KV cache append (decode path, T=1).
    /// Inputs:
    ///   `qk_in [n_qk_heads, T, D]` — Q+K concatenated (n_qk = n_q + n_kv), pre-RoPE
    ///   `v_in  [n_kv_heads, T, D]` — V (no RoPE)
    /// Outputs:
    ///   - `q_out` (returned): [n_q_heads, T, D] post-RoPE Q for attention
    ///   - K/V written into the caches at `cache_pos` (in place)
    pub fn rope_kv_append_fused(
        qk_in: &Tensor,
        v_in: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        inv_freq: &Tensor,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        cache_pos: usize,
        max_seq: usize,
    ) -> gpux_core::Result<Tensor> {
        let t = qk_in.shape.dims()[1] as u32;
        let d = head_dim as u32;

        let q_out = Tensor::uninit(
            &qk_in.ctx,
            Shape::new(vec![n_q_heads, t as usize, head_dim]),
        )?;

        let dims = [
            n_q_heads as u32,
            n_kv_heads as u32,
            t,
            d,
            cache_pos as u32,
            max_seq as u32,
        ];
        let dims_buf = qk_in.ctx.device.create_buffer_with_data(&dims)?;
        let pipeline = qk_in.ctx.shaders.get("rope_kv_append_fused_f32")?;

        let n_total = (n_q_heads + 2 * n_kv_heads) as u64;
        let half_d = (d as u64) / 2;
        let grid_x = (half_d + 7) / 8;
        let grid_y = ((t as u64) + 7) / 8;
        let grid_z = n_total;

        qk_in.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[
                    &qk_in.buffer,
                    &v_in.buffer,
                    &q_out.buffer,
                    &k_cache.buffer,
                    &v_cache.buffer,
                    &inv_freq.buffer,
                    &dims_buf,
                ],
                [grid_x, grid_y, grid_z],
                [8, 8, 1],
            )
        })?;
        Ok(q_out)
    }

    /// Fused K+V cache append: writes both K and V in a single dispatch.
    pub fn kv_cache_append_fused(
        new_k: &Tensor, new_v: &Tensor,
        k_cache: &Tensor, v_cache: &Tensor,
        cache_pos: usize, max_seq: usize,
    ) -> gpux_core::Result<()> {
        let bhkv = new_k.shape.dims()[0] as u32;
        let new_tok = new_k.shape.dims()[1] as u32;
        let d = new_k.shape.dims()[2] as u32;

        let dims = [bhkv, max_seq as u32, d, cache_pos as u32, new_tok];
        let pipeline = new_k.ctx.shaders.get("kv_cache_append_fused_f32")?;

        new_k.ctx.queue.enqueue(|enc| {
            enc.dispatch_inline(
                pipeline,
                &[&new_k.buffer, &new_v.buffer, &k_cache.buffer, &v_cache.buffer],
                &[bytemuck::cast_slice(&dims)],
                [
                    ((d as u64) + 7) / 8,
                    ((new_tok as u64) + 7) / 8,
                    bhkv as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(())
    }

    /// Fused decode attention: scores + softmax + weighted values in one dispatch.
    /// Uses online softmax — no intermediate score/weight buffers needed.
    pub fn fused_attention_decode(
        &self,
        k_cache: &Tensor,
        v_cache: &Tensor,
        scale: f32,
        n_rep: usize,
        t_kv: usize,
        max_seq: usize,
    ) -> gpux_core::Result<Tensor> {
        let bhq = self.shape.dims()[0] as u32;
        let d = self.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![bhq as usize, 1, d as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = [bhq, t_kv as u32, d, n_rep as u32, max_seq as u32];
        let scale_arr = [scale];
        let pipeline = self.ctx.shaders.get("fused_attention_decode_f32")?;

        // One threadgroup per query head, 8 SIMD groups × 32 threads = 256 threads per TG
        let tg_threads = 256u64;
        let num_tgs = bhq as u64;
        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_inline(
                pipeline,
                &[&self.buffer, &k_cache.buffer, &v_cache.buffer, &out.buffer],
                &[bytemuck::cast_slice(&dims), bytemuck::cast_slice(&scale_arr)],
                [num_tgs, 1, 1],
                [tg_threads, 1, 1],
            )
        })?;
        Ok(out)
    }

    pub fn gqa_attention_scores_cached(
        &self,
        k_cache: &Tensor,
        scale: f32,
        n_rep: usize,
        t_kv: usize,
        max_seq: usize,
    ) -> gpux_core::Result<Tensor> {
        let bhq = self.shape.dims()[0] as u32;
        let t_q = self.shape.dims()[1] as u32;
        let d = self.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![bhq as usize, t_q as usize, t_kv]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[
            bhq, t_q, t_kv as u32, d, n_rep as u32, max_seq as u32,
        ])?;
        let scale_buf = self.ctx.device.create_buffer_with_data(&[scale])?;
        let pipeline = self.ctx.shaders.get("gqa_attention_scores_cached_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &k_cache.buffer, &out.buffer, &dims, &scale_buf],
                [
                    ((t_kv as u64) + 7) / 8,
                    ((t_q as u64) + 7) / 8,
                    bhq as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    pub fn gqa_attention_weighted_values_cached(
        &self,
        v_cache: &Tensor,
        n_rep: usize,
        t_kv: usize,
        max_seq: usize,
    ) -> gpux_core::Result<Tensor> {
        let bhq = self.shape.dims()[0] as u32;
        let t_q = self.shape.dims()[1] as u32;
        let d = v_cache.shape.dims()[2] as u32;

        let out_shape = Shape::new(vec![bhq as usize, t_q as usize, d as usize]);
        let out = Tensor::uninit(&self.ctx, out_shape)?;
        let dims = self.ctx.device.create_buffer_with_data(&[
            bhq, t_q, t_kv as u32, d, n_rep as u32, max_seq as u32,
        ])?;
        let pipeline = self.ctx.shaders.get("gqa_attention_weighted_values_cached_f32")?;

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&self.buffer, &v_cache.buffer, &out.buffer, &dims],
                [
                    ((d as u64) + 7) / 8,
                    ((t_q as u64) + 7) / 8,
                    bhq as u64,
                ],
                [8, 8, 1],
            )
        })?;
        Ok(out)
    }

    // ===== Quantization ops =====

    pub fn quantize_q8_0(&self) -> gpux_core::Result<Tensor> {
        // Quantize kernel reads float32 — promote f16 inputs first.
        let src = if self.dtype == DType::F16 { self.to_f32()? } else { self.clone() };
        let numel = src.numel();
        assert!(numel % 32 == 0, "elements must be multiple of 32 for Q8_0");
        let total_blocks = (numel / 32) as u32;
        let out_bytes = DType::Q8_0.bytes_for_elements(numel);
        let out_buf = src.ctx.device.create_buffer(out_bytes)?;

        let k = src.shape.last_dim() as u32;
        let dims = src.ctx.device.create_buffer_with_data(&[total_blocks])?;
        let k_buf = src.ctx.device.create_buffer_with_data(&[k])?;
        let pipeline = src.ctx.shaders.get("quantize_q8_0_f32")?;
        let (grid, tg) = threadgroup_1d(total_blocks as u64);

        src.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&src.buffer, &out_buf, &dims, &k_buf],
                grid,
                tg,
            )
        })?;

        Ok(Tensor::new_with_dtype(out_buf, src.shape.clone(), src.ctx.clone(), DType::Q8_0))
    }

    /// Quantize a tensor to Q4_K_M (super-blocks of 256 elements with per-sub-block scales).
    /// The tensor's last dim must be a multiple of 256.
    pub fn quantize_q4_k(&self) -> gpux_core::Result<Tensor> {
        // Quantize kernel reads f32 — promote f16 inputs first.
        let src = if self.dtype == DType::F16 { self.to_f32()? } else { self.clone() };
        let numel = src.numel();
        assert!(numel % 256 == 0, "elements must be multiple of 256 for Q4_K");
        let total_super_blocks = (numel / 256) as u32;
        let out_bytes = DType::Q4_K.bytes_for_elements(numel);
        let out_buf = src.ctx.device.create_buffer(out_bytes)?;

        let dims = src.ctx.device.create_buffer_with_data(&[total_super_blocks])?;
        let pipeline = src.ctx.shaders.get("quantize_q4_k_f32")?;
        let (grid, tg) = threadgroup_1d(total_super_blocks as u64);

        src.ctx.queue.enqueue(|enc| {
            enc.dispatch(pipeline, &[&src.buffer, &out_buf, &dims], grid, tg)
        })?;

        Ok(Tensor::new_with_dtype(out_buf, src.shape.clone(), src.ctx.clone(), DType::Q4_K))
    }

    pub fn quantize_q4_0(&self) -> gpux_core::Result<Tensor> {
        // Quantize kernel reads float32 — promote f16 inputs first.
        let src = if self.dtype == DType::F16 { self.to_f32()? } else { self.clone() };
        let numel = src.numel();
        assert!(numel % 32 == 0, "elements must be multiple of 32 for Q4_0");
        let total_blocks = (numel / 32) as u32;
        let out_bytes = DType::Q4_0.bytes_for_elements(numel);
        let out_buf = src.ctx.device.create_buffer(out_bytes)?;

        let k = src.shape.last_dim() as u32;
        let dims = src.ctx.device.create_buffer_with_data(&[total_blocks])?;
        let k_buf = src.ctx.device.create_buffer_with_data(&[k])?;
        let pipeline = src.ctx.shaders.get("quantize_q4_0_f32")?;
        let (grid, tg) = threadgroup_1d(total_blocks as u64);

        src.ctx.queue.enqueue(|enc| {
            enc.dispatch(
                pipeline,
                &[&src.buffer, &out_buf, &dims, &k_buf],
                grid,
                tg,
            )
        })?;

        Ok(Tensor::new_with_dtype(out_buf, src.shape.clone(), src.ctx.clone(), DType::Q4_0))
    }

    pub fn matmul_a_bt_quantized(&self, other: &Tensor) -> gpux_core::Result<Tensor> {
        let m = self.shape.dims()[0] as u32;
        let k = self.shape.dims()[1] as u32;
        let n = other.shape.dims()[0] as u32;

        let out = Tensor::uninit(&self.ctx, Shape::new(vec![m as usize, n as usize]))?;

        // SIMD matvec for decode (M=1): 16 rows/TG, 128 threads/TG, ~3.5× faster than scalar.
        if m == 1 && use_simd_matvec() {
            let kernel_name = match other.dtype {
                DType::Q8_0 => "matvec_a_bt_q8_0_f32",
                DType::Q4_0 => "matvec_a_bt_q4_0_f32",
                DType::Q4_K => "matvec_a_bt_q4_k_v2_f32", // lane-per-sub-block, full SIMD util
                _ => return self.matmul_a_bt(other),
            };
            let dims = [n, k];
            let pipeline = self.ctx.shaders.get(kernel_name)?;
            // Q4_0/Q8_0 use 32 rows/TG (Q_SG_PER_TG=8 × Q_ROWS_PER_SG=4) — see quantized.metal
            let rows_per_tg = 32u64;
            let tg_threads = 256u64;
            let num_tgs = (n as u64 + rows_per_tg - 1) / rows_per_tg;
            self.ctx.queue.enqueue(|enc| {
                enc.dispatch_inline(
                    pipeline,
                    &[&self.buffer, &other.buffer, &out.buffer],
                    &[bytemuck::cast_slice(&dims)],
                    [num_tgs, 1, 1],
                    [tg_threads, 1, 1],
                )
            })?;
            return Ok(out);
        }

        // Prefill (M>1). For Q4_0, use the simdgroup_matrix tiled GEMM — each TG
        // produces an 8 × 32 output tile using Apple's hardware matrix-multiply
        // units, with K iterated in 32-element Q4_0 blocks. This scales well to
        // large M (typical RAG / long-context prefill).
        if other.dtype == DType::Q4_0 && use_simd_matvec() {
            // v2: 4 SG, 32x32 output, 4 mc/SG. Empirically best — v3 (8 mc/SG) and
            // v4 (8 mb tiles in flight) hit register pressure and regressed.
            const M_TILE: u32 = 32;
            const N_TILE: u32 = 32;
            const TGMEM: usize = 32 * 32 * 4 + 32 * 32 * 2;

            let pipeline = self.ctx.shaders.get("matmul_a_bt_q4_0_sgm2_f32")?;
            let dims = [m, n, k];
            let grid_x = (n + N_TILE - 1) / N_TILE;
            let grid_y = (m + M_TILE - 1) / M_TILE;
            self.ctx.queue.enqueue(|enc| {
                enc.dispatch_inline_tgmem(
                    pipeline,
                    &[&self.buffer, &other.buffer, &out.buffer],
                    &[bytemuck::cast_slice(&dims)],
                    &[TGMEM],
                    [grid_x as u64, grid_y as u64, 1],
                    [128, 1, 1],
                )
            })?;
            return Ok(out);
        }

        // Q4_K M>1: chunk into sequential M=1 matvec calls. Slow for long
        // prefill, but correct. Replace with a real Q4_K tiled GEMM later.
        if other.dtype == DType::Q4_K {
            let k_us = k as usize;
            let n_us = n as usize;
            let m_us = m as usize;
            let out_data_buf = &out.buffer;
            for row in 0..m_us {
                // Build a [1, K] view by narrowing a copy. Easier: copy out the row.
                let row_input_data: Vec<f32> = self.to_vec()[row * k_us .. (row + 1) * k_us].to_vec();
                let row_input = Tensor::from_data(&self.ctx, vec![1, k_us], &row_input_data)?;
                let row_out = row_input.matmul_a_bt_quantized(other)?;
                let row_out_data = row_out.to_vec();
                // Write back to the output tensor at offset `row * N`.
                let out_bytes_offset = row * n_us * std::mem::size_of::<f32>();
                let dst_ptr = unsafe {
                    (out_data_buf.metal_buffer().contents() as *mut u8)
                        .add(out_data_buf.metal_offset() + out_bytes_offset)
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        row_out_data.as_ptr() as *const u8,
                        dst_ptr,
                        n_us * std::mem::size_of::<f32>(),
                    );
                }
            }
            return Ok(out);
        }

        // Q8_0, large-M Q4_0: legacy scalar tile kernel.
        let dims = [m, n, k];
        let kernel_name = match other.dtype {
            DType::Q8_0 => "matmul_q8_0_f32",
            DType::Q4_0 => "matmul_q4_0_f32",
            _ => return self.matmul_a_bt(other),
        };
        let pipeline = self.ctx.shaders.get(kernel_name)?;
        let (grid, tg) = threadgroup_2d(n as u64, m as u64);

        self.ctx.queue.enqueue(|enc| {
            enc.dispatch_inline(
                pipeline,
                &[&self.buffer, &other.buffer, &out.buffer],
                &[bytemuck::cast_slice(&dims)],
                grid,
                tg,
            )
        })?;
        Ok(out)
    }

    /// Fused quantized matvec + bias for M=1 (decode). Saves a separate bias-add dispatch.
    pub fn matmul_a_bt_bias_quantized(&self, weight: &Tensor, bias: &Tensor) -> gpux_core::Result<Tensor> {
        let m = self.shape.dims()[0] as u32;
        let k = self.shape.last_dim() as u32;
        let n = weight.shape.dims()[0] as u32;

        if m == 1 && use_simd_matvec() {
            let kernel_name = match weight.dtype {
                DType::Q8_0 => "matvec_a_bt_bias_q8_0_f32",
                DType::Q4_0 => "matvec_a_bt_bias_q4_0_f32",
                DType::Q4_K => "matvec_a_bt_bias_q4_k_v2_f32", // lane-per-sub-block
                _ => return self.matmul_a_bt_bias(weight, bias),
            };
            let out = Tensor::uninit(&self.ctx, Shape::new(vec![1, n as usize]))?;
            let dims = [n, k];
            let pipeline = self.ctx.shaders.get(kernel_name)?;
            let rows_per_tg = 32u64;
            let tg_threads = 256u64;
            let num_tgs = (n as u64 + rows_per_tg - 1) / rows_per_tg;
            self.ctx.queue.enqueue(|enc| {
                enc.dispatch_inline(
                    pipeline,
                    &[&self.buffer, &weight.buffer, &bias.buffer, &out.buffer],
                    &[bytemuck::cast_slice(&dims)],
                    [num_tgs, 1, 1],
                    [tg_threads, 1, 1],
                )
            })?;
            return Ok(out);
        }

        // Fallback for prefill: separate matmul + bias add
        let out = self.matmul_a_bt_quantized(weight)?;
        out.broadcast_add_bias(bias)
    }
}
