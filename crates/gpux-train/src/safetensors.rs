use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor, DType};

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

pub fn load_safetensors(
    path: &Path,
    ctx: &Arc<GpuContext>,
) -> gpux_core::Result<HashMap<String, Tensor>> {
    let data = std::fs::read(path)
        .map_err(|e| gpux_core::GpuError::Other(format!("failed to read {}: {}", path.display(), e)))?;

    if data.len() < 8 {
        return Err(gpux_core::GpuError::Other("file too small for safetensors".into()));
    }

    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    if data.len() < 8 + header_len {
        return Err(gpux_core::GpuError::Other("truncated safetensors header".into()));
    }

    let header_str = std::str::from_utf8(&data[8..8 + header_len])
        .map_err(|e| gpux_core::GpuError::Other(format!("invalid header UTF-8: {}", e)))?;

    let header: serde_json::Value = serde_json::from_str(header_str)
        .map_err(|e| gpux_core::GpuError::Other(format!("invalid header JSON: {}", e)))?;

    let tensor_data_start = 8 + header_len;
    let mut tensors = HashMap::new();

    let obj = header.as_object()
        .ok_or_else(|| gpux_core::GpuError::Other("header is not a JSON object".into()))?;

    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }

        let dtype_str = info["dtype"].as_str()
            .ok_or_else(|| gpux_core::GpuError::Other(format!("missing dtype for {}", name)))?;

        let shape: Vec<usize> = info["shape"].as_array()
            .ok_or_else(|| gpux_core::GpuError::Other(format!("missing shape for {}", name)))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let offsets = info["data_offsets"].as_array()
            .ok_or_else(|| gpux_core::GpuError::Other(format!("missing data_offsets for {}", name)))?;
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;

        let abs_start = tensor_data_start + start;
        let abs_end = tensor_data_start + end;

        if abs_end > data.len() {
            return Err(gpux_core::GpuError::Other(format!("tensor {} data out of bounds", name)));
        }

        let raw = &data[abs_start..abs_end];

        let tensor = match dtype_str {
            "F32" => {
                Tensor::from_raw_bytes(ctx, shape, raw, DType::F32)?
            }
            "F16" => {
                Tensor::from_raw_bytes(ctx, shape, raw, DType::F16)?
            }
            "BF16" => {
                let f16_data = bf16_to_f16(raw);
                Tensor::from_raw_bytes(ctx, shape, &f16_data, DType::F16)?
            }
            other => {
                log::warn!("skipping tensor {} with unsupported dtype {}", name, other);
                continue;
            }
        };

        tensors.insert(name.clone(), tensor);
    }

    Ok(tensors)
}

pub fn load_safetensors_dir(
    dir: &Path,
    ctx: &Arc<GpuContext>,
) -> gpux_core::Result<HashMap<String, Tensor>> {
    let mut all_tensors = HashMap::new();

    let mut shard_files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| gpux_core::GpuError::Other(format!("failed to read dir: {}", e)))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();

    shard_files.sort();

    if shard_files.is_empty() {
        return Err(gpux_core::GpuError::Other("no .safetensors files found".into()));
    }

    for path in &shard_files {
        log::info!("loading {}", path.display());
        let tensors = load_safetensors(path, ctx)?;
        all_tensors.extend(tensors);
    }

    Ok(all_tensors)
}

fn bf16_to_f16(raw: &[u8]) -> Vec<u8> {
    let count = raw.len() / 2;
    let mut result = Vec::with_capacity(count * 2);
    for i in 0..count {
        let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        let f32_val = f32::from_bits((bits as u32) << 16);
        let f16_bits = f32_to_f16_bits(f32_val);
        result.extend_from_slice(&f16_bits.to_le_bytes());
    }
    result
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    if exp == 0xFF {
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16;
        }
        let m = (mantissa | 0x00800000) >> (1 - new_exp + 13);
        return (sign | m) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}
