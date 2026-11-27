use metal::{Buffer, DeviceRef, MTLResourceOptions};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct GpuBuffer {
    buffer: Buffer,
    size: usize,
    offset: usize,
}

impl GpuBuffer {
    pub fn new(device: &DeviceRef, size: usize) -> crate::Result<Self> {
        let buffer = device.new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self { buffer, size, offset: 0 })
    }

    pub fn with_data<T: bytemuck::Pod>(device: &DeviceRef, data: &[T]) -> crate::Result<Self> {
        let byte_len = data.len() * std::mem::size_of::<T>();
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            buffer,
            size: byte_len,
            offset: 0,
        })
    }

    /// Create a GpuBuffer aliasing part of a scratch buffer.
    /// The Metal buffer is retained (shared ownership via Obj-C ARC).
    pub(crate) fn from_scratch(scratch_buffer: &Buffer, offset: usize, size: usize) -> Self {
        // to_owned() retains the Obj-C reference, so the scratch buffer stays alive
        let retained = scratch_buffer.to_owned();
        Self {
            buffer: retained,
            size,
            offset,
        }
    }

    /// Create a GpuBuffer aliasing part of a scratch buffer, with initial data.
    pub(crate) fn from_scratch_with_data<T: bytemuck::Pod>(
        scratch_buffer: &Buffer,
        offset: usize,
        data: &[T],
    ) -> Self {
        let byte_len = data.len() * std::mem::size_of::<T>();
        let retained = scratch_buffer.to_owned();
        let dst = unsafe { (retained.contents() as *mut u8).add(offset) };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dst, byte_len);
        }
        Self {
            buffer: retained,
            size: byte_len,
            offset,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn metal_offset(&self) -> usize {
        self.offset
    }

    fn contents_ptr(&self) -> *mut u8 {
        unsafe { (self.buffer.contents() as *mut u8).add(self.offset) }
    }

    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        let ptr = self.contents_ptr() as *const T;
        let len = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        let ptr = self.contents_ptr() as *mut T;
        let len = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    pub fn copy_from_slice<T: bytemuck::Pod>(&mut self, data: &[T]) {
        let byte_len = data.len() * std::mem::size_of::<T>();
        assert!(byte_len <= self.size, "data too large for buffer");
        let dst = self.contents_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dst, byte_len);
        }
    }

    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn read_to_vec<T: bytemuck::Pod + Clone>(&self) -> Vec<T> {
        self.as_slice::<T>().to_vec()
    }

    pub fn write_data<T: bytemuck::Pod>(&self, data: &[T]) {
        let byte_len = data.len() * std::mem::size_of::<T>();
        assert!(byte_len <= self.size, "data too large for buffer");
        let dst = self.contents_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dst, byte_len);
        }
    }

    pub fn fill_zero(&mut self) {
        let dst = self.contents_ptr();
        unsafe {
            std::ptr::write_bytes(dst, 0, self.size);
        }
    }

    /// Zero-copy view into a sub-region of this buffer.
    /// Retains the underlying Metal buffer via Obj-C ARC.
    pub fn narrow(&self, byte_offset: usize, byte_len: usize) -> Self {
        assert!(
            byte_offset + byte_len <= self.size,
            "narrow: offset {} + len {} > buffer size {}",
            byte_offset, byte_len, self.size
        );
        Self {
            buffer: self.buffer.to_owned(),
            size: byte_len,
            offset: self.offset + byte_offset,
        }
    }
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

/// Bump allocator backed by a single pre-allocated Metal buffer.
/// Reset between tokens (after synchronize) to reclaim all scratch memory.
/// Must be activated via `set_active(true)` after model loading to avoid
/// allocating long-lived buffers (weights, KV cache) from scratch.
pub struct ScratchAllocator {
    buffer: Buffer,
    capacity: usize,
    offset: AtomicUsize,
    active: std::sync::atomic::AtomicBool,
}

impl ScratchAllocator {
    pub fn new(device: &DeviceRef, capacity: usize) -> Self {
        let buffer = device.new_buffer(
            capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            buffer,
            capacity,
            offset: AtomicUsize::new(0),
            active: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Enable/disable scratch allocation. Disable during weight loading,
    /// enable during decode.
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Allocate a zero-initialized buffer from scratch.
    pub fn alloc(&self, size: usize) -> Option<GpuBuffer> {
        if !self.is_active() {
            return None;
        }
        let aligned = (size + 255) & !255; // 256-byte alignment for Metal
        let old = self.offset.fetch_add(aligned, Ordering::Relaxed);
        if old + size > self.capacity {
            self.offset.fetch_sub(aligned, Ordering::Relaxed);
            return None;
        }
        // Zero-initialize the region
        let ptr = unsafe { (self.buffer.contents() as *mut u8).add(old) };
        unsafe { std::ptr::write_bytes(ptr, 0, size) };
        Some(GpuBuffer::from_scratch(&self.buffer, old, size))
    }

    /// Allocate a buffer from scratch WITHOUT zero-initialization.
    /// Use only when the kernel will write every element of the output.
    pub fn alloc_uninit(&self, size: usize) -> Option<GpuBuffer> {
        if !self.is_active() {
            return None;
        }
        let aligned = (size + 255) & !255;
        let old = self.offset.fetch_add(aligned, Ordering::Relaxed);
        if old + size > self.capacity {
            self.offset.fetch_sub(aligned, Ordering::Relaxed);
            return None;
        }
        Some(GpuBuffer::from_scratch(&self.buffer, old, size))
    }

    /// Allocate a buffer from scratch with initial data.
    pub fn alloc_with_data<T: bytemuck::Pod>(&self, data: &[T]) -> Option<GpuBuffer> {
        if !self.is_active() {
            return None;
        }
        let byte_len = data.len() * std::mem::size_of::<T>();
        let aligned = (byte_len + 255) & !255;
        let old = self.offset.fetch_add(aligned, Ordering::Relaxed);
        if old + byte_len > self.capacity {
            self.offset.fetch_sub(aligned, Ordering::Relaxed);
            return None;
        }
        Some(GpuBuffer::from_scratch_with_data(&self.buffer, old, data))
    }

    /// Reset the allocator. Call only after GPU synchronize().
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }
}
