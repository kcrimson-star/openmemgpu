use crate::device::VulkanDevice;
use ash::vk;
use openmemgpu_core::error::CoreError;
use std::sync::Arc;

pub struct VulkanBuffer {
    pub device: Arc<VulkanDevice>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size_bytes: u64,
    pub requested_size_bytes: u64,
    pub actual_allocated_bytes: u64,
    pub memory_type_index: u32,
    pub is_device_local: bool,
    pub is_host_visible: bool,
    pub is_host_coherent: bool,
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

impl VulkanBuffer {
    /// Create a host-visible buffer (typically in GTT or SAM BAR), accessible via CPU map
    pub fn new_host_visible(
        device: Arc<VulkanDevice>,
        size_bytes: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, CoreError> {
        let full_usage =
            usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(full_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .raw
                .create_buffer(&buffer_info, None)
                .map_err(|e| CoreError::DriverError(format!("vkCreateBuffer failed: {:?}", e)))?
        };

        let mem_reqs = unsafe { device.raw.get_buffer_memory_requirements(buffer) };

        // Prefer HOST_VISIBLE | HOST_COHERENT, fallback to HOST_VISIBLE
        let mem_type_index = device
            .details
            .find_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .or_else(|| {
                device.details.find_memory_type(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
            })
            .ok_or_else(|| {
                unsafe { device.raw.destroy_buffer(buffer, None) };
                CoreError::OutOfMemory {
                    domain_id: "HostVisibleMemory".into(),
                    requested_bytes: size_bytes,
                    available_bytes: 0,
                }
            })?;

        let mem_type_info = &device.details.types[mem_type_index as usize];
        let is_device_local = mem_type_info.is_device_local;
        let is_host_coherent = mem_type_info.is_host_coherent;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            match device.raw.allocate_memory(&alloc_info, None) {
                Ok(mem) => mem,
                Err(_e) => {
                    device.raw.destroy_buffer(buffer, None);
                    return Err(CoreError::OutOfMemory {
                        domain_id: format!("MemoryType_{}", mem_type_index),
                        requested_bytes: mem_reqs.size,
                        available_bytes: 0,
                    });
                }
            }
        };

        unsafe {
            if let Err(e) = device.raw.bind_buffer_memory(buffer, memory, 0) {
                device.raw.free_memory(memory, None);
                device.raw.destroy_buffer(buffer, None);
                return Err(CoreError::DriverError(format!(
                    "vkBindBufferMemory failed: {:?}",
                    e
                )));
            }
        }

        Ok(Self {
            device,
            buffer,
            memory,
            size_bytes,
            requested_size_bytes: size_bytes,
            actual_allocated_bytes: mem_reqs.size,
            memory_type_index: mem_type_index,
            is_device_local,
            is_host_visible: true,
            is_host_coherent,
        })
    }

    /// Allocate buffer residing in high-speed GPU device-local memory (e.g. dedicated GDDR6 VRAM on dGPU)
    pub fn new_device_local(
        device: Arc<VulkanDevice>,
        size_bytes: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, CoreError> {
        let full_usage =
            usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(full_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .raw
                .create_buffer(&buffer_info, None)
                .map_err(|e| CoreError::DriverError(format!("vkCreateBuffer failed: {:?}", e)))?
        };

        let mem_reqs = unsafe { device.raw.get_buffer_memory_requirements(buffer) };

        // For discrete GPUs: prioritize pure DEVICE_LOCAL (non host-visible) to place in dedicated GDDR6 VRAM
        let mem_type_index = device
            .details
            .types
            .iter()
            .enumerate()
            .find(|(idx, mt)| {
                (mem_reqs.memory_type_bits & (1 << idx)) != 0
                    && mt.is_device_local
                    && !mt.is_host_visible
            })
            .map(|(idx, _)| idx as u32)
            .or_else(|| {
                // Fallback to any DEVICE_LOCAL (e.g. APU carveout or SAM BAR)
                device.details.find_memory_type(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
            })
            .ok_or_else(|| {
                unsafe { device.raw.destroy_buffer(buffer, None) };
                CoreError::OutOfMemory {
                    domain_id: "DeviceLocalMemory".into(),
                    requested_bytes: size_bytes,
                    available_bytes: 0,
                }
            })?;

        let mem_type_info = &device.details.types[mem_type_index as usize];
        let is_host_visible = mem_type_info.is_host_visible;
        let is_host_coherent = mem_type_info.is_host_coherent;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            match device.raw.allocate_memory(&alloc_info, None) {
                Ok(mem) => mem,
                Err(_e) => {
                    device.raw.destroy_buffer(buffer, None);
                    return Err(CoreError::OutOfMemory {
                        domain_id: format!("MemoryType_{}", mem_type_index),
                        requested_bytes: mem_reqs.size,
                        available_bytes: 0,
                    });
                }
            }
        };

        unsafe {
            if let Err(e) = device.raw.bind_buffer_memory(buffer, memory, 0) {
                device.raw.free_memory(memory, None);
                device.raw.destroy_buffer(buffer, None);
                return Err(CoreError::DriverError(format!(
                    "vkBindBufferMemory failed: {:?}",
                    e
                )));
            }
        }

        Ok(Self {
            device,
            buffer,
            memory,
            size_bytes,
            requested_size_bytes: size_bytes,
            actual_allocated_bytes: mem_reqs.size,
            memory_type_index: mem_type_index,
            is_device_local: true,
            is_host_visible,
            is_host_coherent,
        })
    }

    /// Upload data slice from host CPU memory to GPU buffer directly (only supported if is_host_visible)
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<(), CoreError> {
        let byte_len = std::mem::size_of_val(data) as u64;
        if byte_len > self.size_bytes {
            return Err(CoreError::OutOfMemory {
                domain_id: "BufferUpload".into(),
                requested_bytes: byte_len,
                available_bytes: self.size_bytes,
            });
        }

        if !self.is_host_visible {
            return Err(CoreError::DriverError(
                "Cannot call direct upload on non-host-visible buffer; use upload_staged".into(),
            ));
        }

        unsafe {
            let mapped_ptr = self
                .device
                .raw
                .map_memory(self.memory, 0, byte_len, vk::MemoryMapFlags::empty())
                .map_err(|e| CoreError::DriverError(format!("vkMapMemory failed: {:?}", e)))?;

            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped_ptr as *mut u8,
                byte_len as usize,
            );

            if !self.is_host_coherent {
                let range = vk::MappedMemoryRange::default()
                    .memory(self.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                let _ = self.device.raw.flush_mapped_memory_ranges(&[range]);
            }

            self.device.raw.unmap_memory(self.memory);
        }

        Ok(())
    }

    /// Download data slice from GPU buffer to host CPU memory directly (only supported if is_host_visible)
    pub fn download<T: Copy>(&self, out: &mut [T]) -> Result<(), CoreError> {
        let byte_len = std::mem::size_of_val(out) as u64;
        if byte_len > self.size_bytes {
            return Err(CoreError::OutOfMemory {
                domain_id: "BufferDownload".into(),
                requested_bytes: byte_len,
                available_bytes: self.size_bytes,
            });
        }

        if !self.is_host_visible {
            return Err(CoreError::DriverError(
                "Cannot call direct download on non-host-visible buffer; use download_staged"
                    .into(),
            ));
        }

        unsafe {
            let mapped_ptr = self
                .device
                .raw
                .map_memory(self.memory, 0, byte_len, vk::MemoryMapFlags::empty())
                .map_err(|e| CoreError::DriverError(format!("vkMapMemory failed: {:?}", e)))?;

            if !self.is_host_coherent {
                let range = vk::MappedMemoryRange::default()
                    .memory(self.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                let _ = self.device.raw.invalidate_mapped_memory_ranges(&[range]);
            }

            std::ptr::copy_nonoverlapping(
                mapped_ptr as *const u8,
                out.as_mut_ptr() as *mut u8,
                byte_len as usize,
            );

            self.device.raw.unmap_memory(self.memory);
        }

        Ok(())
    }

    /// Upload data to buffer, automatically allocating a host staging buffer if device-local.
    /// Returns transfer elapsed microseconds.
    pub fn upload_staged<T: Copy>(&self, data: &[T]) -> Result<f64, CoreError> {
        let byte_len = std::mem::size_of_val(data) as u64;
        if byte_len > self.size_bytes {
            return Err(CoreError::OutOfMemory {
                domain_id: "BufferUploadStaged".into(),
                requested_bytes: byte_len,
                available_bytes: self.size_bytes,
            });
        }

        if self.is_host_visible {
            let timer = std::time::Instant::now();
            self.upload(data)?;
            return Ok(timer.elapsed().as_secs_f64() * 1_000_000.0);
        }

        // Staging buffer in host-visible memory
        let staging = Self::new_host_visible(
            self.device.clone(),
            byte_len,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        staging.upload(data)?;

        // Copy from host-visible staging buffer to pure device-local buffer
        let elapsed = self
            .device
            .copy_buffer(staging.buffer, self.buffer, byte_len)?;
        Ok(elapsed)
    }

    /// Download data from buffer to host CPU, automatically using a staging buffer if device-local.
    /// Returns transfer elapsed microseconds.
    pub fn download_staged<T: Copy>(&self, out: &mut [T]) -> Result<f64, CoreError> {
        let byte_len = std::mem::size_of_val(out) as u64;
        if byte_len > self.size_bytes {
            return Err(CoreError::OutOfMemory {
                domain_id: "BufferDownloadStaged".into(),
                requested_bytes: byte_len,
                available_bytes: self.size_bytes,
            });
        }

        if self.is_host_visible {
            let timer = std::time::Instant::now();
            self.download(out)?;
            return Ok(timer.elapsed().as_secs_f64() * 1_000_000.0);
        }

        // Staging buffer in host-visible memory
        let staging = Self::new_host_visible(
            self.device.clone(),
            byte_len,
            vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        // Copy from pure device-local buffer to host-visible staging buffer
        let elapsed = self
            .device
            .copy_buffer(self.buffer, staging.buffer, byte_len)?;

        staging.download(out)?;
        Ok(elapsed)
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::VulkanContext;
    use crate::physical_device::PhysicalDeviceEnumerator;

    #[test]
    fn test_device_local_buffer_staging() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let devices = PhysicalDeviceEnumerator::enumerate(&ctx.instance)
            .expect("Failed to enumerate devices");

        for (pdev, details) in devices {
            println!("Testing device-local staging on: {}", details.name);
            let dev = VulkanDevice::new(&ctx.instance, pdev, details.clone())
                .expect("Failed to create VulkanDevice");

            let count = 1024usize;
            let byte_len = (count * std::mem::size_of::<f32>()) as u64;
            let src_data: Vec<f32> = (0..count).map(|i| i as f32 * 1.5).collect();
            let mut dst_data = vec![0.0f32; count];

            let buffer = VulkanBuffer::new_device_local(
                dev.clone(),
                byte_len,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("Failed to create device-local buffer");

            assert!(buffer.is_device_local);
            assert!(buffer.actual_allocated_bytes >= buffer.requested_size_bytes);

            // Upload via staging
            let upload_us = buffer
                .upload_staged(&src_data)
                .expect("upload_staged failed");
            assert!(upload_us >= 0.0);

            // Download via staging
            let download_us = buffer
                .download_staged(&mut dst_data)
                .expect("download_staged failed");
            assert!(download_us >= 0.0);

            // Numerical verification
            for i in 0..count {
                assert_eq!(
                    src_data[i], dst_data[i],
                    "Mismatch at index {}: expected {}, got {}",
                    i, src_data[i], dst_data[i]
                );
            }
            println!(
                "Device-local staging verified for {}: upload={:.2}us, download={:.2}us, allocated={} B",
                details.name, upload_us, download_us, buffer.actual_allocated_bytes
            );
        }
    }
}
