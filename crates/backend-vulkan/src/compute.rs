use crate::buffer::VulkanBuffer;
use crate::device::VulkanDevice;
use crate::shader::ShaderCompiler;
use ash::vk;
use openmemgpu_core::error::CoreError;
use std::sync::Arc;

pub struct ComputePipeline {
    device: Arc<VulkanDevice>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub shader_module: vk::ShaderModule,
    pub workgroup_size: u32,
}

impl ComputePipeline {
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, CoreError> {
        let workgroup_size = device.details.subgroup_size;
        let spv_words = ShaderCompiler::compile_vector_multiply(workgroup_size)?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);

        let shader_module = unsafe {
            device
                .raw
                .create_shader_module(&shader_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("vkCreateShaderModule failed: {:?}", e))
                })?
        };

        // Descriptor set layout with 3 storage buffers: binding 0 (read A), binding 1 (read B), binding 2 (read_write C)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            match device.raw.create_descriptor_set_layout(&dsl_info, None) {
                Ok(dsl) => dsl,
                Err(e) => {
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_descriptor_set_layout failed: {:?}",
                        e
                    )));
                }
            }
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        let pipeline_layout = unsafe {
            match device
                .raw
                .create_pipeline_layout(&pipeline_layout_info, None)
            {
                Ok(pl) => pl,
                Err(e) => {
                    device
                        .raw
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_pipeline_layout failed: {:?}",
                        e
                    )));
                }
            }
        };

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            match device.raw.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            ) {
                Ok(pipelines) => pipelines[0],
                Err((_, e)) => {
                    device.raw.destroy_pipeline_layout(pipeline_layout, None);
                    device
                        .raw
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_compute_pipelines failed: {:?}",
                        e
                    )));
                }
            }
        };

        Ok(Self {
            device,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            shader_module,
            workgroup_size,
        })
    }

    /// Dispatch vector multiplication: C[i] = A[i] * B[i]
    pub fn dispatch(
        &self,
        buf_a: &VulkanBuffer,
        buf_b: &VulkanBuffer,
        buf_c: &VulkanBuffer,
        element_count: u32,
    ) -> Result<f64, CoreError> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let desc_pool = unsafe {
            self.device
                .raw
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("create_descriptor_pool failed: {:?}", e))
                })?
        };

        let set_layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);

        let desc_sets = unsafe {
            match self.device.raw.allocate_descriptor_sets(&alloc_info) {
                Ok(sets) => sets,
                Err(e) => {
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_descriptor_sets failed: {:?}",
                        e
                    )));
                }
            }
        };
        let desc_set = desc_sets[0];

        // Update descriptor set bindings
        let buffer_info_a = [vk::DescriptorBufferInfo::default()
            .buffer(buf_a.buffer)
            .offset(0)
            .range(buf_a.size_bytes)];
        let buffer_info_b = [vk::DescriptorBufferInfo::default()
            .buffer(buf_b.buffer)
            .offset(0)
            .range(buf_b.size_bytes)];
        let buffer_info_c = [vk::DescriptorBufferInfo::default()
            .buffer(buf_c.buffer)
            .offset(0)
            .range(buf_c.size_bytes)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info_a),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info_b),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info_c),
        ];

        unsafe {
            self.device.raw.update_descriptor_sets(&writes, &[]);
        }

        // Command pool and buffer
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let cmd_pool = unsafe {
            match self.device.raw.create_command_pool(&cmd_pool_info, None) {
                Ok(cp) => cp,
                Err(e) => {
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "create_command_pool failed: {:?}",
                        e
                    )));
                }
            }
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffer = unsafe {
            match self.device.raw.allocate_command_buffers(&cmd_alloc_info) {
                Ok(cbs) => cbs[0],
                Err(e) => {
                    self.device.raw.destroy_command_pool(cmd_pool, None);
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_command_buffers failed: {:?}",
                        e
                    )));
                }
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let workgroups = element_count.div_ceil(self.workgroup_size);

        unsafe {
            let _ = self
                .device
                .raw
                .begin_command_buffer(cmd_buffer, &begin_info);
            self.device.raw.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.raw.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            self.device.raw.cmd_dispatch(cmd_buffer, workgroups, 1, 1);
            let _ = self.device.raw.end_command_buffer(cmd_buffer);
        }

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            match self.device.raw.create_fence(&fence_info, None) {
                Ok(f) => f,
                Err(e) => {
                    self.device.raw.destroy_command_pool(cmd_pool, None);
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "create_fence failed: {:?}",
                        e
                    )));
                }
            }
        };

        let submit_cmd_buffers = [cmd_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&submit_cmd_buffers)];

        let timer_start = std::time::Instant::now();
        unsafe {
            self.device
                .raw
                .queue_submit(self.device.compute_queue, &submit_info, fence)
                .map_err(|e| CoreError::DriverError(format!("queue_submit failed: {:?}", e)))?;

            // Wait for completion with timeout of 10 seconds
            self.device
                .raw
                .wait_for_fences(&[fence], true, 10_000_000_000)
                .map_err(|e| CoreError::SyncError(format!("wait_for_fences timed out: {:?}", e)))?;
        }
        let elapsed_micros = timer_start.elapsed().as_secs_f64() * 1_000_000.0;

        // Cleanup resources
        unsafe {
            self.device.raw.destroy_fence(fence, None);
            self.device.raw.destroy_command_pool(cmd_pool, None);
            self.device.raw.destroy_descriptor_pool(desc_pool, None);
        }

        Ok(elapsed_micros)
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_pipeline(self.pipeline, None);
            self.device
                .raw
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .raw
                .destroy_shader_module(self.shader_module, None);
        }
    }
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

pub struct ScaleBiasPipeline {
    device: Arc<VulkanDevice>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub shader_module: vk::ShaderModule,
    pub workgroup_size: u32,
}

unsafe impl Send for ScaleBiasPipeline {}
unsafe impl Sync for ScaleBiasPipeline {}

impl ScaleBiasPipeline {
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, CoreError> {
        let workgroup_size = device.details.subgroup_size;
        let spv_words = ShaderCompiler::compile_vector_scale_bias(workgroup_size)?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);

        let shader_module = unsafe {
            device
                .raw
                .create_shader_module(&shader_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("vkCreateShaderModule failed: {:?}", e))
                })?
        };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            match device.raw.create_descriptor_set_layout(&dsl_info, None) {
                Ok(dsl) => dsl,
                Err(e) => {
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_descriptor_set_layout failed: {:?}",
                        e
                    )));
                }
            }
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        let pipeline_layout = unsafe {
            match device
                .raw
                .create_pipeline_layout(&pipeline_layout_info, None)
            {
                Ok(pl) => pl,
                Err(e) => {
                    device
                        .raw
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_pipeline_layout failed: {:?}",
                        e
                    )));
                }
            }
        };

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            match device.raw.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            ) {
                Ok(pipelines) => pipelines[0],
                Err((_, e)) => {
                    device.raw.destroy_pipeline_layout(pipeline_layout, None);
                    device
                        .raw
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_compute_pipelines failed: {:?}",
                        e
                    )));
                }
            }
        };

        Ok(Self {
            device,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            shader_module,
            workgroup_size,
        })
    }

    /// Dispatch scale-bias: Out[i] = In[i] * 1.5 + 0.5
    pub fn dispatch(
        &self,
        in_buf: &VulkanBuffer,
        out_buf: &VulkanBuffer,
        element_count: u32,
    ) -> Result<f64, CoreError> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let desc_pool = unsafe {
            self.device
                .raw
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("create_descriptor_pool failed: {:?}", e))
                })?
        };

        let set_layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);

        let desc_sets = unsafe {
            match self.device.raw.allocate_descriptor_sets(&alloc_info) {
                Ok(sets) => sets,
                Err(e) => {
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_descriptor_sets failed: {:?}",
                        e
                    )));
                }
            }
        };
        let desc_set = desc_sets[0];

        let buffer_info_in = [vk::DescriptorBufferInfo::default()
            .buffer(in_buf.buffer)
            .offset(0)
            .range(in_buf.size_bytes)];
        let buffer_info_out = [vk::DescriptorBufferInfo::default()
            .buffer(out_buf.buffer)
            .offset(0)
            .range(out_buf.size_bytes)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info_in),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info_out),
        ];

        unsafe {
            self.device.raw.update_descriptor_sets(&writes, &[]);
        }

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let cmd_pool = unsafe {
            match self.device.raw.create_command_pool(&cmd_pool_info, None) {
                Ok(cp) => cp,
                Err(e) => {
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "create_command_pool failed: {:?}",
                        e
                    )));
                }
            }
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffer = unsafe {
            match self.device.raw.allocate_command_buffers(&cmd_alloc_info) {
                Ok(cbs) => cbs[0],
                Err(e) => {
                    self.device.raw.destroy_command_pool(cmd_pool, None);
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_command_buffers failed: {:?}",
                        e
                    )));
                }
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let workgroups = element_count.div_ceil(self.workgroup_size);

        unsafe {
            let _ = self
                .device
                .raw
                .begin_command_buffer(cmd_buffer, &begin_info);
            self.device.raw.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.raw.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            self.device.raw.cmd_dispatch(cmd_buffer, workgroups, 1, 1);
            let _ = self.device.raw.end_command_buffer(cmd_buffer);
        }

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            match self.device.raw.create_fence(&fence_info, None) {
                Ok(f) => f,
                Err(e) => {
                    self.device.raw.destroy_command_pool(cmd_pool, None);
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "create_fence failed: {:?}",
                        e
                    )));
                }
            }
        };

        let submit_cmd_buffers = [cmd_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&submit_cmd_buffers)];

        let timer_start = std::time::Instant::now();
        unsafe {
            self.device
                .raw
                .queue_submit(self.device.compute_queue, &submit_info, fence)
                .map_err(|e| CoreError::DriverError(format!("queue_submit failed: {:?}", e)))?;

            self.device
                .raw
                .wait_for_fences(&[fence], true, 10_000_000_000)
                .map_err(|e| CoreError::SyncError(format!("wait_for_fences timed out: {:?}", e)))?;
        }
        let elapsed_micros = timer_start.elapsed().as_secs_f64() * 1_000_000.0;

        unsafe {
            self.device.raw.destroy_fence(fence, None);
            self.device.raw.destroy_command_pool(cmd_pool, None);
            self.device.raw.destroy_descriptor_pool(desc_pool, None);
        }

        Ok(elapsed_micros)
    }
}

impl Drop for ScaleBiasPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_pipeline(self.pipeline, None);
            self.device
                .raw
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .raw
                .destroy_shader_module(self.shader_module, None);
        }
    }
}

pub struct TwoStageComputePipeline {
    pub stage1_mul: ComputePipeline,
    pub stage2_scale_bias: ScaleBiasPipeline,
}

unsafe impl Send for TwoStageComputePipeline {}
unsafe impl Sync for TwoStageComputePipeline {}

impl TwoStageComputePipeline {
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, CoreError> {
        let stage1_mul = ComputePipeline::new(device.clone())?;
        let stage2_scale_bias = ScaleBiasPipeline::new(device)?;
        Ok(Self {
            stage1_mul,
            stage2_scale_bias,
        })
    }

    /// Dispatch Stage 1 (A * B -> Z) followed by Stage 2 (Z * 1.5 + 0.5 -> Out)
    pub fn dispatch(
        &self,
        buf_a: &VulkanBuffer,
        buf_b: &VulkanBuffer,
        buf_z: &VulkanBuffer,
        buf_out: &VulkanBuffer,
        element_count: u32,
    ) -> Result<(f64, f64), CoreError> {
        let t1 = self
            .stage1_mul
            .dispatch(buf_a, buf_b, buf_z, element_count)?;
        let t2 = self
            .stage2_scale_bias
            .dispatch(buf_z, buf_out, element_count)?;
        Ok((t1, t2))
    }
}

pub struct GemmPipeline {
    device: Arc<VulkanDevice>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub shader_module: vk::ShaderModule,
    pub tile_x: u32,
    pub tile_y: u32,
}

unsafe impl Send for GemmPipeline {}
unsafe impl Sync for GemmPipeline {}

impl GemmPipeline {
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, CoreError> {
        let tile_x = 16u32;
        let tile_y = 16u32;
        let spv_words = ShaderCompiler::compile_gemm(tile_x, tile_y)?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
        let shader_module = unsafe {
            device
                .raw
                .create_shader_module(&shader_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("vkCreateShaderModule failed for GEMM: {:?}", e))
                })?
        };

        // Descriptor set layout with 3 storage buffers (A, B, C) and 1 uniform buffer (params)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            match device.raw.create_descriptor_set_layout(&dsl_info, None) {
                Ok(dsl) => dsl,
                Err(e) => {
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_descriptor_set_layout for GEMM failed: {:?}",
                        e
                    )));
                }
            }
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        let pipeline_layout = unsafe {
            match device
                .raw
                .create_pipeline_layout(&pipeline_layout_info, None)
            {
                Ok(pl) => pl,
                Err(e) => {
                    device
                        .raw
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_pipeline_layout for GEMM failed: {:?}",
                        e
                    )));
                }
            }
        };

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            match device.raw.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            ) {
                Ok(pipelines) => pipelines[0],
                Err((_, e)) => {
                    device.raw.destroy_pipeline_layout(pipeline_layout, None);
                    device
                        .raw
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.raw.destroy_shader_module(shader_module, None);
                    return Err(CoreError::DriverError(format!(
                        "create_compute_pipelines for GEMM failed: {:?}",
                        e
                    )));
                }
            }
        };

        Ok(Self {
            device,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            shader_module,
            tile_x,
            tile_y,
        })
    }

    /// Dispatch GEMM: C[M x N] = A[M x K] * B[K x N]
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        buf_a: &VulkanBuffer,
        buf_b: &VulkanBuffer,
        buf_c: &VulkanBuffer,
        buf_params: &VulkanBuffer,
        m: u32,
        _k: u32,
        n: u32,
    ) -> Result<f64, CoreError> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let desc_pool = unsafe {
            self.device
                .raw
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("create_descriptor_pool failed: {:?}", e))
                })?
        };

        let set_layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);

        let desc_sets = unsafe {
            match self.device.raw.allocate_descriptor_sets(&alloc_info) {
                Ok(sets) => sets,
                Err(e) => {
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_descriptor_sets failed: {:?}",
                        e
                    )));
                }
            }
        };
        let desc_set = desc_sets[0];

        let buf_info_a = [vk::DescriptorBufferInfo::default()
            .buffer(buf_a.buffer)
            .offset(0)
            .range(buf_a.size_bytes)];
        let buf_info_b = [vk::DescriptorBufferInfo::default()
            .buffer(buf_b.buffer)
            .offset(0)
            .range(buf_b.size_bytes)];
        let buf_info_c = [vk::DescriptorBufferInfo::default()
            .buffer(buf_c.buffer)
            .offset(0)
            .range(buf_c.size_bytes)];
        let buf_info_params = [vk::DescriptorBufferInfo::default()
            .buffer(buf_params.buffer)
            .offset(0)
            .range(buf_params.size_bytes)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info_a),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info_b),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info_c),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buf_info_params),
        ];

        unsafe {
            self.device.raw.update_descriptor_sets(&writes, &[]);
        }

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let cmd_pool = unsafe {
            match self.device.raw.create_command_pool(&cmd_pool_info, None) {
                Ok(cp) => cp,
                Err(e) => {
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "create_command_pool failed: {:?}",
                        e
                    )));
                }
            }
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buf = unsafe {
            match self.device.raw.allocate_command_buffers(&cmd_alloc_info) {
                Ok(cbs) => cbs[0],
                Err(e) => {
                    self.device.raw.destroy_command_pool(cmd_pool, None);
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_command_buffers failed: {:?}",
                        e
                    )));
                }
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let groups_x = n.div_ceil(self.tile_x);
        let groups_y = m.div_ceil(self.tile_y);

        unsafe {
            let _ = self.device.raw.begin_command_buffer(cmd_buf, &begin_info);
            self.device.raw.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.raw.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );

            self.device.raw.cmd_dispatch(cmd_buf, groups_x, groups_y, 1);
            let _ = self.device.raw.end_command_buffer(cmd_buf);
        }

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            match self.device.raw.create_fence(&fence_info, None) {
                Ok(f) => f,
                Err(e) => {
                    self.device.raw.destroy_command_pool(cmd_pool, None);
                    self.device.raw.destroy_descriptor_pool(desc_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "create_fence failed: {:?}",
                        e
                    )));
                }
            }
        };

        let cmd_bufs = [cmd_buf];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];

        let t_start = std::time::Instant::now();
        unsafe {
            let submit_res =
                self.device
                    .raw
                    .queue_submit(self.device.compute_queue, &submit_info, fence);
            if let Err(e) = submit_res {
                self.device.raw.destroy_fence(fence, None);
                self.device.raw.destroy_command_pool(cmd_pool, None);
                self.device.raw.destroy_descriptor_pool(desc_pool, None);
                return Err(CoreError::DriverError(format!(
                    "queue_submit failed: {:?}",
                    e
                )));
            }

            let wait_res = self
                .device
                .raw
                .wait_for_fences(&[fence], true, 30_000_000_000); // 30 second timeout for large GEMM
            if let Err(e) = wait_res {
                self.device.raw.destroy_fence(fence, None);
                self.device.raw.destroy_command_pool(cmd_pool, None);
                self.device.raw.destroy_descriptor_pool(desc_pool, None);
                return Err(CoreError::DriverError(format!(
                    "wait_for_fences timed out or failed: {:?}",
                    e
                )));
            }
        }
        let dispatch_duration_us = t_start.elapsed().as_secs_f64() * 1.0e6;

        unsafe {
            self.device.raw.destroy_fence(fence, None);
            self.device.raw.destroy_command_pool(cmd_pool, None);
            self.device.raw.destroy_descriptor_pool(desc_pool, None);
        }

        Ok(dispatch_duration_us)
    }
}

impl Drop for GemmPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_pipeline(self.pipeline, None);
            self.device
                .raw
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .raw
                .destroy_shader_module(self.shader_module, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::VulkanContext;
    use crate::physical_device::PhysicalDeviceEnumerator;
    use openmemgpu_core::device::DeviceKind;

    #[test]
    fn test_single_gpu_vector_multiply_dgpu_and_igpu() {
        let ctx = VulkanContext::new().expect("VulkanContext creation should succeed");
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)
            .expect("Physical device enumeration should succeed");

        let element_count = 1024usize;
        let byte_size = (element_count * std::mem::size_of::<f32>()) as u64;

        // Generate deterministic input data
        let input_a: Vec<f32> = (0..element_count).map(|i| (i as f32) * 1.5).collect();
        let input_b: Vec<f32> = (0..element_count).map(|_| 2.0f32).collect();
        let expected_c: Vec<f32> = (0..element_count).map(|i| (i as f32) * 1.5 * 2.0).collect();

        let mut tested_devices = 0;

        for (pdev, details) in enumerated {
            if details.kind != openmemgpu_core::device::DeviceKind::DiscreteGpu
                && details.kind != openmemgpu_core::device::DeviceKind::IntegratedGpu
            {
                continue;
            }

            println!(
                "\n=== Executing Real Compute Workload on {} ({:?}) ===",
                details.name, details.kind
            );
            println!(
                "  PCI Bus: {:?}",
                details.pci_bus.as_ref().map(|p| &p.pci_address)
            );
            println!("  Subgroup/Workgroup size: {}", details.subgroup_size);

            let timer_total = std::time::Instant::now();

            // 1. Create Logical Device
            let device = VulkanDevice::new(&ctx.instance, pdev, details.clone())
                .expect("VulkanDevice creation should succeed");

            // 2. Create Buffers (A, B, C)
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;

            let timer_alloc = std::time::Instant::now();
            let buf_a = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)
                .expect("Buffer A allocation should succeed");
            let buf_b = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)
                .expect("Buffer B allocation should succeed");
            let buf_c = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)
                .expect("Buffer C allocation should succeed");
            let alloc_time_us = timer_alloc.elapsed().as_secs_f64() * 1_000_000.0;

            // 3. Upload Inputs
            let timer_upload = std::time::Instant::now();
            buf_a.upload(&input_a).expect("Upload A should succeed");
            buf_b.upload(&input_b).expect("Upload B should succeed");
            let upload_time_us = timer_upload.elapsed().as_secs_f64() * 1_000_000.0;

            // 4. Create Compute Pipeline
            let pipeline = ComputePipeline::new(device.clone())
                .expect("ComputePipeline creation should succeed");

            // 5. Dispatch
            let dispatch_time_us = pipeline
                .dispatch(&buf_a, &buf_b, &buf_c, element_count as u32)
                .expect("Dispatch should succeed");

            // 6. Download Output
            let mut output_c = vec![0.0f32; element_count];
            let timer_download = std::time::Instant::now();
            buf_c
                .download(&mut output_c)
                .expect("Download C should succeed");
            let download_time_us = timer_download.elapsed().as_secs_f64() * 1_000_000.0;

            let total_time_us = timer_total.elapsed().as_secs_f64() * 1_000_000.0;

            // 7. Validate Correctness against CPU Reference
            for i in 0..element_count {
                let diff = (output_c[i] - expected_c[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at index {}: got {}, expected {}",
                    i,
                    output_c[i],
                    expected_c[i]
                );
            }

            println!(
                "  [SUCCESS] All {} elements verified correct!",
                element_count
            );
            println!("  Timing Breakdown:");
            println!("    Allocation time: {:.1} us", alloc_time_us);
            println!("    Upload time:     {:.1} us", upload_time_us);
            println!("    Dispatch time:   {:.1} us", dispatch_time_us);
            println!("    Download time:   {:.1} us", download_time_us);
            println!("    Total end-to-end:{:.1} us", total_time_us);

            tested_devices += 1;
        }

        assert_eq!(
            tested_devices, 2,
            "Must execute compute workload on both dGPU and iGPU"
        );
    }

    #[test]
    fn test_two_stage_compute_pipeline_dgpu_and_igpu() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)
            .expect("Failed to enumerate devices");

        for (pdev, details) in enumerated {
            if details.kind != DeviceKind::DiscreteGpu && details.kind != DeviceKind::IntegratedGpu
            {
                continue;
            }

            println!("Testing TwoStageComputePipeline on: {}", details.name);
            let device =
                VulkanDevice::new(&ctx.instance, pdev, details).expect("Device creation failed");
            let pipeline = TwoStageComputePipeline::new(device.clone())
                .expect("TwoStageComputePipeline creation failed");

            let element_count = 1024usize;
            let byte_size = (element_count * std::mem::size_of::<f32>()) as u64;
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER;

            let buf_a = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage).unwrap();
            let buf_b = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage).unwrap();
            let buf_z = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage).unwrap();
            let buf_out = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage).unwrap();

            let input_a: Vec<f32> = (0..element_count).map(|i| i as f32 * 1.5).collect();
            let input_b: Vec<f32> = (0..element_count).map(|_| 2.0f32).collect();
            let expected: Vec<f32> = (0..element_count)
                .map(|i| (i as f32 * 1.5 * 2.0) * 1.5 + 0.5)
                .collect();

            buf_a.upload(&input_a).unwrap();
            buf_b.upload(&input_b).unwrap();

            let (t1, t2) = pipeline
                .dispatch(&buf_a, &buf_b, &buf_z, &buf_out, element_count as u32)
                .unwrap();
            assert!(t1 >= 0.0 && t2 >= 0.0);

            let mut out = vec![0.0f32; element_count];
            buf_out.download(&mut out).unwrap();

            for i in 0..element_count {
                let diff = (out[i] - expected[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at {}: got {}, expected {}",
                    i,
                    out[i],
                    expected[i]
                );
            }
            println!(
                "  [SUCCESS] TwoStageComputePipeline verified on hardware (t1={:.1}us, t2={:.1}us)",
                t1, t2
            );
        }
    }

    #[test]
    fn test_gemm_pipeline_dgpu_and_igpu() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)
            .expect("Failed to enumerate devices");

        for (pdev, details) in enumerated {
            if details.kind != DeviceKind::DiscreteGpu && details.kind != DeviceKind::IntegratedGpu
            {
                continue;
            }

            println!("Testing GemmPipeline on: {}", details.name);
            let device =
                VulkanDevice::new(&ctx.instance, pdev, details).expect("Device creation failed");
            let pipeline = GemmPipeline::new(device.clone()).expect("GemmPipeline creation failed");

            let m = 64u32;
            let k = 64u32;
            let n = 64u32;

            let size_a = (m * k) as usize;
            let size_b = (k * n) as usize;
            let size_c = (m * n) as usize;

            let bytes_a = (size_a * std::mem::size_of::<f32>()) as u64;
            let bytes_b = (size_b * std::mem::size_of::<f32>()) as u64;
            let bytes_c = (size_c * std::mem::size_of::<f32>()) as u64;
            let bytes_params = 16u64; // 4 * u32

            let usage_storage = vk::BufferUsageFlags::STORAGE_BUFFER;
            let usage_uniform = vk::BufferUsageFlags::UNIFORM_BUFFER;

            let buf_a =
                VulkanBuffer::new_host_visible(device.clone(), bytes_a, usage_storage).unwrap();
            let buf_b =
                VulkanBuffer::new_host_visible(device.clone(), bytes_b, usage_storage).unwrap();
            let buf_c =
                VulkanBuffer::new_host_visible(device.clone(), bytes_c, usage_storage).unwrap();
            let buf_params =
                VulkanBuffer::new_host_visible(device.clone(), bytes_params, usage_uniform)
                    .unwrap();

            // Deterministic inputs
            let input_a: Vec<f32> = (0..size_a).map(|i| ((i % 17) as f32) * 0.1).collect();
            let input_b: Vec<f32> = (0..size_b).map(|i| ((i % 13) as f32) * 0.1).collect();
            let params: [u32; 4] = [m, k, n, 0];

            buf_a.upload(&input_a).unwrap();
            buf_b.upload(&input_b).unwrap();
            buf_params.upload(&params).unwrap();

            let dispatch_time_us = pipeline
                .dispatch(&buf_a, &buf_b, &buf_c, &buf_params, m, k, n)
                .unwrap();
            assert!(dispatch_time_us > 0.0);

            let mut out = vec![0.0f32; size_c];
            buf_c.download(&mut out).unwrap();

            // CPU reference calculation
            for row in 0..m as usize {
                for col in 0..n as usize {
                    let mut ref_sum = 0.0f32;
                    for k_idx in 0..k as usize {
                        ref_sum += input_a[row * (k as usize) + k_idx]
                            * input_b[k_idx * (n as usize) + col];
                    }
                    let actual = out[row * (n as usize) + col];
                    let diff = (actual - ref_sum).abs();
                    assert!(
                        diff < 1e-3,
                        "GEMM mismatch at ({}, {}): got {}, expected {}, diff {}",
                        row,
                        col,
                        actual,
                        ref_sum,
                        diff
                    );
                }
            }

            println!(
                "  [SUCCESS] GemmPipeline verified on hardware ({:.1} us for {}x{}x{})",
                dispatch_time_us, m, k, n
            );
        }
    }
}
