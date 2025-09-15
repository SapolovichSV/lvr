#![allow(clippy::cargo)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]

mod shader;
mod utils;

use core::panic;
use std::ffi::{CStr, CString, c_char};

use ash::{
    Entry,
    khr::{self},
    vk::{self, AccessFlags2, Handle, PFN_vkDebugUtilsMessengerCallbackEXT, PhysicalDeviceType},
};
use glfw::{self, Action, GlfwReceiver, Key, ffi::VkInstance_T};

static SCREEN_WIDTH: u32 = 1920;
static SCREEN_HEIGHT: u32 = 1080;
static MAX_FRAMES_IN_FLIGHT: usize = 2;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
#[cfg(debug_assertions)]
const LAYERS_TO_ENABLE: &[&str] = VALIDATION_LAYERS;
#[cfg(not(debug_assertions))]
const LAYERS_TO_ENABLE: &[&str] = &[]; // В релизе слои отключены

struct VulkanInternal {
    surface: vk::SurfaceKHR,
    phys_device: vk::PhysicalDevice,
    log_device: ash::Device,
    queue_index: usize,
    queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_device: khr::swapchain::Device,
    swapchain_image_format: vk::SurfaceFormatKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_view: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    present_complete_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    draw_fence: vk::Fence,
}
impl Drop for VulkanInternal {
    fn drop(&mut self) {
        unsafe {
            // Destroy synchronization objects
            self.log_device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.log_device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.log_device.destroy_fence(self.draw_fence, None);

            // Destroy command buffers and pool
            self.log_device
                .destroy_command_pool(self.command_pool, None);

            // Destroy pipeline and layout
            self.log_device
                .destroy_pipeline(self.graphics_pipeline, None);
            self.log_device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            // Destroy image views
            for &image_view in &self.swapchain_image_view {
                self.log_device.destroy_image_view(image_view, None);
            }

            // Destroy swapchain
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);

            // Destroy device and surface
            self.log_device.destroy_device(None);
            // Note: Surface will be destroyed when the instance is destroyed
        }
    }
}
impl VulkanInternal {
    /// Creates a new VulkanInternal instance with all necessary resources
    ///
    /// # Arguments
    /// * `instance` - The Vulkan instance
    /// * `entry` - The Vulkan entry point
    /// * `window` - The GLFW window to render to
    /// * `device_extension` - The device extensions to enable
    ///
    /// # Returns
    /// * A fully initialized VulkanInternal
    fn new(
        instance: &ash::Instance,
        entry: &ash::Entry,
        window: &glfw::PWindow,
        device_extension: &Vec<&CStr>,
    ) -> Self {
        // Create the Vulkan surface for the window
        let surface = Self::create_surface(instance, window);

        // Select a physical device (GPU)
        let phys_device = Self::select_physical_device(instance, device_extension);

        // Create a logical device and get the queue index
        let (log_device, queue_index) =
            Self::create_logical_device(instance, &phys_device, device_extension.clone());

        // Get the queue handle
        let queue = unsafe { log_device.get_device_queue(queue_index as _, 0) };

        // Create the swapchain
        let (swapchain, swapchain_device, swapchain_image_format, swapchain_extent) =
            Self::create_swapchain(
                instance,
                entry,
                phys_device,
                surface,
                window,
                &log_device,
                queue_index,
            );

        // Get the swapchain images
        let swapchain_images = unsafe { swapchain_device.get_swapchain_images(swapchain).unwrap() };

        // Initialize with default values to be properly set later
        let mut internal = Self {
            surface,
            phys_device,
            log_device,
            queue_index,
            queue,
            swapchain,
            swapchain_device,
            swapchain_image_format,
            swapchain_extent,
            swapchain_images,
            swapchain_image_view: vec![],
            pipeline_layout: vk::PipelineLayout::null(),
            graphics_pipeline: vk::Pipeline::null(),
            command_pool: vk::CommandPool::default(),
            command_buffer: vk::CommandBuffer::default(),
            present_complete_semaphore: vk::Semaphore::null(),
            render_finished_semaphore: vk::Semaphore::null(),
            draw_fence: vk::Fence::null(),
        };

        // Initialize all required resources in the correct order
        internal.create_image_views();
        internal.create_graphics_pipeline();
        internal.create_command_pool(instance);
        internal.create_command_buffer();
        internal.create_sync_objects();
        internal
    }

    fn create_surface(instance: &ash::Instance, window: &glfw::PWindow) -> ash::vk::SurfaceKHR {
        // must live as long as app and dropped manually at the end
        // pick window surface

        unsafe {
            let mut surface: vk::SurfaceKHR = vk::SurfaceKHR::null();
            log::trace!("SurfaceKHR::null()");
            let instance = instance.handle().as_raw() as *mut VkInstance_T;
            log::trace!("instance.handle() ...");
            let surface_ptr: *mut vk::SurfaceKHR = &raw mut surface;
            // let mut surf = [std::ptr::from_ref(&surface).cast_mut()].as_mut_ptr();
            log::trace!("before trying create window");

            window.create_window_surface(instance, std::ptr::null(), surface_ptr.cast());
            log::trace!("after");
            surface
        }
    }

    fn select_physical_device(
        instance: &ash::Instance,
        device_extension: &Vec<&CStr>,
    ) -> vk::PhysicalDevice {
        let devices: Vec<vk::PhysicalDevice> = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Can't enumerate physical devices")
        };
        assert!(!devices.is_empty(),);
        log::debug!("devices: {devices:#?}");

        let mut main_device = vk::PhysicalDevice::default();
        for device in &devices {
            if is_device_suitable(*device, instance, device_extension) {
                log::debug!("device {device:?} is suitable");
                main_device = *device;
            }
        }
        main_device
    }
    fn create_logical_device(
        instance: &ash::Instance,
        phys_device: &vk::PhysicalDevice,
        device_extension: Vec<&CStr>,
    ) -> (ash::Device, usize) {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*phys_device) };
        let graphics_index = queue_family_properties
            .iter()
            .position(|x| x.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap();

        // create this var for sure what ptr will be valid as long as app run
        let queue_priority = 0.0f32;
        let log_device_create_info = vk::DeviceQueueCreateInfo {
            queue_family_index: graphics_index as u32,
            p_queue_priorities: std::ptr::addr_of!(queue_priority),
            queue_count: 1,
            ..Default::default()
        };
        let mut physical_device_features2: vk::PhysicalDeviceFeatures2 =
            vk::PhysicalDeviceFeatures2::default();
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features {
            dynamic_rendering: vk::TRUE,
            synchronization2: vk::TRUE,
            p_next: (&raw mut physical_device_features2).cast::<std::ffi::c_void>(),
            ..Default::default()
        };
        let mut extended_state_features_ext = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {
            extended_dynamic_state: vk::TRUE,
            p_next: (&raw mut vulkan13_features).cast::<std::ffi::c_void>(),
            ..Default::default()
        };

        let len_of_device_extensions = device_extension.len();
        // must live as long as device or SEGFAULT_CORE_DUMPED))))
        let vec_with_cstrings = CStringArray::from(device_extension);
        let device_create_info = vk::DeviceCreateInfo {
            p_next: (&raw mut extended_state_features_ext).cast::<std::ffi::c_void>(),
            queue_create_info_count: 1,
            p_queue_create_infos: &raw const log_device_create_info,
            enabled_extension_count: len_of_device_extensions as u32,
            pp_enabled_extension_names: vec_with_cstrings.as_ptr(),
            ..Default::default()
        };
        (
            unsafe {
                instance
                    .create_device(*phys_device, &device_create_info, None)
                    .expect("Failed to create vk::Device")
            },
            graphics_index,
        )
    }
    fn create_swapchain(
        instance: &ash::Instance,
        entry: &ash::Entry,
        phys_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        window: &glfw::PWindow,
        log_device: &ash::Device,
        graphics_index: usize,
    ) -> (
        vk::SwapchainKHR,
        khr::swapchain::Device,
        vk::SurfaceFormatKHR,
        vk::Extent2D,
    ) {
        let surf_instance = khr::surface::Instance::new(entry, instance);

        let present_supported: bool = unsafe {
            surf_instance
                .get_physical_device_surface_support(phys_device, graphics_index as u32, surface)
                .unwrap()
        };
        log::trace!("present supported {present_supported:?}");
        let present_index = if present_supported {
            graphics_index
        } else {
            panic!()
        };
        let surface_capabilites = unsafe {
            surf_instance
                .get_physical_device_surface_capabilities(phys_device, surface)
                .unwrap()
        };
        let available_formats = unsafe {
            surf_instance
                .get_physical_device_surface_formats(phys_device, surface)
                .unwrap()
        };
        let available_present_modes = unsafe {
            surf_instance
                .get_physical_device_surface_present_modes(phys_device, surface)
                .unwrap()
        };
        let swap_surface_format = Self::choose_swap_surface_format(&available_formats);
        let swap_present_mode = Self::choose_swap_present_mode(&available_present_modes);
        let swap_extent = Self::choose_swap_extent(&surface_capabilites, window);
        let (swapchain, swap_device) = Self::create_swap_chain(
            window,
            surface,
            swap_surface_format,
            swap_present_mode,
            swap_extent,
            graphics_index as u32,
            present_index as u32,
            instance,
            log_device,
            &surface_capabilites,
        );
        (swapchain, swap_device, swap_surface_format, swap_extent)
    }
    /// Image views are needed to access the swapchain images during rendering.
    /// This function creates one image view for each swapchain image.
    ///
    /// # Returns
    /// * A reference to the vector of created image views
    fn create_image_views(&mut self) -> &Vec<vk::ImageView> {
        // Clear any existing image views
        self.swapchain_image_view.clear();

        // Create a template for image view creation
        let image_view_create_info: vk::ImageViewCreateInfo<'_> =
            vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.swapchain_image_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

        // Create an image view for each swapchain image
        for image in &self.swapchain_images {
            // Set the image for this specific image view
            let image_view_create_info = image_view_create_info.image(*image);

            // Create the image view
            let image_view: vk::ImageView = unsafe {
                self.log_device
                    .create_image_view(&image_view_create_info, None)
                    .expect("Failed to create image view")
            };
            self.swapchain_image_view.push(image_view);
        }

        &self.swapchain_image_view
    }

    /// Chooses the optimal surface format from available formats
    fn choose_swap_surface_format(vec: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        use ash::vk::{ColorSpaceKHR, Format};
        *vec.iter()
            .find(|available| {
                available.color_space == ColorSpaceKHR::SRGB_NONLINEAR
                    && available.format == Format::B8G8R8A8_SRGB
            })
            .unwrap_or(&vec[0])
    }

    /// Chooses the optimal present mode from available modes
    fn choose_swap_present_mode(vec: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        use ash::vk::PresentModeKHR;
        *vec.iter()
            .find(|available| **available == PresentModeKHR::MAILBOX)
            .unwrap_or(&PresentModeKHR::FIFO)
    }

    /// Chooses the best swap chain extent based on window size and surface capabilities
    fn choose_swap_extent(
        capabilites: &vk::SurfaceCapabilitiesKHR,
        window: &glfw::PWindow,
    ) -> vk::Extent2D {
        if capabilites.current_extent.width == u32::MAX {
            let (width, height) = window.get_size();
            assert!(!(width < 0 || height < 0), "Invalid window dimensions");
            let (min, max) = (capabilites.min_image_extent, capabilites.max_image_extent);
            vk::Extent2D {
                width: (width as u32).clamp(min.width, max.width),
                height: (height as u32).clamp(min.height, max.height),
            }
        } else {
            capabilites.current_extent
        }
    }

    /// Creates a swapchain with the given parameters
    fn create_swap_chain(
        window: &glfw::PWindow,
        surface: vk::SurfaceKHR,
        swap_surface_format: vk::SurfaceFormatKHR,
        swap_present_mode: vk::PresentModeKHR,
        swap_extend: vk::Extent2D,
        graphics_family: u32,
        present_family: u32,
        instance: &ash::Instance,
        log_device: &ash::Device,
        surface_capabilites: &vk::SurfaceCapabilitiesKHR,
    ) -> (vk::SwapchainKHR, khr::swapchain::Device) {
        use ash::vk::{
            Bool32, CompositeAlphaFlagsKHR, ImageUsageFlags, SharingMode, SurfaceTransformFlagsKHR,
            SwapchainCreateFlagsKHR, SwapchainKHR,
        };
        let clipped: Bool32 = vk::TRUE;
        let pre_transform: SurfaceTransformFlagsKHR = surface_capabilites.current_transform;
        let mut min_image_count = 3.max(surface_capabilites.min_image_count);

        min_image_count = if surface_capabilites.max_image_count > 0
            && min_image_count > surface_capabilites.max_image_count
        {
            surface_capabilites.max_image_count
        } else {
            min_image_count
        };

        let mut swap_chain_create_info = vk::SwapchainCreateInfoKHR {
            flags: SwapchainCreateFlagsKHR::default(),
            surface,
            min_image_count,
            image_format: swap_surface_format.format,
            image_color_space: swap_surface_format.color_space,
            image_extent: swap_extend,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: SharingMode::EXCLUSIVE,
            pre_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: swap_present_mode,
            clipped,
            old_swapchain: SwapchainKHR::null(),
            ..Default::default()
        };

        // Configure queue family indices
        if graphics_family == present_family {
            swap_chain_create_info.image_sharing_mode = SharingMode::EXCLUSIVE;
            swap_chain_create_info.queue_family_index_count = 0;
            swap_chain_create_info.p_queue_family_indices = std::ptr::null();
        } else {
            swap_chain_create_info.image_sharing_mode = SharingMode::CONCURRENT;
            swap_chain_create_info.queue_family_index_count = 2;
            swap_chain_create_info.p_queue_family_indices =
                [graphics_family, present_family].as_ptr();
        }

        let device = ash::khr::swapchain::Device::new(instance, log_device);
        (
            unsafe {
                device
                    .create_swapchain(&swap_chain_create_info, None)
                    .unwrap()
            },
            device,
        )
    }

    fn create_graphics_pipeline(&mut self) {
        log::trace!("start create graphics pipeline");
        let vert_name_holder = CString::new("vertMain").unwrap();
        let shader_code = utils::read_file("shaders/out/slang.spv");
        let shader_module = shader::ShaderWrap::new(&self.log_device, &shader_code);
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader_module.handle)
            .name(&vert_name_holder);
        let frag_name_holder = CString::new("fragMain").unwrap();
        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader_module.handle)
            .name(&frag_name_holder);
        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let view_port_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_slope_factor(1.0)
            .line_width(1.0);
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let binding = [color_blend_attachment];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&binding)
            .logic_op(vk::LogicOp::COPY);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            self.log_device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };
        let binding = [self.swapchain_image_format.format];
        let mut pipeline_rendering_create_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&binding);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut pipeline_rendering_create_info)
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&view_port_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(vk::RenderPass::null());
        let graphics_pipeline = unsafe {
            self.log_device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        };
        self.pipeline_layout = pipeline_layout;
        self.graphics_pipeline = graphics_pipeline;
        log::trace!("succescfully created graphics pipeline!");
    }
    fn create_command_pool(&mut self, instance: &ash::Instance) {
        let graphics_index =
            unsafe { instance.get_physical_device_queue_family_properties(self.phys_device) }
                .iter()
                .position(|x| x.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .unwrap();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_index as u32);
        let command_pool = unsafe {
            self.log_device
                .create_command_pool(&pool_info, None)
                .unwrap()
        };
        self.command_pool = command_pool;
    }
    fn create_command_buffer(&mut self) {
        let alloc_inf = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            self.log_device
                .allocate_command_buffers(&alloc_inf)
                .unwrap()[0]
        };
        self.command_buffer = command_buffer;
    }
    fn record_command_buffer(&self, image_index: usize) {
        unsafe {
            self.log_device
                .begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();
        };
        self.transition_image_layout(
            image_index,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::NONE,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );
        let clear_color = {
            let mut tmp = vk::ClearValue::default();
            let color_value = {
                let mut tmp = vk::ClearColorValue::default();
                tmp.float32 = [0.0, 0.0, 0.0, 1.0];
                tmp
            };
            tmp.color = color_value;
            tmp
        };
        let attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain_image_view[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color);

        let render_area = {
            let rect = vk::Rect2D::default();
            let mut offset = vk::Offset2D::default();
            offset = offset.x(0).y(0);

            let extent = self.swapchain_extent;
            rect.offset(offset).extent(extent)
        };
        log::trace!("render area : {render_area:?}");
        let color_attachments = [attachment_info];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);
        unsafe {
            self.log_device
                .cmd_begin_rendering(self.command_buffer, &rendering_info);
        }
        unsafe {
            self.log_device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
        };

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.swapchain_extent);
        unsafe {
            self.log_device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
            self.log_device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        };
        unsafe { self.log_device.cmd_draw(self.command_buffer, 3, 1, 0, 0) };
        unsafe { self.log_device.cmd_end_rendering(self.command_buffer) };
        self.transition_image_layout(
            image_index,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            // not sure there must be ::None maybe should ::empty()
            AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        );
        unsafe {
            self.log_device
                .end_command_buffer(self.command_buffer)
                .unwrap();
        };
    }
    #[allow(clippy::too_many_arguments)]
    fn transition_image_layout(
        &self,
        image_index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2,
        src_stage_mask: vk::PipelineStageFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
    ) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.swapchain_images[image_index])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let binding = [barrier];
        let dependency_info = vk::DependencyInfo::default()
            .dependency_flags(vk::DependencyFlags::empty())
            .image_memory_barriers(&binding);
        unsafe {
            self.log_device
                .cmd_pipeline_barrier2(self.command_buffer, &dependency_info);
        };
    }
    fn create_sync_objects(&mut self) {
        let present_complete_semaphore = unsafe {
            self.log_device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        let render_finished_semaphore = unsafe {
            self.log_device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        let draw_fence = unsafe {
            self.log_device
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .unwrap()
        };
        self.present_complete_semaphore = present_complete_semaphore;
        self.render_finished_semaphore = render_finished_semaphore;
        self.draw_fence = draw_fence;
    }
}
struct VulkanApp {
    glfw_var: glfw::Glfw,
    window: glfw::PWindow,
    events: GlfwReceiver<(f64, glfw::WindowEvent)>,
    instance: ash::Instance,
    entry: ash::Entry,
    _holder: CStringArray,
    internal: VulkanInternal,
}
impl VulkanApp {
    fn new(width: u32, height: u32, window_name: &str, device_extension: Vec<&CStr>) -> Self {
        let (glfw_var, window, events) = Self::init_window(width, height, window_name);
        let (instance, entry, holder) = Self::init_vulkan(&glfw_var);
        Self {
            internal: VulkanInternal::new(&instance, &entry, &window, &device_extension),
            glfw_var,
            window,
            events,
            instance,
            entry,
            _holder: holder,
        }
    }

    fn init_window(
        width: u32,
        height: u32,
        window_name: &str,
    ) -> (
        glfw::Glfw,
        glfw::PWindow,
        glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
    ) {
        use glfw::fail_on_errors;
        let mut glfw = glfw::init(fail_on_errors!()).expect("Failed to init glfw init");
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

        let (mut window, events) = glfw
            .create_window(width, height, window_name, glfw::WindowMode::Windowed)
            .expect("Failed to create window");
        window.set_key_polling(true);
        (glfw, window, events)
    }
    /// Initializes Vulkan, creating the instance and loading validation layers
    ///
    /// # Arguments
    /// * `glfw_var` - Reference to the initialized GLFW instance
    ///
    /// # Returns
    /// * A tuple containing the Vulkan instance, entry point, and validation layers
    fn init_vulkan(glfw_var: &glfw::Glfw) -> (ash::Instance, ash::Entry, CStringArray) {
        // Get the Vulkan entry point
        let entry = Entry::linked();

        // Setup application info
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 4, 0),
            ..Default::default()
        };

        // Setup debug messenger
        let mut debug_create_info = Self::setup_debug_messenger(Some(callback));

        // Verify validation layers are available
        Self::check_layers(&entry);

        // Get required GLFW extensions and add debug utils if needed
        let mut glfw_extensions = Self::get_required_extensions(glfw_var);

        // Setup validation layers
        let layers = CStringArray::from(LAYERS_TO_ENABLE);

        // Create the Vulkan instance
        let instance = Self::create_instance(
            glfw_extensions,
            &app_info,
            &entry,
            &layers,
            &mut debug_create_info,
        );

        (instance, entry, layers)
    }

    /// Gets the required extensions for Vulkan from GLFW and adds debug extensions if needed
    ///
    /// # Arguments
    /// * `glfw_var` - Reference to the initialized GLFW instance
    ///
    /// # Returns
    /// * A vector of extension names as strings
    /// Sets up the debug messenger for Vulkan validation layers
    ///
    /// # Arguments
    /// * `callback` - Function pointer to the debug callback function
    ///
    /// # Returns
    /// * The debug messenger create info structure
    fn setup_debug_messenger(
        callback: PFN_vkDebugUtilsMessengerCallbackEXT,
    ) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(callback)
    }

    /// Verifies that all required validation layers are available
    ///
    /// # Arguments
    /// * `entry` - The Vulkan entry point
    ///
    /// # Panics
    /// * If any required layer is not available
    fn check_layers(entry: &ash::Entry) {
        let layer_properties = unsafe {
            (*entry)
                .enumerate_instance_layer_properties()
                .expect("Why can't check layer_properites?")
        };
        log::debug!("have layers: {layer_properties:?}");
        if !LAYERS_TO_ENABLE.iter().all(|layer| {
            log::debug!("layer : {layer:?}");
            layer_properties.iter().any(|have| {
                let new_str = have
                    .layer_name_as_c_str()
                    .expect("Failed to get Cstring in layers")
                    .to_str()
                    .unwrap();
                new_str == *layer
            })
        }) {
            panic!("Some missing layers");
        }
        log::info!("layer properties: {layer_properties:?}");
    }

    /// Creates a Vulkan instance with the specified parameters
    ///
    /// # Arguments
    /// * `glfw_extensions` - Vector of extension names as strings
    /// * `app_info` - Application info for Vulkan
    /// * `entry` - Vulkan entry point
    /// * `layers` - Validation layers to enable
    /// * `debug_create_info` - Debug messenger creation info
    ///
    /// # Returns
    /// * The created Vulkan instance
    fn create_instance(
        glfw_extensions: Vec<String>,
        app_info: &vk::ApplicationInfo<'_>,
        entry: &ash::Entry,
        layers: &CStringArray,
        debug_create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> ash::Instance {
        // Get layer names as C strings
        let enabled_layer_c_names = layers.as_ptr();

        // Convert extension names to C strings
        let len_ext_names = glfw_extensions.len();
        let extension_names = vec_strings_to_ptptr(glfw_extensions);

        // Create the instance create info
        let create_info = vk::InstanceCreateInfo {
            p_application_info: app_info,
            enabled_extension_count: len_ext_names as u32,
            pp_enabled_extension_names: extension_names,
            enabled_layer_count: LAYERS_TO_ENABLE.len() as u32,
            pp_enabled_layer_names: enabled_layer_c_names,
            ..Default::default()
        }
        .push_next(debug_create_info);

        // Create the instance
        unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create Vulkan instance")
        }
    }

    fn get_required_extensions(glfw_var: &glfw::Glfw) -> Vec<String> {
        // Get available extensions
        let extension_properties = unsafe {
            Entry::linked()
                .enumerate_instance_extension_properties(None)
                .expect("Failed to get instance extension properties")
        };
        log::info!("Vulkan having extensions: {extension_properties:#?}");

        // Get required GLFW extensions
        let mut glfw_extensions: Vec<String> = glfw_var
            .get_required_instance_extensions()
            .expect("GLFW API unavailable");

        // Add debug utils extension
        glfw_extensions.push("VK_EXT_debug_utils".into());

        // Verify all required extensions are available
        if !glfw_extensions.clone().into_iter().all(|glfw_extension| {
            log::info!("extension = {glfw_extension}");
            let ext: CString = CString::new(glfw_extension).expect("Failed to create CString");
            let cstring_vec: Vec<&CStr> = extension_properties
                .iter()
                .map(|prop| prop.extension_name_as_c_str().unwrap())
                .collect();

            cstring_vec.contains(&ext.as_ref())
        }) {
            panic!("Some required extensions are missing");
        }

        glfw_extensions
    }
    fn main_loop(&mut self) {
        while !self.window.should_close() {
            self.glfw_var.poll_events();
            for (_, event) in glfw::flush_messages(&(self.events)) {
                println!("{event:?}");
                if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                    self.window.set_should_close(true);
                }
            }
            self.draw_frame();
        }
        unsafe { self.internal.log_device.device_wait_idle().unwrap() };
    }
    fn draw_frame(&self) {
        let (image_index, result) = unsafe {
            self.internal
                .swapchain_device
                .acquire_next_image(
                    self.internal.swapchain,
                    u64::MAX,
                    self.internal.present_complete_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
        };
        self.internal.record_command_buffer(image_index as usize);
        unsafe {
            self.internal
                .log_device
                .reset_fences(&[self.internal.draw_fence])
                .unwrap();
        };

        let wait_destination_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let wait_dst_stage_mask = [wait_destination_stage_mask];
        let cmd_buffers = [self.internal.command_buffer];
        let wait_semaphores = [self.internal.present_complete_semaphore];
        let signal_semaphores = [self.internal.render_finished_semaphore];

        let submit_info = vk::SubmitInfo::default()
            .wait_dst_stage_mask(&wait_dst_stage_mask)
            .command_buffers(&cmd_buffers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.internal
                .log_device
                .queue_submit(
                    self.internal.queue,
                    &[submit_info],
                    self.internal.draw_fence,
                )
                .unwrap();
        };
        loop {
            unsafe {
                match self.internal.log_device.wait_for_fences(
                    &[self.internal.draw_fence],
                    true,
                    u64::MAX,
                ) {
                    Ok(()) => break,
                    Err(vk::Result::TIMEOUT) => (),
                    Err(e) => panic!("fail with e: {e}"),
                }
            }
        }
        let binding = [self.internal.render_finished_semaphore];
        let binding2 = [self.internal.swapchain];
        let binding3 = [image_index];
        let present_info_khr = vk::PresentInfoKHR::default()
            .wait_semaphores(&binding)
            .swapchains(&binding2)
            .image_indices(&binding3);
        let result = unsafe {
            self.internal
                .swapchain_device
                .queue_present(self.internal.queue, &present_info_khr)
                .unwrap()
        };
    }
}

extern "system" fn callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;

    let callback_data = unsafe { *p_callback_data };
    let message = unsafe { std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() };

    // Выводим сообщение в зависимости от его уровня важности
    match message_severity {
        Severity::ERROR => {
            eprintln!("[Vulkan ERROR] my print{message_type:?}: {message}");
        }
        Severity::WARNING => {
            println!("[Vulkan WARNING] {message_type:?}: {message}");
        }
        Severity::INFO => {
            println!("[Vulkan INFO] {message_type:?}: {message}");
        }
        Severity::VERBOSE => {
            println!("[Vulkan VERBOSE] {message_type:?}: {message}");
        }
        _ => (),
    }

    vk::FALSE
}

/// Application entry point
fn main() {
    // Initialize logging
    env_logger::init();
    log::info!("Initialized logger");

    // Define required device extensions
    let device_extension: Vec<&CStr> = vec![
        khr::swapchain::NAME,          // Required for presentation
        khr::spirv_1_4::NAME,          // Required for shader compilation
        khr::synchronization2::NAME,   // Required for synchronization
        khr::create_renderpass2::NAME, // Required for rendering
    ];

    // Create and initialize the Vulkan application
    let mut v_app = VulkanApp::new(
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        "Vulkan Renderer",
        device_extension.clone(),
    );

    // Run the main event loop
    v_app.main_loop();
}

// Functions moved to impl VulkanInternal
fn vec_strings_to_ptptr(v_strings: Vec<String>) -> *const *const i8 {
    let c_strings: Vec<CString> = v_strings
        .into_iter()
        .map(|elem| match CString::new(elem) {
            Ok(c_like_str) => c_like_str,
            Err(e) => {
                log::error!(
                    "can't fullfill vk::create_info because can't convert strings cause : {e}"
                );
                panic!();
            }
        })
        .collect();

    let ptrs: Vec<*const c_char> = c_strings.iter().map(|cs| cs.as_ptr()).collect();

    // Get pointer to the array of pointers
    let ptrptr = ptrs.as_ptr();

    // Prevent vectors from being dropped
    std::mem::forget(c_strings);
    std::mem::forget(ptrs);

    ptrptr
}
struct CStringArray {
    _holder: Vec<CString>,
    ptrs: Vec<*const i8>,
}
impl CStringArray {
    fn as_ptr(&self) -> *const *const i8 {
        self.ptrs.as_ptr()
    }
}
impl From<Vec<&CStr>> for CStringArray {
    fn from(value: Vec<&CStr>) -> Self {
        log::trace!("From<Vec<&CStr>> -> CStringArray");
        let _holder: Vec<CString> = value.clone().iter().map(|x| CString::from(*x)).collect();
        let ptrs: Vec<*const i8> = _holder.iter().map(|x| x.as_ptr()).collect();
        // dont sure will it segfault or no so just clone, will see
        Self { _holder, ptrs }
    }
}
impl From<&[&str]> for CStringArray {
    fn from(value: &[&str]) -> Self {
        let holder: Vec<CString> = value.iter().map(|s| CString::new(*s).unwrap()).collect();

        let mut ptrs: Vec<*const i8> = holder.iter().map(|cs| cs.as_ptr()).collect();

        ptrs.push(std::ptr::null());
        CStringArray {
            _holder: holder,
            ptrs,
        }
    }
}
fn is_device_suitable(
    device: vk::PhysicalDevice,
    instance: &ash::Instance,
    must_have_extension: &Vec<&CStr>,
) -> bool {
    let properties = unsafe { (*instance).get_physical_device_properties(device) };
    log::trace!("device: {device:?} properties: {properties:?}");
    let features = unsafe { (*instance).get_physical_device_features(device) };
    log::trace!("device : {device:?} features: {features:?}");
    let queue_families = unsafe { (*instance).get_physical_device_queue_family_properties(device) };

    let extension_properties = unsafe {
        (*instance)
            .enumerate_device_extension_properties(device)
            .expect("Can't get device extension properties")
    };
    let extensions: Vec<&CStr> = extension_properties
        .iter()
        .map(|ext| ext.extension_name_as_c_str().unwrap())
        .collect();
    let has_graphic_queue = queue_families
        .iter()
        .any(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    let has_needed_extension = must_have_extension
        .iter()
        .all(|need| extensions.contains(need));
    let is_dedicated_gpu = properties.device_type == PhysicalDeviceType::DISCRETE_GPU;
    let is_good = has_needed_extension && has_graphic_queue && is_dedicated_gpu;
    log::info!(
        "device: {:?} is {is_good:?}",
        properties.device_name_as_c_str().unwrap()
    );
    is_good
}
