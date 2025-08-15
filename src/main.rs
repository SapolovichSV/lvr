#![warn(clippy::cargo)]
#![warn(clippy::pedantic)]

use core::panic;
use std::{
    ffi::{CStr, CString, c_char},
    mem,
};

use ash::{
    Entry,
    khr::{self},
    vk::{self, Handle, PFN_vkDebugUtilsMessengerCallbackEXT, PhysicalDeviceType},
};
use glfw::{self, Action, GlfwReceiver, Key, ffi::VkInstance_T};

static SCREEN_WIDTH: u32 = 1920;
static SCREEN_HEIGHT: u32 = 1080;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
#[cfg(debug_assertions)]
const LAYERS_TO_ENABLE: &[&str] = VALIDATION_LAYERS;
#[cfg(not(debug_assertions))]
const layers_to_enable: &[&str] = &[]; // В релизе слои отключены
struct VulkanInternal {
    surface: vk::SurfaceKHR,
    phys_device: vk::PhysicalDevice,
    log_device: ash::Device,
    swapchain: vk::SwapchainKHR,
}
impl VulkanInternal {
    fn new(
        instance: &ash::Instance,
        entry: &ash::Entry,
        window: &glfw::PWindow,
        device_extension: &Vec<&CStr>,
    ) -> Self {
        let surface = Self::create_surface(instance, window);
        let phys_device = Self::select_physical_device(instance, device_extension);
        let log_device =
            Self::create_logical_device(instance, &phys_device, device_extension.clone());
        let swapchain = Self::create_swapchain(
            instance,
            entry,
            &phys_device,
            device_extension.clone(),
            &surface,
            window,
        );
        Self {
            surface,
            phys_device,
            log_device,
            swapchain,
        }
    }

    fn create_surface(instance: &ash::Instance, window: &glfw::PWindow) -> ash::vk::SurfaceKHR {
        // must live as long as app and dropped manually at the end
        // pick window surface
        let surface = unsafe {
            let mut surface: vk::SurfaceKHR = vk::SurfaceKHR::null();
            log::trace!("SurfaceKHR::null()");
            let instance = instance.handle().as_raw() as *mut VkInstance_T;
            log::trace!("instance.handle() ...");
            let surface_ptr: *mut vk::SurfaceKHR = &mut surface;
            // let mut surf = [std::ptr::from_ref(&surface).cast_mut()].as_mut_ptr();
            log::trace!("before trying create window");

            window.create_window_surface(instance, std::ptr::null(), surface_ptr as *mut _);
            log::trace!("after");
            surface
        };
        surface
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

        let mut main_device = Default::default();
        devices.iter().for_each(|device| {
            if is_device_suitable(device, instance, device_extension) {
                log::debug!("device {device:?} is suitable");
                main_device = *device;
            }
        });
        main_device
    }
    fn create_logical_device(
        instance: &ash::Instance,
        phys_device: &vk::PhysicalDevice,
        device_extension: Vec<&CStr>,
    ) -> ash::Device {
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
            p_next: &mut physical_device_features2 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        let mut extended_state_features_ext = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {
            extended_dynamic_state: vk::TRUE,
            p_next: &mut vulkan13_features as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };

        let len_of_device_extensions = device_extension.len();
        // must live as long as device or SEGFAULT_CORE_DUMPED))))
        let vec_with_cstrings = CStringArray::from(device_extension);
        let device_create_info = vk::DeviceCreateInfo {
            p_next: &mut extended_state_features_ext as *mut _ as *mut std::ffi::c_void,
            queue_create_info_count: 1,
            p_queue_create_infos: &log_device_create_info,
            enabled_extension_count: len_of_device_extensions as u32,
            pp_enabled_extension_names: vec_with_cstrings.as_ptr(),
            ..Default::default()
        };
        unsafe {
            instance
                .create_device(*phys_device, &device_create_info, None)
                .expect("Failed to create vk::Device")
        }
    }
    fn create_swapchain(
        instance: &ash::Instance,
        entry: &ash::Entry,
        phys_device: &vk::PhysicalDevice,
        device_extension: Vec<&CStr>,
        surface: &vk::SurfaceKHR,
        window: &glfw::PWindow,
    ) -> vk::SwapchainKHR {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*phys_device) };
        let graphics_index = queue_family_properties
            .iter()
            .position(|x| x.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap();

        // create this var for sure what ptr will be valid as long as app run
        let queue_priority = 0.0f32;
        let log_device = vk::DeviceQueueCreateInfo {
            queue_family_index: graphics_index as u32,
            p_queue_priorities: std::ptr::addr_of!(queue_priority),
            queue_count: 1,
            ..Default::default()
        };
        let device = create_logical_device(device_extension, instance, log_device, *phys_device);

        let graphics_queue = unsafe { device.get_device_queue(graphics_index as u32, 0) };

        let surf_instance = khr::surface::Instance::new(entry, instance);

        let present_supported: bool = unsafe {
            surf_instance
                .get_physical_device_surface_support(*phys_device, graphics_index as u32, *surface)
                .unwrap()
        };
        log::trace!("present supported {present_supported:?}");
        let present_index = if present_supported {
            graphics_index
        } else {
            panic!()
        };
        let present_queue = unsafe { device.get_device_queue(present_index as u32, 0) };
        let surface_capabilites = unsafe {
            surf_instance
                .get_physical_device_surface_capabilities(*phys_device, *surface)
                .unwrap()
        };
        let available_formats = unsafe {
            surf_instance
                .get_physical_device_surface_formats(*phys_device, *surface)
                .unwrap()
        };
        let available_present_modes = unsafe {
            surf_instance
                .get_physical_device_surface_present_modes(*phys_device, *surface)
                .unwrap()
        };
        create_swap_chain(
            window,
            surface_capabilites,
            available_present_modes,
            available_formats,
            *surface,
            graphics_index as u32,
            present_index as u32,
            instance,
            &device,
        )
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
    fn init_vulkan(glfw_var: &glfw::Glfw) -> (ash::Instance, ash::Entry, CStringArray) {
        // createSwapChain();
        let entry = Entry::linked();
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 4, 0),
            ..Default::default()
        };
        let mut debug_create_info = setup_debug_messenger(Some(callback));
        check_layers(&entry);

        let extension_properties: Vec<vk::ExtensionProperties> = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to get instance extension properties")
        };
        log::info!("Vulkan having extensions: {extension_properties:#?}");

        let glfw_extensions: Vec<String> = glfw_var
            .get_required_instance_extensions()
            .expect("glfw api unavaible");

        if !glfw_extensions.clone().into_iter().all(|glfw_extension| {
            log::info!("extension = {glfw_extension}");
            let ext: CString = CString::new(glfw_extension).expect("Ahhh failed to get CString");
            let cstring_vec: Vec<&CStr> = extension_properties
                .iter()
                .map(|prop| prop.extension_name_as_c_str().unwrap())
                .collect();

            cstring_vec.contains(&ext.as_ref())
        }) {
            panic!("Some extensions is missing");
        }
        let layers = CStringArray::from(LAYERS_TO_ENABLE);
        let instance = create_instance(
            glfw_extensions,
            &app_info,
            &entry,
            &layers,
            &mut debug_create_info,
        );
        (instance, entry, layers)
    }
}

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
fn create_instance(
    glfw_extensions: Vec<String>,
    app_info: &vk::ApplicationInfo<'_>,
    entry: &ash::Entry,
    layers: &CStringArray,
    debug_create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT,
) -> ash::Instance {
    let enabled_layer_c_names = layers.as_ptr();
    let len_ext_names = glfw_extensions.len();
    let extension_names = vec_strings_to_ptptr(glfw_extensions);
    let create_info = vk::InstanceCreateInfo {
        p_application_info: app_info,
        enabled_extension_count: len_ext_names as u32,
        pp_enabled_extension_names: extension_names,
        enabled_layer_count: LAYERS_TO_ENABLE.len() as u32,
        pp_enabled_layer_names: enabled_layer_c_names,
        ..Default::default()
    }
    .push_next(debug_create_info);
    unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("can't create instance")
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

    let mut main_device = Default::default();
    devices.iter().for_each(|device| {
        if is_device_suitable(device, instance, device_extension) {
            log::debug!("device {device:?} is suitable");
            main_device = *device;
        }
    });
    main_device
}
fn create_logical_device(
    device_extension: Vec<&CStr>,
    instance: &ash::Instance,
    log_device_create_info: vk::DeviceQueueCreateInfo<'_>,
    main_device: vk::PhysicalDevice,
) -> ash::Device {
    let mut physical_device_features2: vk::PhysicalDeviceFeatures2 =
        vk::PhysicalDeviceFeatures2::default();
    let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features {
        dynamic_rendering: vk::TRUE,
        p_next: &mut physical_device_features2 as *mut _ as *mut std::ffi::c_void,
        ..Default::default()
    };
    let mut extended_state_features_ext = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {
        extended_dynamic_state: vk::TRUE,
        p_next: &mut vulkan13_features as *mut _ as *mut std::ffi::c_void,
        ..Default::default()
    };

    let len_of_device_extensions = device_extension.len();
    // must live as long as device or SEGFAULT_CORE_DUMPED))))
    let vec_with_cstrings = CStringArray::from(device_extension);
    let device_create_info = vk::DeviceCreateInfo {
        p_next: &mut extended_state_features_ext as *mut _ as *mut std::ffi::c_void,
        queue_create_info_count: 1,
        p_queue_create_infos: &log_device_create_info,
        enabled_extension_count: len_of_device_extensions as u32,
        pp_enabled_extension_names: vec_with_cstrings.as_ptr(),
        ..Default::default()
    };
    unsafe {
        instance
            .create_device(main_device, &device_create_info, None)
            .expect("Failed to create vk::Device")
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

    // Возвращаем VK_FALSE, чтобы не прерывать выполнение
    vk::FALSE
}
fn main() {
    env_logger::init();
    log::info!("Initialized logger");
    let device_extension: Vec<&CStr> = vec![
        khr::swapchain::NAME,
        khr::spirv_1_4::NAME,
        khr::synchronization2::NAME,
        khr::create_renderpass2::NAME,
    ];
    let mut v_app = VulkanApp::new(
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        "tryingVULKAN!!!!",
        device_extension.clone(),
    );

    // pick window surface
    let surface = unsafe {
        let mut surface: vk::SurfaceKHR = vk::SurfaceKHR::null();
        log::trace!("SurfaceKHR::null()");
        let instance = v_app.instance.handle().as_raw() as *mut VkInstance_T;
        log::trace!("instance.handle() ...");
        let surface_ptr: *mut vk::SurfaceKHR = &mut surface;
        // let mut surf = [std::ptr::from_ref(&surface).cast_mut()].as_mut_ptr();
        log::trace!("before trying create window");
        v_app
            .window
            .create_window_surface(instance, std::ptr::null(), surface_ptr as *mut _);
        log::trace!("after");
        surface
    };

    // select physical device

    let main_device = select_physical_device(&v_app.instance, &device_extension);
    //logical device
    let queue_family_properties = unsafe {
        v_app
            .instance
            .get_physical_device_queue_family_properties(main_device)
    };
    let graphics_index = queue_family_properties
        .iter()
        .position(|x| x.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .unwrap();

    // create this var for sure what ptr will be valid as long as app run
    let queue_priority = 0.0f32;
    let log_device = vk::DeviceQueueCreateInfo {
        queue_family_index: graphics_index as u32,
        p_queue_priorities: std::ptr::addr_of!(queue_priority),
        queue_count: 1,
        ..Default::default()
    };
    let device = create_logical_device(device_extension, &v_app.instance, log_device, main_device);

    let graphics_queue = unsafe { device.get_device_queue(graphics_index as u32, 0) };

    let surf_instance = khr::surface::Instance::new(&v_app.entry, &v_app.instance);

    let present_supported: bool = unsafe {
        surf_instance
            .get_physical_device_surface_support(main_device, graphics_index as u32, surface)
            .unwrap()
    };
    log::trace!("present supported {present_supported:?}");
    let present_index = match present_supported {
        true => graphics_index,
        false => panic!(),
    };
    let present_queue = unsafe { device.get_device_queue(present_index as u32, 0) };
    let surface_capabilites = unsafe {
        surf_instance
            .get_physical_device_surface_capabilities(main_device, surface)
            .unwrap()
    };
    let available_formats = unsafe {
        surf_instance
            .get_physical_device_surface_formats(main_device, surface)
            .unwrap()
    };
    let available_present_modes = unsafe {
        surf_instance
            .get_physical_device_surface_present_modes(main_device, surface)
            .unwrap()
    };
    create_swap_chain(
        &v_app.window,
        surface_capabilites,
        available_present_modes,
        available_formats,
        surface,
        graphics_index as u32,
        present_index as u32,
        &v_app.instance,
        &device,
    );

    while !v_app.window.should_close() {
        // Swap front and back buffers

        // Poll for and process events
        v_app.glfw_var.poll_events();
        for (_, event) in glfw::flush_messages(&(v_app.events)) {
            println!("{event:?}");
            if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                v_app.window.set_should_close(true)
            }
        }
    }
    unsafe { device.destroy_device(None) };
    unsafe { v_app.instance.destroy_instance(None) };
    mem::drop(v_app._holder);
}
fn choose_swap_surface_format(vec: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    use ash::vk::*;
    *vec.iter()
        .find(|available| {
            available.color_space == ColorSpaceKHR::SRGB_NONLINEAR
                && available.format == Format::B8G8R8A8_SRGB
        })
        .unwrap_or(&vec[0])
}
fn choose_swap_present_mode(vec: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    use ash::vk::*;
    *vec.iter()
        .find(|available| **available == PresentModeKHR::MAILBOX)
        .unwrap_or(&PresentModeKHR::FIFO)
}
fn choose_swap_extent(
    capabilites: &vk::SurfaceCapabilitiesKHR,
    window: &glfw::PWindow,
) -> vk::Extent2D {
    if capabilites.current_extent.width == u32::MAX {
        let (width, height) = window.get_size();
        assert!(!(width < 0 || height < 0), "wtf width or height < 0");
        let (min, max) = (capabilites.min_image_extent, capabilites.max_image_extent);
        vk::Extent2D {
            width: (width as u32).clamp(min.width, max.width),
            height: (height as u32).clamp(min.height, max.height),
        }
    } else {
        capabilites.current_extent
    }
}
fn create_swap_chain(
    window: &glfw::PWindow,
    surface_capabilites: vk::SurfaceCapabilitiesKHR,
    present_modes: Vec<vk::PresentModeKHR>,
    formats: Vec<vk::SurfaceFormatKHR>,
    surface: vk::SurfaceKHR,
    graphics_family: u32,
    present_family: u32,
    instance: &ash::Instance,
    log_device: &ash::Device,
) -> vk::SwapchainKHR {
    use ash::vk::{
        Bool32, CompositeAlphaFlagsKHR, ImageUsageFlags, SharingMode, SurfaceTransformFlagsKHR,
        SwapchainCreateFlagsKHR, SwapchainKHR,
    };
    // mean true by specification
    let clipped: Bool32 = 1_u32;
    let pre_transform: SurfaceTransformFlagsKHR = { surface_capabilites.current_transform };
    let swap_surface_format = choose_swap_surface_format(&formats);
    let swap_present_mode = choose_swap_present_mode(&present_modes);
    let swap_extend = choose_swap_extent(&surface_capabilites, window);
    //  minImageCount = ( surfaceCapabilities.maxImageCount > 0
    //&& minImageCount > surfaceCapabilities.maxImageCount )
    // ? surfaceCapabilities.maxImageCount : minImageCount;
    //
    let mut min_image_count = 3.max(surface_capabilites.min_image_count);
    min_image_count = if surface_capabilites.max_image_count > 0
        && min_image_count > surface_capabilites.max_image_count
    {
        surface_capabilites.max_image_count
    } else {
        min_image_count
    };
    let mut image_count = surface_capabilites.min_image_count + 1;
    if surface_capabilites.max_image_count > 0 && image_count > surface_capabilites.max_image_count
    {
        image_count = surface_capabilites.max_image_count;
    }

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
    if graphics_family == present_family {
        swap_chain_create_info.image_sharing_mode = SharingMode::EXCLUSIVE;
        swap_chain_create_info.queue_family_index_count = 0;
        swap_chain_create_info.p_queue_family_indices = std::ptr::null();
    } else {
        swap_chain_create_info.image_sharing_mode = SharingMode::CONCURRENT;
        swap_chain_create_info.queue_family_index_count = 2;
        swap_chain_create_info.p_queue_family_indices = [graphics_family, present_family].as_ptr();
    }
    let device = ash::khr::swapchain::Device::new(instance, log_device);
    unsafe {
        device
            .create_swapchain(&swap_chain_create_info, None)
            .unwrap()
    }
}
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
    device: &vk::PhysicalDevice,
    instance: &ash::Instance,
    must_have_extension: &Vec<&CStr>,
) -> bool {
    let properties = unsafe { (*instance).get_physical_device_properties(*device) };
    log::trace!("device: {device:?} properties: {properties:?}");
    let features = unsafe { (*instance).get_physical_device_features(*device) };
    log::trace!("device : {device:?} features: {features:?}");
    let queue_families =
        unsafe { (*instance).get_physical_device_queue_family_properties(*device) };

    let extension_properties = unsafe {
        (*instance)
            .enumerate_device_extension_properties(*device)
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
