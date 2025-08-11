use core::panic;
use std::{
    ffi::{CStr, CString, c_char},
    mem,
};

use ash::{
    Entry, khr,
    vk::{self, PhysicalDeviceType},
};
use glfw::{self, Action, Context, Key};
static SCREEN_WIDTH: u32 = 1920;
static SCREEN_HEIGHT: u32 = 1080;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
#[cfg(debug_assertions)]
const LAYERS_TO_ENABLE: &[&str] = VALIDATION_LAYERS;
#[cfg(not(debug_assertions))]
const layers_to_enable: &[&str] = &[]; // В релизе слои отключены

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

    let (mut window, events) = glfw
        .create_window(width, height, window_name, glfw::WindowMode::Windowed)
        .expect("Failed to create window");
    window.make_current();
    window.set_key_polling(true);
    (glfw, window, events)
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
    if devices.is_empty() {
        panic!();
    }
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

    let (mut glfw, mut window, events) =
        init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "tryingVULKAN!!!");

    let entry = Entry::linked();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 4, 0),
        ..Default::default()
    };
    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
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
        .pfn_user_callback(Some(callback));
    check_layers(&entry);

    let extension_properties: Vec<vk::ExtensionProperties> = unsafe {
        entry
            .enumerate_instance_extension_properties(None)
            .expect("Failed to get instance extension properties")
    };
    log::info!("Vulkan having extensions: {extension_properties:#?}");

    let glfw_extensions: Vec<String> = glfw
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
    // must live as long as app and dropped manually at the end
    let layers = CStringArray::from(LAYERS_TO_ENABLE);
    let instance = create_instance(
        glfw_extensions,
        &app_info,
        &entry,
        &layers,
        &mut debug_create_info,
    );

    // select physical device

    let device_extension: Vec<&CStr> = vec![
        khr::swapchain::NAME,
        khr::spirv_1_4::NAME,
        khr::synchronization2::NAME,
        khr::create_renderpass2::NAME,
    ];
    let main_device = select_physical_device(&instance, &device_extension);
    //logical device
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(main_device) };
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
    let device = create_logical_device(device_extension, &instance, log_device, main_device);

    let graphics_queue = unsafe { device.get_device_queue(graphics_index as u32, 0) };

    while !window.should_close() {
        // Swap front and back buffers
        window.swap_buffers();

        // Poll for and process events
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            println!("{event:?}");
            if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                window.set_should_close(true)
            }
        }
    }
    unsafe { device.destroy_device(None) };
    unsafe { instance.destroy_instance(None) };
    mem::drop(layers);
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
