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
    let enabled_layer_c_names = layers.as_ptr();
    let len_ext_names = glfw_extensions.len();
    let extension_names = vec_strings_to_ptptr(glfw_extensions);
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        enabled_extension_count: len_ext_names as u32,
        pp_enabled_extension_names: extension_names,
        enabled_layer_count: LAYERS_TO_ENABLE.len() as u32,
        pp_enabled_layer_names: enabled_layer_c_names,
        ..Default::default()
    };
    let instance: ash::Instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("can't create instance")
    };
    let devices: Vec<vk::PhysicalDevice> = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Can't enumerate physical devices")
    };
    if devices.is_empty() {
        panic!();
    }
    log::debug!("devices: {devices:#?}");

    let device_extension: Vec<&CStr> = vec![
        khr::swapchain::NAME,
        khr::spirv_1_4::NAME,
        khr::synchronization2::NAME,
        khr::create_renderpass2::NAME,
    ];

    let mut main_device = Default::default();
    devices.iter().for_each(|device| {
        if is_device_suitable(device, &instance, &device_extension) {
            log::debug!("device {device:?} is suitable");
            main_device = *device;
        }
    });

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
