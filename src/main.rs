use std::{error::Error, ffi::CStr};

use ash::{
    Entry,
    vk::{self, ExtendsFormatProperties2},
};
use glfw::{self, Action, Context, Key};
static SCREEN_WIDTH: u32 = 1920;
static SCREEN_HEIGHT: u32 = 1080;

fn main() {
    env_logger::init();
    use glfw::fail_on_errors;
    log::info!("Initialized logger");

    let mut glfw = glfw::init(fail_on_errors!()).expect("Failed to init glfw init");

    let (mut window, events) = glfw
        .create_window(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            "TRYING VULKAN!",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create window");
    window.make_current();
    window.set_key_polling(true);

    let entry = Entry::linked();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 4, 0),
        ..Default::default()
    };

    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };
    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("can't create instance")
    };
    let extension_properties: Vec<vk::ExtensionProperties> = unsafe {
        entry
            .enumerate_instance_extension_properties(None)
            .expect("Failed to get instance extension properties")
    };

    let glfw_extensions: Vec<String> = glfw
        .get_required_instance_extensions()
        .expect("glfw api unavaible");

    glfw_extensions.into_iter().all(|glfw_extension| {
        log::info!("extension = {glfw_extension}");
        let ext_as_cstring = std::ffi::CString::new(glfw_extension.into_bytes())
            .expect("Theres must not be nulls(i think)");
        extension_properties
            .iter()
            .any(|extension| match extension.extension_name_as_c_str() {
                Ok(str) => str == ext_as_cstring.as_c_str(),
                Err(e) => {
                    log::error!("Bullshti whit CStrings {e}");
                    panic!("");
                }
            })
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
}
