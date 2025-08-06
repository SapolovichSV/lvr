use ash::vk;
use glfw::{Action, Context, Key};
use std::error::Error;
struct VulkanApp {
    window: glfw::PWindow,
    events: glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
    glfw: glfw::Glfw,
}

impl VulkanApp {
    fn new() -> Result<Self, Box<dyn Error>> {
        use glfw::fail_on_errors;
        let mut glfw = glfw::init(fail_on_errors!()).unwrap();

        let (mut window, events): (glfw::PWindow, glfw::GlfwReceiver<(f64, glfw::WindowEvent)>) =
            glfw.create_window(800, 600, "glfw window rust", glfw::WindowMode::Windowed)
                .expect("Failed to create glfw window");
        window.make_current();
        window.set_key_polling(true);
        let entry = unsafe { ash::Entry::load()? };
        let appinfo = vk::ApplicationInfo {
            application_version: vk::make_api_version(0, 1, 4, 0),
            ..Default::default()
        };
        let createInfo = vk::InstanceCreateInfo {
            p_application_info: &appinfo,
            ..Default::default()
        };

        Ok(Self {
            window,
            events,
            glfw,
        })
    }

    fn run(&mut self) {
        log::info!("Running application");
    }
    fn init_vulkan() {}
    fn init_window() {}
    fn main_loop(&mut self) {
        while !self.window.should_close() {
            self.window.swap_buffers();
        }
    }
}

fn main() {
    env_logger::init();

    match VulkanApp::new() {
        Ok(mut app) => app.run(),
        Err(error) => log::error!("Failed to create application cause : {error}"),
    }
}
