use ash::vk;
pub struct ShaderWrap<'a> {
    device: &'a ash::Device,
    handle: vk::ShaderModule,
}
impl<'a> ShaderWrap<'a> {
    /// may panic
    pub fn new(device: &'a ash::Device, code: &[u32]) -> Self {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        let shader_module = unsafe { device.create_shader_module(&create_info, None).unwrap() };

        Self {
            device,
            handle: shader_module,
        }
    }
}
impl Drop for ShaderWrap<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.handle, None);
        }
    }
}
