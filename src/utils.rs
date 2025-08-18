use std::{io::Error, path::Path};
#[derive(Debug)]
pub struct ReadFileError(String);

pub fn read_file<T: AsRef<Path>>(filename: T) -> Vec<u32> {
    log::trace!("try reading file: {}", filename.as_ref().display());
    let data: Vec<u32> = include_bytes!("../shaders/slang.spv")
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    log::trace!("file data is {data:?}");
    data
}
impl From<std::io::Error> for ReadFileError {
    fn from(value: Error) -> Self {
        let descr: String = String::from("can't read file error: ") + value.to_string().as_str();
        Self(descr)
    }
}
impl From<String> for ReadFileError {
    fn from(value: String) -> Self {
        Self(value)
    }
}
impl From<ReadFileError> for String {
    fn from(value: ReadFileError) -> Self {
        value.0
    }
}
