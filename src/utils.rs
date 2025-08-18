use std::{io::Error, path::Path};
#[derive(Debug)]
pub struct ReadFileError(String);

pub fn read_file<T: AsRef<Path>>(filename: T) -> Result<Vec<u32>, ReadFileError> {
    log::trace!("try reading file: {}", filename.as_ref().display());
    let data = std::fs::read_to_string(filename)?;
    let parsed: Vec<u32> = data.chars().map(|char| char as u32).collect();
    Ok(parsed)
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
