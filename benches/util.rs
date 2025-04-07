use ark_serialize::{CanonicalSerialize, Compress};
use std::{
    fs::OpenOptions,
    io::{Write as StdWrite},
};

pub const SIZE_LOG_FILE: &str = "./benches/logs/sizes.csv";

// Record DESC,SIZE in the CSV file
pub fn record_size(desc: impl AsRef<str>, val: &(impl CanonicalSerialize + ?Sized)) {
    let mut f = OpenOptions::new().append(true).open(SIZE_LOG_FILE).unwrap();
    let size = val.serialized_size(Compress::No);
    let compressed_size = val.serialized_size(Compress::Yes);
    writeln!(f, "{},{},{}", desc.as_ref(), size, compressed_size).unwrap();
}
