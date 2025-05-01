#![allow(dead_code)]
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use std::{
    fs::{File, read_to_string},
    io::Write,
};
use base64::prelude::{Engine as _, BASE64_STANDARD};

// Record serialized value to file
pub fn record_value<T: CanonicalSerialize + ?Sized>(filename: &str, val: &T) {
    let mut f = File::create(filename).unwrap();
    let mut ser_val: Vec<u8> = Vec::new();
    val.serialize_compressed(&mut ser_val).unwrap();
    //println!("Write {} B\n{:?}", ser_val.len(), ser_val);
    let ser_str = BASE64_STANDARD.encode(ser_val);
    //println!("Base64 Write {}", ser_str);
    writeln!(f, "{}", ser_str).unwrap();
}

// Get deserialized value from file
pub fn get_value<T: CanonicalDeserialize + ?Sized>(filename: &str) -> T {
    let mut ser_str = read_to_string(filename).unwrap();
    ser_str.pop();
    //println!("Base64 Read {}", ser_str);
    let ser_val = BASE64_STANDARD.decode(ser_str).unwrap();
    //println!("Read {} B\n{:?}", ser_val.len(), ser_val);
    T::deserialize_compressed(&*ser_val).unwrap()
}
