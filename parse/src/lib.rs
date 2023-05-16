use std::ffi::CStr;
use std::os::raw::c_char;

use batch::Batch;
use input_features::InputFeatureSetType;

use crate::data_loader::BatchReader;

mod batch;
mod data_loader;
mod input_features;

#[no_mangle]
pub unsafe extern "C" fn batch_new(
    batch_size: u32,
    max_features: u32,
    indices_per_feature: u32,
) -> *mut Batch {
    let batch = Batch::new(
        batch_size as usize,
        max_features as usize,
        indices_per_feature as usize,
    );
    Box::into_raw(Box::new(batch))
}

#[no_mangle]
pub unsafe extern "C" fn batch_drop(batch: *mut Batch) {
    let _ = Box::from_raw(batch);
}

macro_rules! export_batch_getters {
    ($($getter:ident $(as $cast_type:ty)?: $exported:ident -> $type:ty,)*) => {$(
        #[no_mangle]
        pub unsafe extern "C" fn $exported(batch: *mut Batch) -> $type {
            batch.as_mut().unwrap().$getter() $(as $cast_type)*
        }
    )*}
}
export_batch_getters! {
    capacity as u32                 : batch_get_capacity -> u32,
    len as u32                      : batch_get_len -> u32,
    stm_feature_buffer_ptr          : batch_get_stm_feature_buffer_ptr -> *const i64,
    nstm_feature_buffer_ptr         : batch_get_nstm_feature_buffer_ptr -> *const i64,
    values_ptr                      : batch_get_values_ptr -> *const f32,
    total_features as u32           : batch_get_total_features -> u32,
    indices_per_feature as u32      : batch_get_indices_per_feature -> u32,
    cp_ptr                          : batch_get_cp_ptr -> *const f32,
    wdl_ptr                         : batch_get_wdl_ptr -> *const f32,
}

#[no_mangle]
pub unsafe extern "C" fn batch_reader_new(
    path: *const c_char,
    batch_size: u32,
    feature_set: InputFeatureSetType,
) -> *mut BatchReader {
    let reader = (|| {
        let path = CStr::from_ptr(path).to_str().ok()?;
        let reader = BatchReader::new(path.as_ref(), feature_set, batch_size as usize).ok()?;
        Some(reader)
    })();
    match reader {
        Some(reader) => Box::into_raw(Box::new(reader)),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn batch_reader_dataset_size(reader: *mut BatchReader) -> u64 {
    let reader = reader.as_mut().unwrap();
    reader.dataset_size()
}

#[no_mangle]
pub unsafe extern "C" fn batch_reader_drop(reader: *mut BatchReader) {
    let _ = Box::from_raw(reader);
}

#[no_mangle]
pub unsafe extern "C" fn input_feature_set_get_max_features(
    feature_set: InputFeatureSetType,
) -> u32 {
    feature_set.max_features() as u32
}

#[no_mangle]
pub unsafe extern "C" fn input_feature_set_get_indices_per_feature(
    feature_set: InputFeatureSetType,
) -> u32 {
    feature_set.indices_per_feature() as u32
}

#[no_mangle]
pub unsafe extern "C" fn read_batch(reader: *mut BatchReader) -> *mut Batch {
    let reader = reader.as_mut().unwrap();
    match reader.next_batch() {
        Some(v) => Box::into_raw(v),
        None => std::ptr::null_mut(),
    }
}
