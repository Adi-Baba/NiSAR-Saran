//! Read-only Rust HDF5 boundary for the paired GCOV audit.
//!
//! The existing qualified `nz-hdf5` reader is reused for container inventory.
//! Raw numeric tile streaming is the next adapter layer; this boundary keeps
//! the audit from reimplementing the already-tested HDF5 qualification logic.

use nz_hdf5::{H5CatalogReport, H5CoverageReport, H5CoverageRow};
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use ndarray::s;
use std::path::Path;

#[allow(non_local_definitions)]
mod h5_types {
    #[derive(hdf5::H5Type, Debug, Clone, Copy)]
    #[repr(C)]
    pub struct H5F32Complex {
        pub r: f32,
        pub i: f32,
    }
}
use h5_types::H5F32Complex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexValue {
    pub re: f64,
    pub im: f64,
}

#[derive(Debug, Clone)]
pub struct PairInventory {
    pub before: H5CatalogReport,
    pub after: H5CatalogReport,
    pub matching_paths: Vec<String>,
}

pub fn inventory_pair(before: &Path, after: &Path) -> Result<PairInventory, String> {
    let before = nz_hdf5::catalog_h5(before.to_str().ok_or("invalid before path")?)
        .map_err(|e| format!("before catalog: {e:?}"))?;
    let after = nz_hdf5::catalog_h5(after.to_str().ok_or("invalid after path")?)
        .map_err(|e| format!("after catalog: {e:?}"))?;
    let mut matching = before.datasets.iter().filter_map(|b| {
        after.datasets.iter().any(|a| a.path == b.path).then(|| b.path.clone())
    }).collect::<Vec<_>>();
    matching.sort(); matching.dedup();
    Ok(PairInventory { before, after, matching_paths: matching })
}

pub fn numeric_rows(report: &H5CoverageReport) -> impl Iterator<Item = &H5CoverageRow> {
    report.rows.iter().filter(|row| row.dtype.contains("Float") || row.dtype.contains("Unsigned") || row.dtype.contains("Integer"))
}

/// Stream a numeric HDF5 dataset as row tiles converted to f64.
///
/// Rank-2 payloads are read in bounded row blocks of at most `rows_per_tile`
/// rows, strided on both axes, so a single tile never exceeds
/// `rows_per_tile * columns` elements and full-file coverage costs one
/// strided pass. Scalar and rank-1 datasets are read once because their
/// catalogued sizes are small. Unsupported compound, string, reference, and
/// enum payloads return `Ok(false)` and are retained in the ledger as
/// non-numeric rather than silently coerced. A missing dataset is an `Err`;
/// callers record it as a failed row, never as a zero.
pub fn stream_numeric_f64<F>(path: &Path, dataset_path: &str, rows_per_tile: usize, spatial_stride: usize, mut visit: F) -> Result<bool, String>
where F: FnMut(&[f64]) {
    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    let ds = file.dataset(dataset_path).map_err(|e| e.to_string())?;
    let desc = ds.dtype().map_err(|e| e.to_string())?.to_descriptor().map_err(|e| e.to_string())?;
    let shape = ds.shape();
    let rows = shape.first().copied().unwrap_or(1);
    let cols = if shape.len() >= 2 { shape[1] } else { 1 };
    let tile = rows_per_tile.max(1);
    let stride = spatial_stride.max(1);
    macro_rules! stream_2d { ($ty:ty, $convert:expr) => {{
        for start in (0..rows).step_by(tile) {
            let end = (start + tile).min(rows);
            let a = ds.read_slice::<$ty, _, ndarray::Ix2>(s![start..end; stride, 0..cols; stride]).map_err(|e| e.to_string())?;
            let values: Vec<f64> = a.iter().map($convert).collect(); visit(&values);
        }
    }}}
    macro_rules! stream_1d { ($ty:ty, $convert:expr) => {{
        let a = ds.read_raw::<$ty>().map_err(|e| e.to_string())?;
        let values: Vec<f64> = a.iter().map($convert).collect(); visit(&values);
    }}}
    match desc {
        TypeDescriptor::Float(FloatSize::U4) => { if shape.len() == 2 { stream_2d!(f32, |x: &f32| *x as f64); } else { stream_1d!(f32, |x: &f32| *x as f64); } }
        TypeDescriptor::Float(FloatSize::U8) => { if shape.len() == 2 { stream_2d!(f64, |x: &f64| *x); } else { stream_1d!(f64, |x: &f64| *x); } }
        TypeDescriptor::Unsigned(IntSize::U1) => { if shape.len() == 2 { stream_2d!(u8, |x: &u8| *x as f64); } else { stream_1d!(u8, |x: &u8| *x as f64); } }
        TypeDescriptor::Unsigned(IntSize::U2) => { if shape.len() == 2 { stream_2d!(u16, |x: &u16| *x as f64); } else { stream_1d!(u16, |x: &u16| *x as f64); } }
        TypeDescriptor::Unsigned(IntSize::U4) => { if shape.len() == 2 { stream_2d!(u32, |x: &u32| *x as f64); } else { stream_1d!(u32, |x: &u32| *x as f64); } }
        TypeDescriptor::Unsigned(IntSize::U8) => { if shape.len() == 2 { stream_2d!(u64, |x: &u64| *x as f64); } else { stream_1d!(u64, |x: &u64| *x as f64); } }
        TypeDescriptor::Integer(IntSize::U1) => { if shape.len() == 2 { stream_2d!(i8, |x: &i8| *x as f64); } else { stream_1d!(i8, |x: &i8| *x as f64); } }
        TypeDescriptor::Integer(IntSize::U2) => { if shape.len() == 2 { stream_2d!(i16, |x: &i16| *x as f64); } else { stream_1d!(i16, |x: &i16| *x as f64); } }
        TypeDescriptor::Integer(IntSize::U4) => { if shape.len() == 2 { stream_2d!(i32, |x: &i32| *x as f64); } else { stream_1d!(i32, |x: &i32| *x as f64); } }
        TypeDescriptor::Integer(IntSize::U8) => { if shape.len() == 2 { stream_2d!(i64, |x: &i64| *x as f64); } else { stream_1d!(i64, |x: &i64| *x as f64); } }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Stream an HDF5 compound `{r: f32, i: f32}` dataset in bounded hyperslabs.
///
/// Only the exact 8-byte `{r: f32 @0, i: f32 @4}` layout is accepted, so a
/// differently-ordered, padded, or wider compound can never be misread as
/// complex data. Other compounds return `Ok(false)` so their unsupported
/// status remains an explicit row in the evidence ledger. Tiles are bounded
/// exactly like [`stream_numeric_f64`]: at most `rows_per_tile` rows per
/// visitor call, strided on both axes.
pub fn stream_complex_f32<F>(
    path: &Path,
    dataset_path: &str,
    rows_per_tile: usize,
    spatial_stride: usize,
    mut visit: F,
) -> Result<bool, String>
where
    F: FnMut(&[ComplexValue]),
{
    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    let ds = file.dataset(dataset_path).map_err(|e| e.to_string())?;
    let descriptor = ds.dtype().map_err(|e| e.to_string())?.to_descriptor().map_err(|e| e.to_string())?;
    let TypeDescriptor::Compound(compound) = descriptor else { return Ok(false); };
    let valid = compound.size == 8 && compound.fields.len() == 2 &&
        compound.fields[0].name == "r" && compound.fields[1].name == "i" &&
        compound.fields.iter().all(|field| field.ty == TypeDescriptor::Float(FloatSize::U4));
    if !valid { return Ok(false); }
    let shape = ds.shape();
    let stride = spatial_stride.max(1);
    let convert = |values: &[H5F32Complex]| values.iter().map(|z| ComplexValue { re: z.r as f64, im: z.i as f64 }).collect::<Vec<_>>();
    if shape.len() >= 2 {
        let rows = shape[0];
        let cols = shape[1];
        let tile = rows_per_tile.max(1);
        for start in (0..rows).step_by(tile) {
            let end = (start + tile).min(rows);
            let array = ds.read_slice::<H5F32Complex, _, ndarray::Ix2>(s![start..end; stride, 0..cols; stride]).map_err(|e| e.to_string())?;
            let raw: Vec<H5F32Complex> = array.iter().copied().collect();
            visit(&convert(&raw));
        }
    } else {
        let raw = ds.read_raw::<H5F32Complex>().map_err(|e| e.to_string())?;
        visit(&convert(&raw));
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complex_value_layout_is_explicit() {
        assert_eq!(std::mem::size_of::<H5F32Complex>(), 8);
        assert_eq!(ComplexValue { re: 1.0, im: -2.0 }.im, -2.0);
    }
}
