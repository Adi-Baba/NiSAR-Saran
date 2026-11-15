//! Bounded-ingestion integration tests for the HDF5 boundary.
//!
//! Each test builds a small real HDF5 file, streams it through the public
//! boundary, and checks tiling bounds, strided subsampling, exact complex
//! layout matching, and the unsupported-type sentinel — the same contract
//! the October run relied on over 288 matched paths.

#![allow(non_local_definitions)]

use audit_hdf5::{stream_complex_f32, stream_numeric_f64};
use hdf5::H5Type;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!("audit_hdf5_bounded_{}_{}.h5", std::process::id(), id))
}

#[derive(H5Type, Clone, Copy, Debug)]
#[repr(C)]
struct Pair {
    r: f32,
    i: f32,
}

#[derive(H5Type, Clone, Copy, Debug)]
#[repr(C)]
struct Wrong {
    x: f64,
    y: f64,
}

fn collect_numeric(path: &std::path::Path, name: &str, tile: usize, stride: usize) -> (bool, Vec<Vec<f64>>) {
    let mut tiles = Vec::new();
    let ok = stream_numeric_f64(path, name, tile, stride, |values| tiles.push(values.to_vec())).unwrap();
    (ok, tiles)
}

#[test]
fn tiled_reads_cover_the_full_grid_within_bounds() {
    let path = scratch();
    let file = hdf5::File::create(&path).unwrap();
    let data: ndarray::Array2<f32> = ndarray::Array2::from_shape_fn((6, 8), |(r, c)| (r * 8 + c) as f32);
    file.new_dataset::<f32>().shape([6, 8]).create("grid").unwrap().write(&data).unwrap();
    file.close().unwrap();

    let (ok, tiles) = collect_numeric(&path, "grid", 2, 1);
    assert!(ok);
    // Three tiles of exactly 2x8: no tile exceeds rows_per_tile * columns.
    assert_eq!(tiles.len(), 3);
    assert!(tiles.iter().all(|t| t.len() == 16));
    let flat: Vec<f64> = tiles.concat();
    let expected: Vec<f64> = (0..48).map(|v| v as f64).collect();
    assert_eq!(flat, expected);
    std::fs::remove_file(&path).ok();
}

#[test]
fn strided_reads_subsample_both_axes() {
    let path = scratch();
    let file = hdf5::File::create(&path).unwrap();
    let data: ndarray::Array2<f32> = ndarray::Array2::from_shape_fn((6, 8), |(r, c)| (r * 8 + c) as f32);
    file.new_dataset::<f32>().shape([6, 8]).create("grid").unwrap().write(&data).unwrap();
    file.close().unwrap();

    let (ok, tiles) = collect_numeric(&path, "grid", 4, 2);
    assert!(ok);
    let flat: Vec<f64> = tiles.concat();
    // Rows {0,2,4} x cols {0,2,4,6} in row-major order.
    let mut expected = Vec::new();
    for r in [0, 2, 4] {
        for c in [0, 2, 4, 6] {
            expected.push((r * 8 + c) as f64);
        }
    }
    assert_eq!(flat, expected);
    std::fs::remove_file(&path).ok();
}

#[test]
fn unsigned_payloads_convert_without_coercion_gaps() {
    let path = scratch();
    let file = hdf5::File::create(&path).unwrap();
    let data: ndarray::Array2<u8> = ndarray::Array2::from_shape_fn((4, 4), |(r, c)| (r * 4 + c) as u8);
    file.new_dataset::<u8>().shape([4, 4]).create("mask").unwrap().write(&data).unwrap();
    file.close().unwrap();

    let (ok, tiles) = collect_numeric(&path, "mask", 64, 1);
    assert!(ok);
    assert_eq!(tiles.concat(), (0..16).map(|v| v as f64).collect::<Vec<_>>());
    std::fs::remove_file(&path).ok();
}

#[test]
fn exact_complex_layout_streams_and_wrong_layouts_refuse() {
    let path = scratch();
    let file = hdf5::File::create(&path).unwrap();
    let good: ndarray::Array2<Pair> =
        ndarray::Array2::from_shape_fn((2, 3), |(r, c)| Pair { r: r as f32, i: c as f32 });
    file.new_dataset::<Pair>().shape([2, 3]).create("good").unwrap().write(&good).unwrap();
    let bad: ndarray::Array2<Wrong> =
        ndarray::Array2::from_shape_fn((2, 2), |(r, c)| Wrong { x: r as f64, y: c as f64 });
    file.new_dataset::<Wrong>().shape([2, 2]).create("bad").unwrap().write(&bad).unwrap();
    file.close().unwrap();

    let mut seen = Vec::new();
    let ok = stream_complex_f32(&path, "good", 1, 1, |tile| {
        // One row of 3 per visitor call: the tile bound holds.
        assert_eq!(tile.len(), 3);
        seen.extend(tile.iter().map(|z| (z.re, z.im)));
    })
    .unwrap();
    assert!(ok);
    assert_eq!(seen, vec![(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]);

    // f64 fields are not the accepted 8-byte {r,i} f32 layout.
    let refused = stream_complex_f32(&path, "bad", 8, 1, |_| panic!("must not visit")).unwrap();
    assert!(!refused);
    std::fs::remove_file(&path).ok();
}

#[test]
fn strings_are_unsupported_and_missing_is_an_error() {
    let path = scratch();
    let file = hdf5::File::create(&path).unwrap();
    let words = [hdf5::types::FixedAscii::<4>::from_ascii(b"HHHH").unwrap()];
    file.new_dataset::<hdf5::types::FixedAscii<4>>()
        .shape([1])
        .create("tags")
        .unwrap()
        .write(&words[..])
        .unwrap();
    file.close().unwrap();

    assert!(!stream_numeric_f64(&path, "tags", 8, 1, |_| panic!("must not visit")).unwrap());
    assert!(!stream_complex_f32(&path, "tags", 8, 1, |_| panic!("must not visit")).unwrap());
    assert!(stream_numeric_f64(&path, "absent", 8, 1, |_| {}).is_err());
    std::fs::remove_file(&path).ok();
}
