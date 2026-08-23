//! The plain-array matrix types of the public API, and the conversions to the
//! `nalgebra` types used for the math behind it.
use nalgebra as na;

/// A 3×3 matrix in row-major order: `m[row][col]`.
///
/// Row-major is the layout Kalibr's YAML uses, so a matrix reads the same in
/// Rust as it does in the calibration file.
pub type Mat3 = [[f64; 3]; 3];

/// A 4×4 homogeneous transform in row-major order: `m[row][col]`.
pub type Mat4 = [[f64; 4]; 4];

/// A 3-element vector, `[x, y, z]`.
pub type Vec3 = [f64; 3];

/// The 3×3 identity.
pub const IDENTITY3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// The 3×3 zero matrix.
pub const ZERO3: Mat3 = [[0.0; 3]; 3];

/// The zero vector.
pub const ZERO_VEC3: Vec3 = [0.0; 3];

pub(crate) fn mat3(m: &Mat3) -> na::Matrix3<f64> {
    na::Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    )
}

pub(crate) fn unmat3(m: &na::Matrix3<f64>) -> Mat3 {
    let mut out = ZERO3;
    for (r, row) in out.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = m[(r, c)];
        }
    }
    out
}

/// `[x, y, z]` as an `nalgebra` column vector.
pub(crate) fn vec3(v: &Vec3) -> na::Vector3<f64> {
    na::Vector3::new(v[0], v[1], v[2])
}

pub(crate) fn unvec3(v: &na::Vector3<f64>) -> Vec3 {
    [v.x, v.y, v.z]
}

/// Euclidean norm, so call sites do not have to convert just to measure one.
pub(crate) fn norm3(v: &Vec3) -> f64 {
    v[0].hypot(v[1]).hypot(v[2])
}

/// Distance between two vectors.
pub(crate) fn dist3(a: &Vec3, b: &Vec3) -> f64 {
    norm3(&[a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}
