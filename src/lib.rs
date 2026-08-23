//! Kalibr-compatible IMU intrinsic calibration for Rust.
//!
//! * [`allan`] estimates noise densities and bias random walks from a long
//!   stationary recording.
//! * [`estimate`] estimates scale, axis misalignment and biases from static
//!   orientations and the rotations between them.
//! * [`ImuIntrinsics`] and [`ImuCorrector`] hold those parameters in Kalibr's
//!   YAML layout and apply them to a live stream.
//!
//! ```rust
//! use imu_calib::ImuIntrinsics;
//!
//! let intrinsics = ImuIntrinsics::identity();
//! let corrector = intrinsics.corrector().unwrap();
//!
//! let (accel, gyro) = corrector.correct([0.0, 0.0, 9.81], [0.01, -0.02, 0.0]);
//! ```

use nalgebra as na;

mod conv;

use conv::{mat3, unvec3, vec3};
/// Plain-array matrix and vector types. The public API is expressed entirely
/// in these, so `nalgebra` stays an implementation detail of this crate.
pub use conv::{Mat3, Mat4, Vec3, IDENTITY3, ZERO3, ZERO_VEC3};

pub mod allan;
pub mod estimate;

#[cfg(feature = "kalibr")]
pub mod kalibr;

/// IMU message container compatible with standard ROS sensor messages.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImuMsg {
    /// Orientation quaternion `[x, y, z, w]`.
    pub orientation: [f64; 4],
    /// Row-major 3x3 covariance matrix for orientation.
    pub orientation_covariance: [f64; 9],
    /// Angular velocity `[x, y, z]` in rad/s.
    pub angular_velocity: [f64; 3],
    /// Row-major 3x3 covariance matrix for angular velocity.
    pub angular_velocity_covariance: [f64; 9],
    /// Linear acceleration `[x, y, z]` in m/s^2.
    pub linear_acceleration: [f64; 3],
    /// Row-major 3x3 covariance matrix for linear acceleration.
    pub linear_acceleration_covariance: [f64; 9],
}

impl ImuMsg {
    /// Create a new `ImuMsg` with the given linear acceleration and angular velocity.
    pub fn new(linear_accel: &[f64; 3], angular_vel: &[f64; 3]) -> Self {
        Self {
            linear_acceleration: *linear_accel,
            angular_velocity: *angular_vel,
            ..Default::default()
        }
    }
}

/// Which of Kalibr's IMU intrinsic models a parameter set belongs to.
///
/// Selected on the Kalibr command line with `--imu-models`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImuModel {
    /// `calibrated`: only noise parameters, no deterministic intrinsics.
    #[default]
    Calibrated,
    /// `scale-misalignment`: scale, misalignment and g-sensitivity.
    ScaleMisalignment,
    /// `scale-misalignment-size-effect`: additionally per-axis accelerometer
    /// lever arms. This crate parses the lever arms but does not apply them.
    ScaleMisalignmentSizeEffect,
}

impl ImuModel {
    /// The string Kalibr writes into the `model` field of its YAML output.
    pub fn as_str(&self) -> &'static str {
        match self {
            ImuModel::Calibrated => "calibrated",
            ImuModel::ScaleMisalignment => "scale-misalignment",
            ImuModel::ScaleMisalignmentSizeEffect => "scale-misalignment-size-effect",
        }
    }

    /// Parse the `model` field of a Kalibr YAML file.
    pub fn from_str_kalibr(s: &str) -> anyhow::Result<Self> {
        match s {
            "calibrated" => Ok(ImuModel::Calibrated),
            "scale-misalignment" => Ok(ImuModel::ScaleMisalignment),
            "scale-misalignment-size-effect" => Ok(ImuModel::ScaleMisalignmentSizeEffect),
            other => Err(anyhow::anyhow!("unknown Kalibr IMU model `{other}`")),
        }
    }
}

/// Continuous-time stochastic error model of an IMU.
///
/// These are exactly the four values Kalibr reads from its input `imu0.yaml`,
/// plus the update rate used to discretise them. Obtain them with
/// [`allan::AllanEstimator`] from a long stationary recording, or read them off
/// the datasheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuNoise {
    /// Accelerometer white-noise density σ_a \[m/s²/√Hz\].
    pub accel_noise_density: f64,
    /// Accelerometer bias random-walk σ_ba \[m/s³/√Hz\].
    pub accel_random_walk: f64,
    /// Gyroscope white-noise density σ_g \[rad/s/√Hz\].
    pub gyro_noise_density: f64,
    /// Gyroscope bias random-walk σ_bg \[rad/s²/√Hz\].
    pub gyro_random_walk: f64,
    /// IMU update rate \[Hz\].
    pub update_rate: f64,
}

impl Default for ImuNoise {
    fn default() -> Self {
        Self {
            accel_noise_density: 0.0,
            accel_random_walk: 0.0,
            gyro_noise_density: 0.0,
            gyro_random_walk: 0.0,
            update_rate: 200.0,
        }
    }
}

impl ImuNoise {
    /// Discrete-time accelerometer noise σ: `σ_d = σ / √Δt`.
    pub fn accel_noise_discrete(&self) -> f64 {
        self.accel_noise_density * self.update_rate.sqrt()
    }

    /// Discrete-time gyroscope noise σ: `σ_d = σ / √Δt`.
    pub fn gyro_noise_discrete(&self) -> f64 {
        self.gyro_noise_density * self.update_rate.sqrt()
    }

    /// Discrete-time accelerometer bias random-walk σ: `σ_d = σ · √Δt`.
    pub fn accel_bias_discrete(&self) -> f64 {
        self.accel_random_walk / self.update_rate.sqrt()
    }

    /// Discrete-time gyroscope bias random-walk σ: `σ_d = σ · √Δt`.
    pub fn gyro_bias_discrete(&self) -> f64 {
        self.gyro_random_walk / self.update_rate.sqrt()
    }
}

/// Kalibr-compatible IMU intrinsic parameters.
///
/// The matrices are stored exactly as Kalibr reports them, i.e. in the
/// ideal → raw direction described at the [crate] level. Build an
/// [`ImuCorrector`] with [`corrector`](Self::corrector) to apply them.
#[derive(Debug, Clone)]
pub struct ImuIntrinsics {
    /// Which Kalibr model these parameters belong to.
    pub model: ImuModel,

    /// Accelerometer scale-misalignment matrix `M_a` (lower-triangular 3×3).
    ///
    /// Kalibr YAML key `accelerometers.M`.
    pub accel_m: Mat3,

    /// Accelerometer bias `b_a` \[m/s²\].
    ///
    /// Kalibr models the bias as a time-varying B-spline and does not export
    /// it; this is estimated by [`estimate`] and stored as an extension.
    pub accel_bias: Vec3,

    /// Gyroscope scale-misalignment matrix `M_g` (lower-triangular 3×3).
    ///
    /// Kalibr YAML key `gyroscopes.M`.
    pub gyro_m: Mat3,

    /// Rotation `C_gyro_i` from the accelerometer frame to the gyroscope frame.
    ///
    /// Kalibr YAML key `gyroscopes.C_gyro_i`.
    pub c_gyro_i: Mat3,

    /// G-sensitivity `A` \[(rad/s)/(m/s²)\]: acceleration coupling into the gyro.
    ///
    /// Kalibr YAML key `gyroscopes.A`. Zero when not calibrated.
    pub gyro_a: Mat3,

    /// Gyroscope bias `b_g` \[rad/s\]. See [`accel_bias`](Self::accel_bias).
    pub gyro_bias: Vec3,

    /// Stochastic error model.
    pub noise: ImuNoise,

    /// Topic the calibration was recorded from (Kalibr YAML key `rostopic`).
    pub rostopic: Option<String>,

    /// Transformation from the body frame to this IMU (Kalibr key `T_i_b`).
    ///
    /// Identity for the reference IMU of a Kalibr run.
    pub t_i_b: Option<Mat4>,

    /// Time offset with respect to IMU0 \[s\] (Kalibr key `time_offset`).
    pub time_offset: f64,

    /// Per-axis accelerometer lever arms of the `scale-misalignment-size-effect`
    /// model \[m\], in the order `rx_i, ry_i, rz_i`. Parsed but not applied.
    pub accel_lever_arms: Option<[Vec3; 3]>,
}

impl Default for ImuIntrinsics {
    fn default() -> Self {
        Self::identity()
    }
}

impl ImuIntrinsics {
    /// An identity (pass-through) parameter set with zero noise.
    pub fn identity() -> Self {
        Self {
            model: ImuModel::Calibrated,
            accel_m: IDENTITY3,
            accel_bias: ZERO_VEC3,
            gyro_m: IDENTITY3,
            c_gyro_i: IDENTITY3,
            gyro_a: ZERO3,
            gyro_bias: ZERO_VEC3,
            noise: ImuNoise::default(),
            rostopic: None,
            t_i_b: None,
            time_offset: 0.0,
            accel_lever_arms: None,
        }
    }

    /// Pre-compute the inverted matrices needed to correct measurements.
    ///
    /// Fails if `M_a` or `M_g` is singular, which means the parameters are not
    /// a valid calibration.
    pub fn corrector(&self) -> anyhow::Result<ImuCorrector> {
        ImuCorrector::new(self)
    }

    /// Per-axis accelerometer scale factors, the diagonal of `M_a`.
    pub fn accel_scale(&self) -> [f64; 3] {
        [self.accel_m[0][0], self.accel_m[1][1], self.accel_m[2][2]]
    }

    /// Per-axis gyroscope scale factors, the diagonal of `M_g`.
    pub fn gyro_scale(&self) -> [f64; 3] {
        [self.gyro_m[0][0], self.gyro_m[1][1], self.gyro_m[2][2]]
    }
}

/// Applies [`ImuIntrinsics`] to raw measurements.
///
/// Holds the pre-inverted matrices so that correcting one sample costs two
/// matrix-vector products. Create it with [`ImuIntrinsics::corrector`].
#[derive(Debug, Clone)]
pub struct ImuCorrector {
    /// `M_a⁻¹`
    accel_inv: na::Matrix3<f64>,
    /// `C_gyro_iᵀ · M_g⁻¹`
    gyro_inv: na::Matrix3<f64>,
    /// `A · C_gyro_i`
    gyro_accel_coupling: na::Matrix3<f64>,
    accel_bias: na::Vector3<f64>,
    gyro_bias: na::Vector3<f64>,
}

impl ImuCorrector {
    /// Build a corrector from a parameter set.
    pub fn new(intrinsics: &ImuIntrinsics) -> anyhow::Result<Self> {
        let accel_inv = mat3(&intrinsics.accel_m)
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("accelerometer matrix M_a is singular"))?;
        let gyro_m_inv = mat3(&intrinsics.gyro_m)
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("gyroscope matrix M_g is singular"))?;
        let c_gyro_i = mat3(&intrinsics.c_gyro_i);

        Ok(Self {
            accel_inv,
            gyro_inv: c_gyro_i.transpose() * gyro_m_inv,
            gyro_accel_coupling: mat3(&intrinsics.gyro_a) * c_gyro_i,
            accel_bias: vec3(&intrinsics.accel_bias),
            gyro_bias: vec3(&intrinsics.gyro_bias),
        })
    }

    /// A corrector that passes measurements through unchanged.
    pub fn identity() -> Self {
        Self {
            accel_inv: na::Matrix3::identity(),
            gyro_inv: na::Matrix3::identity(),
            gyro_accel_coupling: na::Matrix3::zeros(),
            accel_bias: na::Vector3::zeros(),
            gyro_bias: na::Vector3::zeros(),
        }
    }

    /// Correct a raw accelerometer reading: `a = M_a⁻¹ (a_raw − b_a)`.
    pub fn correct_accel(&self, raw_accel: [f64; 3]) -> [f64; 3] {
        let a = self.accel_inv * (na::Vector3::from(raw_accel) - self.accel_bias);
        unvec3(&a)
    }

    /// Correct a raw gyroscope reading, given the already corrected acceleration:
    /// `w = C_gyro_iᵀ M_g⁻¹ (w_raw − b_g − A C_gyro_i a)`.
    pub fn correct_gyro(&self, raw_gyro: [f64; 3], corrected_accel: [f64; 3]) -> [f64; 3] {
        let w_raw = na::Vector3::from(raw_gyro);
        let a = na::Vector3::from(corrected_accel);
        let w = self.gyro_inv * (w_raw - self.gyro_bias - self.gyro_accel_coupling * a);
        unvec3(&w)
    }

    /// Correct both readings, returning `(accel, gyro)`.
    pub fn correct(&self, raw_accel: [f64; 3], raw_gyro: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        let a = self.correct_accel(raw_accel);
        let w = self.correct_gyro(raw_gyro, a);
        (a, w)
    }

    /// The gyroscope bias implied by an averaged stationary measurement.
    ///
    /// At rest the true angular rate is zero, so
    ///
    /// ```text
    /// b_g = w_raw − A · C_gyro_i · a_ideal
    /// ```
    ///
    /// Pass the mean raw accelerometer and gyroscope readings of a stationary
    /// stretch. The result goes straight into [`ImuIntrinsics::gyro_bias`].
    /// Worth doing at start-up: bias moves with temperature and between power
    /// cycles, scale and misalignment do not.
    pub fn gyro_bias_from_static(
        &self,
        mean_raw_accel: [f64; 3],
        mean_raw_gyro: [f64; 3],
    ) -> [f64; 3] {
        let a = na::Vector3::from(self.correct_accel(mean_raw_accel));
        let b = na::Vector3::from(mean_raw_gyro) - self.gyro_accel_coupling * a;
        unvec3(&b)
    }

    /// Correct an [`ImuMsg`] in place.
    pub fn correct_msg_in_place(&self, imu: &mut ImuMsg) {
        let (a, w) = self.correct(imu.linear_acceleration, imu.angular_velocity);
        imu.linear_acceleration = a;
        imu.angular_velocity = w;
    }

    /// Correct an [`ImuMsg`], returning the corrected message.
    pub fn correct_msg(&self, mut imu: ImuMsg) -> ImuMsg {
        self.correct_msg_in_place(&mut imu);
        imu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use conv::{unmat3, unvec3};

    const EPS: f64 = 1e-12;

    /// A small rotation, as `C_gyro_i`.
    fn small_rotation() -> na::Matrix3<f64> {
        na::Rotation3::from_euler_angles(0.001, -0.002, 0.0015)
            .matrix()
            .to_owned()
    }

    fn assert_close(a: [f64; 3], b: [f64; 3], eps: f64) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < eps, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn identity_passes_through() {
        let c = ImuIntrinsics::identity().corrector().unwrap();
        let (a, g) = c.correct([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]);
        assert_close(a, [1.0, 2.0, 3.0], EPS);
        assert_close(g, [0.1, 0.2, 0.3], EPS);
    }

    #[test]
    fn correction_inverts_the_kalibr_forward_model() {
        let accel_m = na::Matrix3::new(1.01, 0.0, 0.0, 0.004, 0.99, 0.0, -0.002, 0.003, 1.02);
        let accel_bias = na::Vector3::new(0.05, -0.1, 0.2);
        let gyro_m = na::Matrix3::new(0.98, 0.0, 0.0, 0.003, 1.01, 0.0, -0.001, 0.002, 0.99);
        let gyro_bias = na::Vector3::new(0.01, -0.02, 0.005);
        let gyro_a = na::Matrix3::new(
            1e-3, 2e-4, -3e-4, //
            -5e-4, 7e-4, 1e-4, //
            2e-4, -1e-4, 6e-4,
        );
        let c_gyro_i = small_rotation();

        let mut intr = ImuIntrinsics::identity();
        intr.accel_m = unmat3(&accel_m);
        intr.accel_bias = unvec3(&accel_bias);
        intr.gyro_m = unmat3(&gyro_m);
        intr.gyro_bias = unvec3(&gyro_bias);
        intr.gyro_a = unmat3(&gyro_a);
        intr.c_gyro_i = unmat3(&c_gyro_i);

        let a_ideal = na::Vector3::new(0.3, -1.2, 9.7);
        let w_ideal = na::Vector3::new(0.4, -0.15, 0.9);

        // Forward model, exactly as Kalibr writes it in IccSensors.py.
        let a_raw = accel_m * a_ideal + accel_bias;
        let w_raw = gyro_m * (c_gyro_i * w_ideal) + gyro_a * (c_gyro_i * a_ideal) + gyro_bias;

        let c = intr.corrector().unwrap();
        let (a, w) = c.correct(a_raw.into(), w_raw.into());

        assert_close(a, a_ideal.into(), 1e-12);
        assert_close(w, w_ideal.into(), 1e-12);
    }

    #[test]
    fn static_gyro_bias_is_recovered() {
        let accel_m = na::Matrix3::new(1.01, 0.0, 0.0, 0.004, 0.99, 0.0, -0.002, 0.003, 1.02);
        let accel_bias = na::Vector3::new(0.05, -0.1, 0.2);
        let gyro_a = na::Matrix3::new(
            1e-3, 2e-4, -3e-4, //
            -5e-4, 7e-4, 1e-4, //
            2e-4, -1e-4, 6e-4,
        );
        let c_gyro_i = small_rotation();
        let true_bias = na::Vector3::new(0.011, -0.023, 0.006);

        let mut intr = ImuIntrinsics::identity();
        intr.accel_m = unmat3(&accel_m);
        intr.accel_bias = unvec3(&accel_bias);
        intr.gyro_a = unmat3(&gyro_a);
        intr.c_gyro_i = unmat3(&c_gyro_i);

        // A stationary sensor in some arbitrary attitude.
        let a_ideal = na::Vector3::new(1.7, -3.1, 9.15);
        let a_raw = accel_m * a_ideal + accel_bias;
        let w_raw = gyro_a * (c_gyro_i * a_ideal) + true_bias;

        // The stored bias must not influence the estimate.
        intr.gyro_bias = [99.0, -99.0, 99.0];
        let c = intr.corrector().unwrap();
        let estimated = c.gyro_bias_from_static(a_raw.into(), w_raw.into());
        assert_close(estimated, true_bias.into(), 1e-12);
    }

    #[test]
    fn singular_matrix_is_rejected() {
        let mut intr = ImuIntrinsics::identity();
        intr.accel_m = ZERO3;
        assert!(intr.corrector().is_err());
    }

    #[test]
    fn noise_discretisation() {
        let noise = ImuNoise {
            accel_noise_density: 1.86e-3,
            accel_random_walk: 4.33e-4,
            gyro_noise_density: 1.87e-4,
            gyro_random_walk: 2.66e-5,
            update_rate: 200.0,
        };
        let dt: f64 = 1.0 / 200.0;
        assert!((noise.accel_noise_discrete() - 1.86e-3 / dt.sqrt()).abs() < 1e-15);
        assert!((noise.gyro_noise_discrete() - 1.87e-4 / dt.sqrt()).abs() < 1e-15);
        assert!((noise.accel_bias_discrete() - 4.33e-4 * dt.sqrt()).abs() < 1e-15);
        assert!((noise.gyro_bias_discrete() - 2.66e-5 * dt.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn msg_correction() {
        let mut intr = ImuIntrinsics::identity();
        intr.accel_bias = [1.0, 1.0, 1.0];
        intr.gyro_bias = [0.5, 0.5, 0.5];
        let c = intr.corrector().unwrap();

        let corrected = c.correct_msg(ImuMsg::new(&[2.0, 3.0, 4.0], &[1.5, 2.5, 3.5]));
        assert_close(corrected.linear_acceleration, [1.0, 2.0, 3.0], EPS);
        assert_close(corrected.angular_velocity, [1.0, 2.0, 3.0], EPS);
    }
}
