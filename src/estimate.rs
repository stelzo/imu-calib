//! Estimating the deterministic IMU intrinsics from static orientations.
//!
//! # Accelerometer
//!
//! [`estimate_accel_intrinsics`] fits `M_a⁻¹` and `b_a` over all poses with
//! Levenberg-Marquardt. `M_a⁻¹` is parameterised as lower-triangular, which
//! fixes the otherwise free rotation of the sensor triad.
//!
//! Needs at least 9 orientations for the full model. 12 to 20 well-spread ones
//! is a good target. With fewer, [`AccelParametrization::ScaleBias`] drops the
//! misalignment terms and needs only 6.
//!
//! The gyroscope bias is the mean reading while stationary.
//!
//! # Entry points
//!
//! [`GuidedCalibration`] runs the whole thing as a state machine for a live
//! stream. [`calibrate_recording`] does the same over a finished recording,
//! finding the static stretches itself.

use crate::{ImuIntrinsics, ImuModel, ImuNoise};
use nalgebra as na;

use crate::conv::{dist3, norm3, unmat3, unvec3, vec3, Mat3, Vec3};

/// Standard gravity \[m/s²\].
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// The averaged IMU reading of one static orientation.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticPose {
    /// Mean raw accelerometer reading \[m/s²\].
    pub accel_mean: Vec3,
    /// Mean raw gyroscope reading \[rad/s\].
    pub gyro_mean: Vec3,
    /// Per-axis standard deviation of the accelerometer during the pose.
    pub accel_std: Vec3,
    /// Per-axis standard deviation of the gyroscope during the pose.
    pub gyro_std: Vec3,
    /// Number of samples averaged.
    pub samples: usize,
    /// Timestamp of the first sample \[s\].
    pub start_time: f64,
    /// Timestamp of the last sample \[s\].
    pub end_time: f64,
}

impl StaticPose {
    /// Average a slice of samples into a pose.
    ///
    /// `samples` must be `(timestamp, accel, gyro)` triples and non-empty.
    pub fn from_samples(samples: &[Sample]) -> anyhow::Result<Self> {
        if samples.is_empty() {
            return Err(anyhow::anyhow!("cannot build a static pose from 0 samples"));
        }
        let n = samples.len() as f64;
        let mut accel_mean = na::Vector3::zeros();
        let mut gyro_mean = na::Vector3::zeros();
        for (_, a, g) in samples {
            accel_mean += na::Vector3::from(*a);
            gyro_mean += na::Vector3::from(*g);
        }
        accel_mean /= n;
        gyro_mean /= n;

        let mut accel_var = na::Vector3::zeros();
        let mut gyro_var = na::Vector3::zeros();
        for (_, a, g) in samples {
            let da = na::Vector3::from(*a) - accel_mean;
            let dg = na::Vector3::from(*g) - gyro_mean;
            accel_var += da.component_mul(&da);
            gyro_var += dg.component_mul(&dg);
        }
        let denom = (n - 1.0).max(1.0);
        accel_var /= denom;
        gyro_var /= denom;

        Ok(Self {
            accel_mean: unvec3(&accel_mean),
            gyro_mean: unvec3(&gyro_mean),
            accel_std: unvec3(&accel_var.map(f64::sqrt)),
            gyro_std: unvec3(&gyro_var.map(f64::sqrt)),
            samples: samples.len(),
            start_time: samples[0].0,
            end_time: samples[samples.len() - 1].0,
        })
    }
}


/// How many free parameters the accelerometer model has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccelParametrization {
    /// Full lower-triangular scale and misalignment plus bias — 9 parameters,
    /// needs at least 9 distinct orientations. This is what Kalibr uses.
    #[default]
    ScaleMisalignmentBias,
    /// Per-axis scale plus bias — 6 parameters, needs at least 6 orientations.
    /// Use this when only the classic six axis-aligned positions are available.
    ScaleBias,
}

impl AccelParametrization {
    fn num_parameters(&self) -> usize {
        match self {
            AccelParametrization::ScaleMisalignmentBias => 9,
            AccelParametrization::ScaleBias => 6,
        }
    }

    fn min_poses(&self) -> usize {
        self.num_parameters()
    }
}

/// Result of [`estimate_accel_intrinsics`].
#[derive(Debug, Clone)]
pub struct AccelEstimate {
    /// Kalibr's `M_a`, lower-triangular, mapping ideal → raw.
    pub accel_m: Mat3,
    /// Accelerometer bias \[m/s²\].
    pub accel_bias: Vec3,
    /// RMS of `‖corrected‖ − g` over all poses \[m/s²\]. A good calibration
    /// lands well below the sensor's own noise floor.
    pub rms_residual: f64,
    /// Largest single-pose residual \[m/s²\].
    pub max_residual: f64,
    /// Number of poses used.
    pub poses: usize,
}

/// Estimate `M_a` and `b_a` from static orientations.
///
/// `gravity` is the local gravity magnitude; use [`STANDARD_GRAVITY`] unless a
/// better local value is known.
pub fn estimate_accel_intrinsics(
    poses: &[StaticPose],
    gravity: f64,
    parametrization: AccelParametrization,
) -> anyhow::Result<AccelEstimate> {
    let min = parametrization.min_poses();
    if poses.len() < min {
        return Err(anyhow::anyhow!(
            "{:?} needs at least {min} static orientations, got {}",
            parametrization,
            poses.len()
        ));
    }
    if gravity <= 0.0 || !gravity.is_finite() {
        return Err(anyhow::anyhow!("gravity must be a positive, finite value"));
    }

    let measurements: Vec<na::Vector3<f64>> = poses.iter().map(|p| vec3(&p.accel_mean)).collect();

    // Parameter layout, with T = M_a⁻¹ lower triangular:
    //   ScaleMisalignmentBias: [t00, t10, t11, t20, t21, t22, bx, by, bz]
    //   ScaleBias:             [t00, t11, t22, bx, by, bz]
    let x0 = match parametrization {
        AccelParametrization::ScaleMisalignmentBias => {
            na::DVector::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        }
        AccelParametrization::ScaleBias => {
            na::DVector::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        }
    };

    let unpack = move |x: &na::DVector<f64>| -> (na::Matrix3<f64>, na::Vector3<f64>) {
        match parametrization {
            AccelParametrization::ScaleMisalignmentBias => (
                na::Matrix3::new(x[0], 0.0, 0.0, x[1], x[2], 0.0, x[3], x[4], x[5]),
                na::Vector3::new(x[6], x[7], x[8]),
            ),
            AccelParametrization::ScaleBias => (
                na::Matrix3::new(x[0], 0.0, 0.0, 0.0, x[1], 0.0, 0.0, 0.0, x[2]),
                na::Vector3::new(x[3], x[4], x[5]),
            ),
        }
    };

    let residual = |x: &na::DVector<f64>| -> na::DVector<f64> {
        let (t, b) = unpack(x);
        na::DVector::from_iterator(
            measurements.len(),
            measurements.iter().map(|m| (t * (m - b)).norm() - gravity),
        )
    };

    let solution = levenberg_marquardt(x0, &residual, LmConfig::default())?;
    let (t, accel_bias) = unpack(&solution.x);

    let accel_m = t.try_inverse().ok_or_else(|| {
        anyhow::anyhow!("the estimated accelerometer matrix is singular; check the input poses")
    })?;

    let r = residual(&solution.x);
    let rms_residual = (r.dot(&r) / r.len() as f64).sqrt();
    let max_residual = r.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));

    Ok(AccelEstimate {
        accel_m: unmat3(&accel_m),
        accel_bias: unvec3(&accel_bias),
        rms_residual,
        max_residual,
        poses: poses.len(),
    })
}

/// Estimate the gyroscope bias as the mean reading over all static poses,
/// weighted by the number of samples in each.
pub fn estimate_gyro_bias(poses: &[StaticPose]) -> anyhow::Result<Vec3> {
    if poses.is_empty() {
        return Err(anyhow::anyhow!("need at least one static pose"));
    }
    let mut sum = na::Vector3::zeros();
    let mut n = 0usize;
    for p in poses {
        sum += vec3(&p.gyro_mean) * p.samples as f64;
        n += p.samples;
    }
    Ok(unvec3(&(sum / n as f64)))
}

// ---------------------------------------------------------------------------
// Gyroscope intrinsics
// ---------------------------------------------------------------------------

/// The angular motion between two consecutive static poses.
#[derive(Debug, Clone)]
pub struct MotionSegment {
    /// Gravity direction (unit vector) in the sensor frame at the start,
    /// measured by the already calibrated accelerometer.
    pub gravity_start: Vec3,
    /// Gravity direction in the sensor frame at the end.
    pub gravity_end: Vec3,
    /// Raw gyroscope samples between the two poses as `(dt, gyro_raw)`, where
    /// `dt` is the interval since the previous sample \[s\].
    pub samples: Vec<(f64, Vec3)>,
}

impl MotionSegment {
    /// Total rotation angle swept, using the raw readings \[rad\]. Segments
    /// that barely rotate carry almost no information.
    pub fn swept_angle(&self) -> f64 {
        self.samples.iter().map(|(dt, w)| norm3(w) * dt).sum()
    }
}

/// Result of [`estimate_gyro_intrinsics`].
#[derive(Debug, Clone)]
pub struct GyroEstimate {
    /// Kalibr's `M_g`, lower-triangular.
    pub gyro_m: Mat3,
    /// Kalibr's `C_gyro_i`, the rotation from the accelerometer frame to the
    /// gyroscope frame.
    pub c_gyro_i: Mat3,
    /// RMS angle between the predicted and the measured gravity direction at
    /// the end of each segment \[rad\].
    pub rms_residual: f64,
    /// Largest single-segment angular error \[rad\].
    pub max_residual: f64,
    /// Number of motion segments used.
    pub segments: usize,
}

/// Estimate `M_g` and `C_gyro_i` from the rotations between static poses.
///
/// `gyro_bias` must already be known (see [`estimate_gyro_bias`]).
///
/// Needs at least 3 segments to be determined at all; 8 or more with rotations
/// about clearly different axes gives a usable result. Segments that sweep
/// less than a few degrees are ignored.
pub fn estimate_gyro_intrinsics(
    segments: &[MotionSegment],
    gyro_bias: Vec3,
) -> anyhow::Result<GyroEstimate> {
    /// Segments sweeping less than this contribute noise rather than signal.
    const MIN_SWEEP: f64 = 10.0_f64.to_radians();

    let gyro_bias = vec3(&gyro_bias);

    let usable: Vec<&MotionSegment> = segments
        .iter()
        .filter(|s| !s.samples.is_empty() && s.swept_angle() > MIN_SWEEP)
        .collect();

    if usable.len() < 3 {
        return Err(anyhow::anyhow!(
            "gyroscope scale/misalignment needs at least 3 motion segments of more than \
             {:.0}°, got {} of {}",
            MIN_SWEEP.to_degrees(),
            usable.len(),
            segments.len()
        ));
    }

    // T_g is the full 3x3 correction matrix: w_ideal = T_g (w_raw − b_g).
    let x0 = na::DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    let unpack = |x: &na::DVector<f64>| {
        na::Matrix3::new(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8])
    };

    let residual = |x: &na::DVector<f64>| -> na::DVector<f64> {
        let t = unpack(x);
        let mut out = na::DVector::zeros(usable.len() * 3);
        for (i, seg) in usable.iter().enumerate() {
            let predicted = propagate_gravity(seg, &t, &gyro_bias);
            let diff = predicted - vec3(&seg.gravity_end);
            out[i * 3] = diff[0];
            out[i * 3 + 1] = diff[1];
            out[i * 3 + 2] = diff[2];
        }
        out
    };

    let solution = levenberg_marquardt(x0, &residual, LmConfig::default())?;
    let t_g = unpack(&solution.x);

    // Kalibr stores the forward matrices: w_raw − b_g = M_g · C_gyro_i · w_ideal,
    // so M_g · C_gyro_i = T_g⁻¹. Split that product with an LQ decomposition.
    let forward = t_g.try_inverse().ok_or_else(|| {
        anyhow::anyhow!("the estimated gyroscope matrix is singular; check the input segments")
    })?;
    let (gyro_m, c_gyro_i) = split_lower_triangular_rotation(&forward)?;

    // Report the residual as an angle, which is easier to judge than a chord.
    let mut sum_sq = 0.0;
    let mut max = 0.0f64;
    for seg in &usable {
        let predicted = propagate_gravity(seg, &t_g, &gyro_bias);
        let angle = predicted
            .normalize()
            .dot(&vec3(&seg.gravity_end).normalize())
            .clamp(-1.0, 1.0)
            .acos();
        sum_sq += angle * angle;
        max = max.max(angle);
    }

    Ok(GyroEstimate {
        gyro_m: unmat3(&gyro_m),
        c_gyro_i: unmat3(&c_gyro_i),
        rms_residual: (sum_sq / usable.len() as f64).sqrt(),
        max_residual: max,
        segments: usable.len(),
    })
}

/// Rotate the start gravity direction through the segment's angular rates.
///
/// A vector that is fixed in the world rotates as `dv/dt = −ω × v` when
/// expressed in the rotating sensor frame, so each step applies the rotation
/// `exp(−[ω]× dt)`.
fn propagate_gravity(
    seg: &MotionSegment,
    t_g: &na::Matrix3<f64>,
    bias: &na::Vector3<f64>,
) -> na::Vector3<f64> {
    let mut v = vec3(&seg.gravity_start);
    for (dt, w_raw) in &seg.samples {
        let w = t_g * (vec3(w_raw) - bias);
        v = rotate_by_rotation_vector(&v, &(-w * *dt));
    }
    v
}

/// Rodrigues rotation of `v` by the rotation vector `phi`.
fn rotate_by_rotation_vector(v: &na::Vector3<f64>, phi: &na::Vector3<f64>) -> na::Vector3<f64> {
    let theta = phi.norm();
    if theta < 1e-12 {
        // First-order term is all that survives at this magnitude.
        return v + phi.cross(v);
    }
    let k = phi / theta;
    let (s, c) = theta.sin_cos();
    v * c + k.cross(v) * s + k * (k.dot(v)) * (1.0 - c)
}

/// Factor `X = L · R` into a lower-triangular `L` with positive diagonal and a
/// rotation `R`, which is exactly how Kalibr splits `M_g · C_gyro_i`.
fn split_lower_triangular_rotation(
    x: &na::Matrix3<f64>,
) -> anyhow::Result<(na::Matrix3<f64>, na::Matrix3<f64>)> {
    // Xᵀ = Rᵀ Lᵀ is a QR decomposition: Q = Rᵀ, upper-triangular = Lᵀ.
    let qr = x.transpose().qr();
    let mut q = qr.q();
    let mut r = qr.r();

    // QR is only unique up to the signs of the diagonal; force L's diagonal
    // (the scale factors) positive.
    for i in 0..3 {
        if r[(i, i)] < 0.0 {
            for j in 0..3 {
                r[(i, j)] = -r[(i, j)];
                q[(j, i)] = -q[(j, i)];
            }
        }
    }

    let lower = r.transpose();
    let rotation = q.transpose();

    let det = rotation.determinant();
    if det < 0.0 {
        return Err(anyhow::anyhow!(
            "the estimated gyroscope frame is a reflection (det = {det:.3}), which is not \
             physically meaningful; the motion segments are probably degenerate"
        ));
    }

    Ok((lower, rotation))
}

// ---------------------------------------------------------------------------
// Offline calibration of a finished recording
// ---------------------------------------------------------------------------

/// One IMU sample: `(timestamp [s], raw accelerometer [m/s²], raw gyroscope [rad/s])`.
pub type Sample = (f64, [f64; 3], [f64; 3]);

/// Configuration of [`calibrate_recording`].
#[derive(Debug, Clone, PartialEq)]
pub struct RecordingConfig {
    /// Local gravity magnitude \[m/s²\].
    pub gravity: f64,
    /// Length of the stationary stretch at the very start of the recording
    /// \[s\]. It sets the noise floor the motion detector compares against, so
    /// the recording has to begin with the sensor at rest.
    pub init_seconds: f64,
    /// Shortest stretch that counts as a usable static orientation \[s\].
    pub min_static_seconds: f64,
    /// Length of the sliding window used to classify still versus moving \[s\].
    pub stillness_window_seconds: f64,
    /// Multiple of the initial noise level above which motion is declared.
    pub threshold_scale: f64,
    /// Accelerometer model to fit.
    pub accel_parametrization: AccelParametrization,
    /// Also estimate gyroscope scale and misalignment from the rotations
    /// between the static stretches.
    pub estimate_gyro_intrinsics: bool,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            gravity: STANDARD_GRAVITY,
            init_seconds: 4.0,
            min_static_seconds: 1.0,
            stillness_window_seconds: 0.2,
            threshold_scale: 8.0,
            accel_parametrization: AccelParametrization::ScaleMisalignmentBias,
            estimate_gyro_intrinsics: true,
        }
    }
}

/// Calibrate from a finished recording, finding the static orientations itself.
///
/// This is the offline counterpart of [`GuidedCalibration`]: instead of
/// prompting a person through the poses, it segments the whole recording into
/// stretches where the sensor was at rest and uses every one of them.
///
/// The recording must start with the sensor stationary — that stretch sets
/// the noise floor everything else is compared against.
pub fn calibrate_recording(
    samples: &[Sample],
    cfg: &RecordingConfig,
) -> anyhow::Result<Calibration> {
    let rate = sample_rate(samples)
        .ok_or_else(|| anyhow::anyhow!("the recording has no usable timestamps"))?;

    let window = ((cfg.stillness_window_seconds * rate).round() as usize).max(2);
    let init_len = ((cfg.init_seconds * rate).round() as usize).max(window);
    let min_static = ((cfg.min_static_seconds * rate).round() as usize).max(window);

    if samples.len() < init_len {
        return Err(anyhow::anyhow!(
            "the recording is {:.1} s long, which is shorter than the {:.1} s stationary \
             period the motion detector needs to calibrate itself",
            samples.len() as f64 / rate,
            cfg.init_seconds
        ));
    }

    let reference = StaticPose::from_samples(&samples[..init_len])?;
    let thresholds = MotionThresholds::from_reference(&reference, cfg.threshold_scale);

    let intervals = static_intervals(samples, window, min_static, &thresholds);
    if intervals.is_empty() {
        return Err(anyhow::anyhow!(
            "found no stationary stretch of at least {:.1} s in the recording",
            cfg.min_static_seconds
        ));
    }

    let poses = intervals
        .iter()
        .map(|&(a, b)| StaticPose::from_samples(&samples[a..b]))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let accel = estimate_accel_intrinsics(&poses, cfg.gravity, cfg.accel_parametrization)?;
    let gyro_bias = estimate_gyro_bias(&poses)?;

    let mut intrinsics = ImuIntrinsics {
        model: ImuModel::ScaleMisalignment,
        accel_m: accel.accel_m,
        accel_bias: accel.accel_bias,
        gyro_bias,
        noise: ImuNoise {
            update_rate: rate,
            ..ImuNoise::default()
        },
        ..ImuIntrinsics::identity()
    };

    let (gyro, gyro_error) = if cfg.estimate_gyro_intrinsics {
        let corrector = intrinsics.corrector()?;
        let segments = motion_segments(samples, &intervals, &corrector);
        match estimate_gyro_intrinsics(&segments, gyro_bias) {
            Ok(g) => {
                intrinsics.gyro_m = g.gyro_m;
                intrinsics.c_gyro_i = g.c_gyro_i;
                (Some(g), None)
            }
            Err(e) => (None, Some(e.to_string())),
        }
    } else {
        (None, None)
    };

    let report = CalibrationReport {
        accel,
        gyro,
        gyro_error,
        poses: poses.len(),
        duration: samples[samples.len() - 1].0 - samples[0].0,
        update_rate: rate,
    };

    Ok(Calibration { intrinsics, report })
}

/// Mean sample rate over the whole recording \[Hz\].
fn sample_rate(samples: &[Sample]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let span = samples[samples.len() - 1].0 - samples[0].0;
    if span <= 0.0 || !span.is_finite() {
        return None;
    }
    Some((samples.len() - 1) as f64 / span)
}

/// How quiet a window has to be to count as stationary.
#[derive(Debug, Clone, Copy)]
struct MotionThresholds {
    accel: f64,
    gyro: f64,
    reference_gyro: Vec3,
}

impl MotionThresholds {
    fn from_reference(reference: &StaticPose, scale: f64) -> Self {
        Self {
            accel: norm3(&reference.accel_std).max(1e-4) * scale,
            gyro: norm3(&reference.gyro_std).max(1e-5) * scale,
            reference_gyro: reference.gyro_mean,
        }
    }

    fn is_still(&self, pose: &StaticPose) -> bool {
        norm3(&pose.accel_std) <= self.accel
            && norm3(&pose.gyro_std) <= self.gyro
            && dist3(&pose.gyro_mean, &self.reference_gyro) <= self.gyro
    }
}

/// Maximal `[start, end)` ranges over which the sensor was at rest.
fn static_intervals(
    samples: &[Sample],
    window: usize,
    min_static: usize,
    thresholds: &MotionThresholds,
) -> Vec<(usize, usize)> {
    let mut intervals = Vec::new();
    if samples.len() < window {
        return intervals;
    }

    // A window ending at `i` being quiet means samples [i - window + 1, i] are.
    let mut run_start: Option<usize> = None;
    for end in (window - 1)..samples.len() {
        let quiet = StaticPose::from_samples(&samples[end + 1 - window..=end])
            .map(|pose| thresholds.is_still(&pose))
            .unwrap_or(false);

        if quiet {
            run_start.get_or_insert(end + 1 - window);
        } else if let Some(start) = run_start.take() {
            push_interval(&mut intervals, start, end, min_static);
        }
    }
    if let Some(start) = run_start {
        push_interval(&mut intervals, start, samples.len(), min_static);
    }
    intervals
}

fn push_interval(intervals: &mut Vec<(usize, usize)>, start: usize, end: usize, min: usize) {
    if end.saturating_sub(start) >= min {
        intervals.push((start, end));
    }
}

/// Build the rotation between each pair of consecutive static stretches.
fn motion_segments(
    samples: &[Sample],
    intervals: &[(usize, usize)],
    corrector: &crate::ImuCorrector,
) -> Vec<MotionSegment> {
    let gravity_at = |range: (usize, usize)| -> na::Vector3<f64> {
        let mut sum = na::Vector3::zeros();
        for (_, a, _) in &samples[range.0..range.1] {
            sum += na::Vector3::from(corrector.correct_accel(*a));
        }
        (sum / (range.1 - range.0) as f64).normalize()
    };

    let mut segments = Vec::new();
    for pair in intervals.windows(2) {
        let end_prev = pair[0].1;
        let start_next = pair[1].0;
        if start_next <= end_prev || end_prev == 0 {
            continue;
        }

        let mut gyro = Vec::with_capacity(start_next - end_prev);
        for i in end_prev..start_next {
            let dt = samples[i].0 - samples[i - 1].0;
            if dt <= 0.0 || !dt.is_finite() {
                continue;
            }
            gyro.push((dt, samples[i].2));
        }

        segments.push(MotionSegment {
            gravity_start: unvec3(&gravity_at(pair[0])),
            gravity_end: unvec3(&gravity_at(pair[1])),
            samples: gyro,
        });
    }
    segments
}

/// Configuration of a [`GuidedCalibration`].
#[derive(Debug, Clone, PartialEq)]
pub struct GuidedConfig {
    /// Local gravity magnitude \[m/s²\].
    pub gravity: f64,
    /// How many static orientations to capture, including the initial one.
    pub num_poses: usize,
    /// Length of the initial stationary period \[s\]. It sets the stillness
    /// thresholds and gives a first gyro bias estimate, so do not skip it.
    pub init_seconds: f64,
    /// How many samples to average per static orientation.
    pub samples_per_pose: usize,
    /// Length of the sliding window used to decide whether the IMU is still.
    pub stillness_window: usize,
    /// Multiple of the initial noise level above which motion is declared.
    pub threshold_scale: f64,
    /// Accelerometer model to fit.
    pub accel_parametrization: AccelParametrization,
    /// Also estimate gyroscope scale and misalignment from the rotations
    /// between poses. When false, only the gyro bias is estimated.
    pub estimate_gyro_intrinsics: bool,
}

impl Default for GuidedConfig {
    fn default() -> Self {
        Self {
            gravity: STANDARD_GRAVITY,
            num_poses: 12,
            init_seconds: 4.0,
            samples_per_pose: 400,
            stillness_window: 40,
            threshold_scale: 8.0,
            accel_parametrization: AccelParametrization::ScaleMisalignmentBias,
            estimate_gyro_intrinsics: true,
        }
    }
}

/// What the session is currently waiting for.
///
/// Returned by [`GuidedCalibration::push`] so the caller can prompt the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Collecting the initial stationary period. Do not touch the IMU.
    Initialising {
        /// Samples collected so far.
        collected: usize,
        /// Samples needed.
        total: usize,
    },
    /// The previous pose is captured; pick the IMU up and turn it.
    WaitingForMotion {
        /// Index of the pose that will be captured next.
        next_pose: usize,
        /// Total number of poses requested.
        total: usize,
    },
    /// Motion was seen; put the IMU down and hold it still.
    WaitingForStillness {
        /// Index of the pose that will be captured next.
        next_pose: usize,
        /// Total number of poses requested.
        total: usize,
    },
    /// Averaging samples of the current orientation. Keep holding still.
    Collecting {
        /// Index of the pose being captured.
        pose: usize,
        /// Samples collected so far.
        collected: usize,
        /// Samples needed.
        total: usize,
    },
    /// A pose was just captured. Emitted for exactly one sample.
    PoseCaptured {
        /// Index of the pose that was captured.
        pose: usize,
        /// Total number of poses requested.
        total: usize,
    },
    /// All poses are captured; call [`GuidedCalibration::finish`].
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Initialising,
    WaitingForMotion,
    WaitingForStillness,
    Collecting,
    Done,
}

/// How well the estimated parameters fit the recorded data.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    /// Accelerometer fit.
    pub accel: AccelEstimate,
    /// Gyroscope fit, if scale and misalignment were estimated.
    pub gyro: Option<GyroEstimate>,
    /// Why the gyroscope scale and misalignment were not estimated, when they
    /// were asked for but could not be recovered. The bias is still valid.
    pub gyro_error: Option<String>,
    /// Static orientations captured.
    pub poses: usize,
    /// Length of the whole session \[s\].
    pub duration: f64,
    /// Estimated update rate \[Hz\].
    pub update_rate: f64,
}

impl CalibrationReport {
    /// A short human-readable summary, suitable for printing to a terminal.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "poses: {}   duration: {:.1} s   rate: {:.1} Hz\n",
            self.poses, self.duration, self.update_rate
        ));
        s.push_str(&format!(
            "accelerometer: rms {:.4} m/s^2, max {:.4} m/s^2\n",
            self.accel.rms_residual, self.accel.max_residual
        ));
        match (&self.gyro, &self.gyro_error) {
            (Some(g), _) => s.push_str(&format!(
                "gyroscope:     rms {:.3}°, max {:.3}° over {} segments\n",
                g.rms_residual.to_degrees(),
                g.max_residual.to_degrees(),
                g.segments
            )),
            (None, Some(why)) => s.push_str(&format!("gyroscope:     bias only — {why}\n")),
            (None, None) => s.push_str("gyroscope:     bias only\n"),
        }
        s
    }
}

/// The finished product of a [`GuidedCalibration`].
#[derive(Debug, Clone)]
pub struct Calibration {
    /// The estimated parameters, ready to be written as Kalibr YAML.
    pub intrinsics: ImuIntrinsics,
    /// How well they fit.
    pub report: CalibrationReport,
}

/// Guided multi-orientation calibration.
///
/// Feed every incoming IMU sample to [`push`](Self::push) and act on the
/// [`Status`] it returns. Once it reports [`Status::Done`], call
/// [`finish`](Self::finish).
///
/// ```no_run
/// use imu_calib::estimate::{GuidedCalibration, GuidedConfig, Status};
///
/// let mut session = GuidedCalibration::new(GuidedConfig::default());
/// // in the IMU callback:
/// match session.push(0.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]) {
///     Status::WaitingForMotion { next_pose, total } => {
///         println!("turn the IMU to a new orientation ({next_pose}/{total})");
///     }
///     Status::Done => {
///         let calibration = session.finish().unwrap();
///         println!("{}", calibration.report.summary());
///     }
///     _ => {}
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GuidedCalibration {
    cfg: GuidedConfig,
    phase: Phase,
    /// Every sample of the session.
    samples: Vec<Sample>,
    /// Index ranges (inclusive start, exclusive end) of the captured poses.
    intervals: Vec<(usize, usize)>,
    /// Index at which the current collection window started.
    collect_start: usize,
    /// Thresholds derived from the initial stationary period.
    accel_threshold: f64,
    gyro_threshold: f64,
    /// Gyro bias from the initial stationary period, used for motion detection.
    init_gyro_mean: Vec3,
    first_time: Option<f64>,
}

impl GuidedCalibration {
    /// Start a new session.
    pub fn new(cfg: GuidedConfig) -> Self {
        Self {
            cfg,
            phase: Phase::Initialising,
            samples: Vec::new(),
            intervals: Vec::new(),
            collect_start: 0,
            accel_threshold: f64::INFINITY,
            gyro_threshold: f64::INFINITY,
            init_gyro_mean: crate::ZERO_VEC3,
            first_time: None,
        }
    }

    /// Push one sample and get the current status back.
    pub fn push(&mut self, t: f64, accel: [f64; 3], gyro: [f64; 3]) -> Status {
        if self.phase == Phase::Done {
            return Status::Done;
        }
        if self.first_time.is_none() {
            self.first_time = Some(t);
        }
        self.samples.push((t, accel, gyro));

        match self.phase {
            Phase::Initialising => self.step_initialising(t),
            Phase::WaitingForMotion => self.step_waiting_for_motion(),
            Phase::WaitingForStillness => self.step_waiting_for_stillness(),
            Phase::Collecting => self.step_collecting(),
            Phase::Done => Status::Done,
        }
    }

    /// Convenience wrapper for [`ImuMsg`](crate::ImuMsg).
    pub fn push_msg(&mut self, t: f64, msg: &crate::ImuMsg) -> Status {
        self.push(t, msg.linear_acceleration, msg.angular_velocity)
    }

    /// Whether all requested poses have been captured.
    pub fn is_done(&self) -> bool {
        self.phase == Phase::Done
    }

    /// How many static orientations have been captured so far.
    pub fn captured_poses(&self) -> usize {
        self.intervals.len()
    }

    fn init_target_samples(&self) -> usize {
        // Before the rate is known, fall back to the configured window length.
        let rate = self.estimate_rate().unwrap_or(200.0);
        ((self.cfg.init_seconds * rate).round() as usize).max(self.cfg.stillness_window)
    }

    fn step_initialising(&mut self, _t: f64) -> Status {
        let target = self.init_target_samples();
        let collected = self.samples.len();
        if collected < target {
            return Status::Initialising {
                collected,
                total: target,
            };
        }

        // Derive the motion thresholds and the first pose from this window.
        let window = &self.samples[..];
        let pose = match StaticPose::from_samples(window) {
            Ok(p) => p,
            Err(_) => {
                return Status::Initialising {
                    collected,
                    total: target,
                }
            }
        };

        self.init_gyro_mean = pose.gyro_mean;
        let accel_noise = norm3(&pose.accel_std).max(1e-4);
        let gyro_noise = norm3(&pose.gyro_std).max(1e-5);
        self.accel_threshold = accel_noise * self.cfg.threshold_scale;
        self.gyro_threshold = gyro_noise * self.cfg.threshold_scale;

        self.intervals.push((0, self.samples.len()));
        self.phase = if self.intervals.len() >= self.cfg.num_poses {
            Phase::Done
        } else {
            Phase::WaitingForMotion
        };

        Status::PoseCaptured {
            pose: self.intervals.len(),
            total: self.cfg.num_poses,
        }
    }

    fn step_waiting_for_motion(&mut self) -> Status {
        if self.is_moving() {
            self.phase = Phase::WaitingForStillness;
            return Status::WaitingForStillness {
                next_pose: self.intervals.len() + 1,
                total: self.cfg.num_poses,
            };
        }
        Status::WaitingForMotion {
            next_pose: self.intervals.len() + 1,
            total: self.cfg.num_poses,
        }
    }

    fn step_waiting_for_stillness(&mut self) -> Status {
        if !self.is_moving() {
            self.phase = Phase::Collecting;
            self.collect_start = self.samples.len();
            return Status::Collecting {
                pose: self.intervals.len() + 1,
                collected: 0,
                total: self.cfg.samples_per_pose,
            };
        }
        Status::WaitingForStillness {
            next_pose: self.intervals.len() + 1,
            total: self.cfg.num_poses,
        }
    }

    fn step_collecting(&mut self) -> Status {
        if self.is_moving() {
            // The IMU was disturbed; throw the partial window away.
            self.phase = Phase::WaitingForStillness;
            return Status::WaitingForStillness {
                next_pose: self.intervals.len() + 1,
                total: self.cfg.num_poses,
            };
        }

        let collected = self.samples.len() - self.collect_start;
        if collected < self.cfg.samples_per_pose {
            return Status::Collecting {
                pose: self.intervals.len() + 1,
                collected,
                total: self.cfg.samples_per_pose,
            };
        }

        self.intervals
            .push((self.collect_start, self.samples.len()));
        self.phase = if self.intervals.len() >= self.cfg.num_poses {
            Phase::Done
        } else {
            Phase::WaitingForMotion
        };

        Status::PoseCaptured {
            pose: self.intervals.len(),
            total: self.cfg.num_poses,
        }
    }

    /// Decide from the most recent window whether the IMU is being moved.
    fn is_moving(&self) -> bool {
        let n = self.cfg.stillness_window.max(2);
        if self.samples.len() < n {
            return true;
        }
        let window = &self.samples[self.samples.len() - n..];
        let pose = match StaticPose::from_samples(window) {
            Ok(p) => p,
            Err(_) => return true,
        };
        let gyro_offset = dist3(&pose.gyro_mean, &self.init_gyro_mean);
        norm3(&pose.accel_std) > self.accel_threshold
            || norm3(&pose.gyro_std) > self.gyro_threshold
            || gyro_offset > self.gyro_threshold
    }

    fn estimate_rate(&self) -> Option<f64> {
        let n = self.samples.len();
        if n < 2 {
            return None;
        }
        let span = self.samples[n - 1].0 - self.samples[0].0;
        if span <= 0.0 {
            return None;
        }
        Some((n - 1) as f64 / span)
    }

    /// The averaged static poses captured so far.
    pub fn poses(&self) -> anyhow::Result<Vec<StaticPose>> {
        self.intervals
            .iter()
            .map(|&(a, b)| StaticPose::from_samples(&self.samples[a..b]))
            .collect()
    }

    /// Build the motion segments between consecutive poses.
    ///
    /// `corrector` supplies the gravity directions, so the accelerometer must
    /// already be calibrated when this is called.
    fn motion_segments(&self, corrector: &crate::ImuCorrector) -> Vec<MotionSegment> {
        motion_segments(&self.samples, &self.intervals, corrector)
    }

    /// Run the estimators and assemble the calibration.
    ///
    /// Can be called before the session is done, as long as enough poses have
    /// been captured for the chosen accelerometer model.
    pub fn finish(&self) -> anyhow::Result<Calibration> {
        let poses = self.poses()?;
        let accel =
            estimate_accel_intrinsics(&poses, self.cfg.gravity, self.cfg.accel_parametrization)?;
        let gyro_bias = estimate_gyro_bias(&poses)?;

        let mut intrinsics = ImuIntrinsics {
            model: ImuModel::ScaleMisalignment,
            accel_m: accel.accel_m,
            accel_bias: accel.accel_bias,
            gyro_bias,
            noise: ImuNoise {
                update_rate: self.estimate_rate().unwrap_or(200.0),
                ..ImuNoise::default()
            },
            ..ImuIntrinsics::identity()
        };

        let (gyro, gyro_error) = if self.cfg.estimate_gyro_intrinsics {
            let corrector = intrinsics.corrector()?;
            let segments = self.motion_segments(&corrector);
            match estimate_gyro_intrinsics(&segments, gyro_bias) {
                Ok(g) => {
                    intrinsics.gyro_m = g.gyro_m;
                    intrinsics.c_gyro_i = g.c_gyro_i;
                    (Some(g), None)
                }
                // Bias-only is still a useful result, so report why the rest
                // was dropped rather than throwing the whole session away.
                Err(e) => (None, Some(e.to_string())),
            }
        } else {
            (None, None)
        };

        let report = CalibrationReport {
            accel,
            gyro,
            gyro_error,
            poses: poses.len(),
            duration: self.duration(),
            update_rate: intrinsics.noise.update_rate,
        };

        Ok(Calibration { intrinsics, report })
    }

    /// Length of the session so far \[s\].
    pub fn duration(&self) -> f64 {
        match (self.first_time, self.samples.last()) {
            (Some(first), Some(last)) => last.0 - first,
            _ => 0.0,
        }
    }
}

/// Tuning of the [`levenberg_marquardt`] solver.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LmConfig {
    max_iterations: usize,
    /// Stop when the cost improves by less than this fraction.
    relative_tolerance: f64,
    /// Stop when the parameter step is smaller than this.
    step_tolerance: f64,
    initial_lambda: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            relative_tolerance: 1e-12,
            step_tolerance: 1e-12,
            initial_lambda: 1e-3,
        }
    }
}

struct LmSolution {
    x: na::DVector<f64>,
    #[allow(dead_code)]
    cost: f64,
    #[allow(dead_code)]
    iterations: usize,
}

/// Small dense Levenberg-Marquardt with a numerically differentiated Jacobian.
///
/// The problems here have at most 9 parameters and a few dozen residuals, so a
/// finite-difference Jacobian is both fast enough and far less error-prone than
/// hand-derived analytic ones.
fn levenberg_marquardt<F>(
    mut x: na::DVector<f64>,
    residual: &F,
    cfg: LmConfig,
) -> anyhow::Result<LmSolution>
where
    F: Fn(&na::DVector<f64>) -> na::DVector<f64>,
{
    let cost_of = |r: &na::DVector<f64>| r.dot(r);

    let mut r = residual(&x);
    let mut cost = cost_of(&r);
    let mut lambda = cfg.initial_lambda;
    let n = x.len();

    for iteration in 0..cfg.max_iterations {
        let j = numeric_jacobian(residual, &x, &r);
        let jt = j.transpose();
        let jtj = &jt * &j;
        let jtr = &jt * &r;

        let mut accepted = false;
        // Try progressively more damping until the step actually helps.
        for _ in 0..30 {
            let mut damped = jtj.clone();
            for i in 0..n {
                let d = jtj[(i, i)];
                damped[(i, i)] = d + lambda * if d > 0.0 { d } else { 1.0 };
            }

            let step = match damped.clone().lu().solve(&(-&jtr)) {
                Some(s) => s,
                None => {
                    lambda *= 10.0;
                    continue;
                }
            };

            let candidate = &x + &step;
            let candidate_r = residual(&candidate);
            let candidate_cost = cost_of(&candidate_r);

            if candidate_cost.is_finite() && candidate_cost < cost {
                let improvement = (cost - candidate_cost) / cost.max(f64::MIN_POSITIVE);
                let step_norm = step.norm();
                x = candidate;
                r = candidate_r;
                cost = candidate_cost;
                lambda = (lambda * 0.3).max(1e-12);
                accepted = true;

                if improvement < cfg.relative_tolerance || step_norm < cfg.step_tolerance {
                    return Ok(LmSolution {
                        x,
                        cost,
                        iterations: iteration + 1,
                    });
                }
                break;
            }
            lambda *= 10.0;
        }

        if !accepted {
            // No damping produced an improvement: this is the minimum we get.
            return Ok(LmSolution {
                x,
                cost,
                iterations: iteration + 1,
            });
        }
    }

    Ok(LmSolution {
        x,
        cost,
        iterations: cfg.max_iterations,
    })
}

/// Forward-difference Jacobian with a step scaled to each parameter.
fn numeric_jacobian<F>(
    residual: &F,
    x: &na::DVector<f64>,
    r0: &na::DVector<f64>,
) -> na::DMatrix<f64>
where
    F: Fn(&na::DVector<f64>) -> na::DVector<f64>,
{
    const BASE_STEP: f64 = 1e-7;

    let mut j = na::DMatrix::zeros(r0.len(), x.len());
    let mut perturbed = x.clone();
    for col in 0..x.len() {
        let h = BASE_STEP * x[col].abs().max(1.0);
        let original = perturbed[col];
        perturbed[col] = original + h;
        let r = residual(&perturbed);
        perturbed[col] = original;
        for row in 0..r0.len() {
            j[(row, col)] = (r[row] - r0[row]) / h;
        }
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::mat3;

    /// Frobenius distance between two matrices, for the tolerance assertions.
    fn m3_err(a: &Mat3, b: &Mat3) -> f64 {
        (mat3(a) - mat3(b)).norm()
    }

    /// Distance between two vectors.
    fn v3_err(a: &Vec3, b: &Vec3) -> f64 {
        dist3(a, b)
    }

    /// Twelve reasonably spread orientations of the sensor.
    fn orientations() -> Vec<na::Rotation3<f64>> {
        let mut out = Vec::new();
        for (r, p, y) in [
            (0.0, 0.0, 0.0),
            (std::f64::consts::FRAC_PI_2, 0.0, 0.0),
            (std::f64::consts::PI, 0.0, 0.0),
            (-std::f64::consts::FRAC_PI_2, 0.0, 0.0),
            (0.0, std::f64::consts::FRAC_PI_2, 0.0),
            (0.0, -std::f64::consts::FRAC_PI_2, 0.0),
            (0.6, 0.4, 0.2),
            (-0.7, 0.9, -0.3),
            (1.2, -0.8, 0.5),
            (2.0, 0.3, -1.1),
            (-1.4, -1.0, 0.8),
            (0.9, 1.3, 2.4),
        ] {
            out.push(na::Rotation3::from_euler_angles(r, p, y));
        }
        out
    }

    fn truth() -> ImuIntrinsics {
        let mut intr = ImuIntrinsics::identity();
        intr.accel_m = [
            [1.012, 0.0, 0.0],
            [0.004, 0.991, 0.0],
            [-0.006, 0.003, 1.021],
        ];
        intr.accel_bias = [0.08, -0.15, 0.21];
        intr.gyro_m = [
            [0.983, 0.0, 0.0],
            [0.005, 1.013, 0.0],
            [-0.004, 0.007, 0.994],
        ];
        intr.c_gyro_i = unmat3(
            &na::Rotation3::from_euler_angles(0.004, -0.003, 0.006)
                .matrix()
                .to_owned(),
        );
        intr.gyro_bias = [0.012, -0.021, 0.007];
        intr
    }

    /// Raw accelerometer reading for a sensor rotated by `r`.
    fn raw_accel(intr: &ImuIntrinsics, r: &na::Rotation3<f64>, gravity: f64) -> [f64; 3] {
        // Specific force in the sensor frame while at rest.
        let ideal = r.inverse() * na::Vector3::new(0.0, 0.0, gravity);
        unvec3(&(mat3(&intr.accel_m) * ideal + vec3(&intr.accel_bias)))
    }

    fn pose_at(intr: &ImuIntrinsics, r: &na::Rotation3<f64>, gravity: f64) -> StaticPose {
        let accel = raw_accel(intr, r, gravity);
        let gyro = intr.gyro_bias;
        let samples: Vec<(f64, [f64; 3], [f64; 3])> =
            (0..100).map(|i| (i as f64 * 0.005, accel, gyro)).collect();
        StaticPose::from_samples(&samples).unwrap()
    }

    #[test]
    fn accel_intrinsics_are_recovered() {
        let truth = truth();
        let g = STANDARD_GRAVITY;
        let poses: Vec<StaticPose> = orientations()
            .iter()
            .map(|r| pose_at(&truth, r, g))
            .collect();

        let est = estimate_accel_intrinsics(&poses, g, AccelParametrization::ScaleMisalignmentBias)
            .unwrap();

        assert!(
            m3_err(&est.accel_m, &truth.accel_m) < 1e-6,
            "M_a off by {:e}\n{:?}\n{:?}",
            m3_err(&est.accel_m, &truth.accel_m),
            est.accel_m,
            truth.accel_m
        );
        assert!(
            v3_err(&est.accel_bias, &truth.accel_bias) < 1e-6,
            "bias off by {:e}",
            v3_err(&est.accel_bias, &truth.accel_bias)
        );
        assert!(est.rms_residual < 1e-8);
    }

    #[test]
    fn scale_bias_model_needs_only_six_poses() {
        let mut truth = ImuIntrinsics::identity();
        truth.accel_m = [[1.02, 0.0, 0.0], [0.0, 0.98, 0.0], [0.0, 0.0, 1.01]];
        truth.accel_bias = [0.1, -0.2, 0.05];
        let g = STANDARD_GRAVITY;

        let poses: Vec<StaticPose> = orientations()
            .iter()
            .take(6)
            .map(|r| pose_at(&truth, r, g))
            .collect();

        let est = estimate_accel_intrinsics(&poses, g, AccelParametrization::ScaleBias).unwrap();
        assert!(m3_err(&est.accel_m, &truth.accel_m) < 1e-6);
        assert!(v3_err(&est.accel_bias, &truth.accel_bias) < 1e-6);
    }

    #[test]
    fn too_few_poses_is_an_error() {
        let truth = truth();
        let poses: Vec<StaticPose> = orientations()
            .iter()
            .take(4)
            .map(|r| pose_at(&truth, r, STANDARD_GRAVITY))
            .collect();
        assert!(estimate_accel_intrinsics(
            &poses,
            STANDARD_GRAVITY,
            AccelParametrization::ScaleMisalignmentBias
        )
        .is_err());
    }

    #[test]
    fn gyro_bias_is_the_static_mean() {
        let truth = truth();
        let poses: Vec<StaticPose> = orientations()
            .iter()
            .map(|r| pose_at(&truth, r, STANDARD_GRAVITY))
            .collect();
        let bias = estimate_gyro_bias(&poses).unwrap();
        assert!(v3_err(&bias, &truth.gyro_bias) < 1e-12);
    }

    #[test]
    fn lower_triangular_rotation_split_round_trips() {
        let lower = na::Matrix3::new(1.01, 0.0, 0.0, 0.02, 0.99, 0.0, -0.01, 0.03, 1.02);
        let rotation = na::Rotation3::from_euler_angles(0.01, -0.02, 0.03)
            .matrix()
            .to_owned();
        let product = lower * rotation;

        let (l, r) = split_lower_triangular_rotation(&product).unwrap();
        assert!((l - lower).norm() < 1e-12, "{l} vs {lower}");
        assert!((r - rotation).norm() < 1e-12, "{r} vs {rotation}");
        // Lower triangular with positive diagonal.
        assert!(l[(0, 1)].abs() < 1e-15 && l[(0, 2)].abs() < 1e-15 && l[(1, 2)].abs() < 1e-15);
        assert!(l[(0, 0)] > 0.0 && l[(1, 1)] > 0.0 && l[(2, 2)] > 0.0);
    }

    #[test]
    fn gyro_intrinsics_are_recovered_from_rotations() {
        let truth = truth();
        let g = STANDARD_GRAVITY;
        let dt = 1.0 / 400.0;
        let orientations = orientations();

        // Kalibr's forward gyro matrix, ignoring g-sensitivity.
        let forward = mat3(&truth.gyro_m) * mat3(&truth.c_gyro_i);

        let mut segments = Vec::new();
        for pair in orientations.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            // Rotate from `from` to `to` at constant angular velocity.
            let delta = from.inverse() * to;
            let axis_angle = delta.scaled_axis();
            let duration = 1.5;
            let steps = (duration / dt) as usize;
            let omega_body = axis_angle / duration;

            let mut samples = Vec::with_capacity(steps);
            for _ in 0..steps {
                let raw = forward * omega_body + vec3(&truth.gyro_bias);
                samples.push((dt, unvec3(&raw)));
            }

            let gravity_dir =
                |r: &na::Rotation3<f64>| (r.inverse() * na::Vector3::new(0.0, 0.0, g)).normalize();

            segments.push(MotionSegment {
                gravity_start: unvec3(&gravity_dir(&from)),
                gravity_end: unvec3(&gravity_dir(&to)),
                samples,
            });
        }

        let est = estimate_gyro_intrinsics(&segments, truth.gyro_bias).unwrap();

        assert!(
            m3_err(&est.gyro_m, &truth.gyro_m) < 1e-4,
            "M_g off by {:e}\n{:?}\n{:?}",
            m3_err(&est.gyro_m, &truth.gyro_m),
            est.gyro_m,
            truth.gyro_m
        );
        assert!(
            m3_err(&est.c_gyro_i, &truth.c_gyro_i) < 1e-4,
            "C_gyro_i off by {:e}",
            m3_err(&est.c_gyro_i, &truth.c_gyro_i)
        );
        assert!(est.rms_residual < 1e-4);
    }

    #[test]
    fn degenerate_segments_are_rejected() {
        let segments = vec![MotionSegment {
            gravity_start: [0.0, 0.0, 1.0],
            gravity_end: [0.0, 0.0, 1.0],
            samples: vec![(0.005, crate::ZERO_VEC3); 100],
        }];
        assert!(estimate_gyro_intrinsics(&segments, crate::ZERO_VEC3).is_err());
    }

    /// Synthesize a full recording: a stationary start, then alternating
    /// rotations and stationary holds, exactly what the offline path expects.
    fn synthetic_recording(truth: &ImuIntrinsics, rate: f64) -> Vec<Sample> {
        let g = STANDARD_GRAVITY;
        let dt = 1.0 / rate;
        let forward_gyro = mat3(&truth.gyro_m) * mat3(&truth.c_gyro_i);

        let mut seed = 0x5eed_1234u64;
        let mut jitter = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2e-3
        };

        let mut samples: Vec<Sample> = Vec::new();
        let mut t = 0.0;

        let emit = |samples: &mut Vec<Sample>,
                    t: &mut f64,
                    attitude: &na::Rotation3<f64>,
                    omega_body: na::Vector3<f64>,
                    jitter: &mut dyn FnMut() -> f64| {
            let a_ideal = attitude.inverse() * na::Vector3::new(0.0, 0.0, g);
            let a_raw = mat3(&truth.accel_m) * a_ideal + vec3(&truth.accel_bias);
            let w_raw = forward_gyro * omega_body
                + mat3(&truth.gyro_a) * (mat3(&truth.c_gyro_i) * a_ideal)
                + vec3(&truth.gyro_bias);
            samples.push((
                *t,
                [
                    a_raw[0] + jitter(),
                    a_raw[1] + jitter(),
                    a_raw[2] + jitter(),
                ],
                [
                    w_raw[0] + jitter(),
                    w_raw[1] + jitter(),
                    w_raw[2] + jitter(),
                ],
            ));
            *t += dt;
        };

        let orientations = orientations();

        // Four seconds stationary at the start, which sets the noise floor.
        for _ in 0..(4.0 * rate) as usize {
            emit(
                &mut samples,
                &mut t,
                &orientations[0],
                na::Vector3::zeros(),
                &mut jitter,
            );
        }

        for pair in orientations.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            let delta = from.inverse() * to;
            let axis_angle = delta.scaled_axis();
            let duration = 1.5;
            let steps = (duration * rate) as usize;
            let omega_body = axis_angle / duration;

            for step in 0..steps {
                let fraction = step as f64 / steps as f64;
                let attitude = from * na::Rotation3::from_scaled_axis(axis_angle * fraction);
                emit(&mut samples, &mut t, &attitude, omega_body, &mut jitter);
            }
            // Two seconds stationary in the new attitude.
            for _ in 0..(2.0 * rate) as usize {
                emit(&mut samples, &mut t, &to, na::Vector3::zeros(), &mut jitter);
            }
        }

        samples
    }

    #[test]
    fn offline_recording_finds_every_pose_and_recovers_the_parameters() {
        let truth = truth();
        let samples = synthetic_recording(&truth, 200.0);

        let calibration = calibrate_recording(&samples, &RecordingConfig::default()).unwrap();

        assert_eq!(
            calibration.report.poses,
            orientations().len(),
            "expected one static stretch per orientation, got {}",
            calibration.report.poses
        );

        let est = &calibration.intrinsics;
        assert!(
            m3_err(&est.accel_m, &truth.accel_m) < 5e-3,
            "M_a off by {:e}\n{:?}\n{:?}",
            m3_err(&est.accel_m, &truth.accel_m),
            est.accel_m,
            truth.accel_m
        );
        assert!(
            v3_err(&est.accel_bias, &truth.accel_bias) < 5e-3,
            "accel bias off by {:e}",
            v3_err(&est.accel_bias, &truth.accel_bias)
        );
        assert!(
            v3_err(&est.gyro_bias, &truth.gyro_bias) < 5e-3,
            "gyro bias off by {:e}",
            v3_err(&est.gyro_bias, &truth.gyro_bias)
        );

        let gyro = calibration
            .report
            .gyro
            .as_ref()
            .unwrap_or_else(|| panic!("gyro fit skipped: {:?}", calibration.report.gyro_error));
        assert!(
            gyro.rms_residual.to_degrees() < 1.0,
            "gyro residual {:.3}°",
            gyro.rms_residual.to_degrees()
        );
        assert!(
            m3_err(&est.gyro_m, &truth.gyro_m) < 2e-2,
            "M_g off by {:e}\n{:?}\n{:?}",
            m3_err(&est.gyro_m, &truth.gyro_m),
            est.gyro_m,
            truth.gyro_m
        );
    }

    #[test]
    fn a_recording_that_never_settles_is_rejected() {
        let truth = truth();
        let mut samples = synthetic_recording(&truth, 200.0);
        // Keep only the rotations, so nothing is ever at rest.
        samples.truncate(4 * 200);
        let cfg = RecordingConfig {
            init_seconds: 30.0,
            ..RecordingConfig::default()
        };
        let err = calibrate_recording(&samples, &cfg).unwrap_err();
        assert!(err.to_string().contains("shorter than"), "{err}");
    }

    #[test]
    fn guided_session_runs_end_to_end() {
        let truth = truth();
        let g = STANDARD_GRAVITY;
        let dt = 1.0 / 200.0;
        let orientations = orientations();

        let cfg = GuidedConfig {
            gravity: g,
            num_poses: orientations.len(),
            init_seconds: 2.0,
            samples_per_pose: 200,
            stillness_window: 20,
            threshold_scale: 8.0,
            accel_parametrization: AccelParametrization::ScaleMisalignmentBias,
            estimate_gyro_intrinsics: false,
        };
        let mut session = GuidedCalibration::new(cfg);

        // A little deterministic jitter, otherwise the noise floor is exactly
        // zero and the stillness thresholds degenerate.
        let mut seed = 12345u64;
        let mut jitter = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2e-3
        };

        let mut t = 0.0;
        let push_static = |session: &mut GuidedCalibration,
                           r: &na::Rotation3<f64>,
                           t: &mut f64,
                           n: usize,
                           jitter: &mut dyn FnMut() -> f64| {
            let base = raw_accel(&truth, r, g);
            for _ in 0..n {
                let accel = [base[0] + jitter(), base[1] + jitter(), base[2] + jitter()];
                let gyro = [
                    truth.gyro_bias[0] + jitter(),
                    truth.gyro_bias[1] + jitter(),
                    truth.gyro_bias[2] + jitter(),
                ];
                session.push(*t, accel, gyro);
                *t += dt;
            }
        };

        // Initial stationary period plus first pose.
        push_static(&mut session, &orientations[0], &mut t, 600, &mut jitter);

        for r in orientations.iter().skip(1) {
            // Obvious motion so the detector leaves the "still" state.
            for _ in 0..60 {
                session.push(t, [0.0, 0.0, 0.0], [3.0, -2.0, 1.5]);
                t += dt;
            }
            push_static(&mut session, r, &mut t, 400, &mut jitter);
        }

        assert!(
            session.is_done(),
            "captured {} poses",
            session.captured_poses()
        );

        let calibration = session.finish().unwrap();
        let est = &calibration.intrinsics;
        assert!(
            m3_err(&est.accel_m, &truth.accel_m) < 5e-3,
            "M_a off by {:e}",
            m3_err(&est.accel_m, &truth.accel_m)
        );
        assert!(
            v3_err(&est.gyro_bias, &truth.gyro_bias) < 5e-3,
            "gyro bias off by {:e}",
            v3_err(&est.gyro_bias, &truth.gyro_bias)
        );
    }
}
