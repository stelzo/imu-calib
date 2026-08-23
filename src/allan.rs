//! Allan-variance estimation of the IMU noise parameters.
//!
//! Feed a long stationary recording into an [`AllanEstimator`] to get the four
//! numbers Kalibr expects in its input `imu0.yaml`:
//! `accelerometer_noise_density`, `accelerometer_random_walk`,
//! `gyroscope_noise_density` and `gyroscope_random_walk`.
//!
//! The estimator is streaming: memory is `O(number of cluster sizes)` and
//! independent of recording length.
//!
//! ```rust
//! use imu_calib::allan::AllanEstimator;
//!
//! let mut est = AllanEstimator::new(Default::default());
//! // ... in the IMU callback:
//! est.push(0.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]);
//! ```

use crate::ImuNoise;

/// Configuration of an [`AllanEstimator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllanConfig {
    /// Shortest cluster time τ \[s\]. Defaults to `0.1`.
    pub min_cluster_time: f64,
    /// Longest cluster time τ \[s\]. Should be at most a tenth of the total
    /// recording length. Defaults to `1000.0`.
    pub max_cluster_time: f64,
    /// Number of logarithmically spaced cluster times. Defaults to `100`.
    pub num_clusters: usize,
    /// Minimum number of cluster pairs a τ needs before its Allan deviation is
    /// considered trustworthy. Defaults to `10`.
    pub min_pairs: usize,
}

impl Default for AllanConfig {
    fn default() -> Self {
        Self {
            min_cluster_time: 0.1,
            max_cluster_time: 1000.0,
            num_clusters: 100,
            min_pairs: 10,
        }
    }
}

/// Running accumulators for a single cluster size and a single scalar channel.
#[derive(Debug, Clone, Default)]
struct ClusterAccumulator {
    /// Sum of the samples of the cluster currently being filled.
    sum: f64,
    /// How many samples the current cluster already holds.
    count: usize,
    /// Mean of the previous completed cluster, if there was one.
    prev_mean: Option<f64>,
    /// Σ (ȳ_{k+1} − ȳ_k)² over all completed neighbouring cluster pairs.
    sum_sq_diff: f64,
    /// Number of terms in `sum_sq_diff`.
    pairs: usize,
}

impl ClusterAccumulator {
    fn push(&mut self, value: f64, cluster_len: usize) {
        self.sum += value;
        self.count += 1;
        if self.count == cluster_len {
            let mean = self.sum / cluster_len as f64;
            if let Some(prev) = self.prev_mean {
                let d = mean - prev;
                self.sum_sq_diff += d * d;
                self.pairs += 1;
            }
            self.prev_mean = Some(mean);
            self.sum = 0.0;
            self.count = 0;
        }
    }

    /// Allan variance, or `None` if not enough clusters completed.
    fn variance(&self, min_pairs: usize) -> Option<f64> {
        if self.pairs < min_pairs {
            return None;
        }
        Some(self.sum_sq_diff / (2.0 * self.pairs as f64))
    }
}

/// One row of the Allan deviation table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllanPoint {
    /// Cluster time τ \[s\].
    pub tau: f64,
    /// Allan deviation of the accelerometer axes \[m/s²\].
    pub accel: [f64; 3],
    /// Allan deviation of the gyroscope axes \[rad/s\].
    pub gyro: [f64; 3],
    /// Number of cluster pairs this row was averaged over.
    pub pairs: usize,
}

/// The Allan deviation curves of one recording.
#[derive(Debug, Clone, PartialEq)]
pub struct AllanResult {
    /// One entry per usable cluster time, sorted by τ.
    pub points: Vec<AllanPoint>,
    /// Sample rate inferred from the timestamps \[Hz\].
    pub update_rate: f64,
    /// Length of the recording \[s\].
    pub duration: f64,
    /// Number of samples the estimate is based on.
    pub samples: usize,
}

/// A straight-line fit `σ(τ) = c · τ^slope` in log-log space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlopeFit {
    /// Value of the fitted line at the reference τ.
    pub value: f64,
    /// τ at which `value` was read off \[s\].
    pub reference_tau: f64,
    /// Number of Allan points that entered the fit.
    pub points: usize,
}

impl AllanResult {
    /// Noise parameters in Kalibr's convention, averaged over the three axes.
    ///
    /// White noise is fitted on the `−1/2` slope region and read at `τ = 1 s`;
    /// the bias random walk is fitted on the `+1/2` slope region and read at
    /// `τ = 3 s`.
    ///
    /// Fails if the curve does not cover enough of either region — usually a
    /// sign that the recording is too short.
    pub fn noise(&self) -> anyhow::Result<ImuNoise> {
        let accel_n = self.fit_axes(Sensor::Accel, -0.5, 1.0)?;
        let gyro_n = self.fit_axes(Sensor::Gyro, -0.5, 1.0)?;
        let accel_k = self.fit_axes(Sensor::Accel, 0.5, 3.0)?;
        let gyro_k = self.fit_axes(Sensor::Gyro, 0.5, 3.0)?;

        Ok(ImuNoise {
            accel_noise_density: accel_n.value,
            accel_random_walk: accel_k.value,
            gyro_noise_density: gyro_n.value,
            gyro_random_walk: gyro_k.value,
            update_rate: self.update_rate,
        })
    }

    /// Per-axis white-noise density, in case the axes differ noticeably.
    pub fn noise_density_per_axis(&self, sensor: Sensor) -> anyhow::Result<[f64; 3]> {
        let mut out = [0.0; 3];
        for (axis, slot) in out.iter_mut().enumerate() {
            *slot = self.fit_axis(sensor, axis, -0.5, 1.0)?.value;
        }
        Ok(out)
    }

    /// Per-axis bias random walk.
    pub fn random_walk_per_axis(&self, sensor: Sensor) -> anyhow::Result<[f64; 3]> {
        let mut out = [0.0; 3];
        for (axis, slot) in out.iter_mut().enumerate() {
            *slot = self.fit_axis(sensor, axis, 0.5, 3.0)?.value;
        }
        Ok(out)
    }

    /// Mean of the three per-axis fits.
    fn fit_axes(&self, sensor: Sensor, slope: f64, reference_tau: f64) -> anyhow::Result<SlopeFit> {
        let mut value = 0.0;
        let mut points = 0;
        for axis in 0..3 {
            let fit = self.fit_axis(sensor, axis, slope, reference_tau)?;
            value += fit.value;
            points += fit.points;
        }
        Ok(SlopeFit {
            value: value / 3.0,
            reference_tau,
            points,
        })
    }

    /// Fit `log σ = log c + slope · log τ` over the τ range where the measured
    /// local slope is closest to `slope`, then evaluate at `reference_tau`.
    fn fit_axis(
        &self,
        sensor: Sensor,
        axis: usize,
        slope: f64,
        reference_tau: f64,
    ) -> anyhow::Result<SlopeFit> {
        let selected = self.select_region(sensor, axis, slope);

        // A couple of adjacent points can line up at any slope by chance once
        // the long-tau end of the curve gets noisy, and a sensor with no
        // measurable random walk has no +1/2 region at all. Demanding a real
        // stretch of curve is what keeps that from being reported as a number.
        let span = match (selected.first(), selected.last()) {
            (Some((first, _)), Some((last, _))) if *first > 0.0 => last / first,
            _ => 0.0,
        };
        if selected.len() < MIN_REGION_POINTS || span < MIN_REGION_SPAN {
            return Err(anyhow::anyhow!(
                "the Allan deviation of the {} {} axis has no clear slope {slope:+} region \
                 ({} point(s) spanning {span:.1}x in tau, need {MIN_REGION_POINTS} spanning \
                 {MIN_REGION_SPAN}x). {}",
                sensor.name(),
                AXIS_NAMES[axis],
                selected.len(),
                if slope > 0.0 {
                    "Either the recording is too short to reach the bias random walk, or this \
                     sensor does not show one over the cluster times covered"
                } else {
                    "The recording is probably too short"
                }
            ));
        }

        // Least squares on the intercept only, since the slope is fixed by the
        // noise model: log c = mean(log σ − slope · log τ).
        let mut log_c = 0.0;
        for &(tau, dev) in &selected {
            log_c += dev.ln() - slope * tau.ln();
        }
        log_c /= selected.len() as f64;

        Ok(SlopeFit {
            value: (log_c + slope * reference_tau.ln()).exp(),
            reference_tau,
            points: selected.len(),
        })
    }

    /// Pick the longest contiguous run of points whose local log-log slope is
    /// within `SLOPE_TOLERANCE` of the target.
    ///
    /// For a rising target the search starts at the curve's minimum. On a real
    /// IMU that minimum is the bias-instability floor, and the random walk is
    /// by definition the part after it; without that restriction a dip and
    /// recovery in the white-noise region can pass for a +1/2 slope.
    fn select_region(&self, sensor: Sensor, axis: usize, target: f64) -> Vec<(f64, f64)> {
        const SLOPE_TOLERANCE: f64 = 0.25;
        /// Factor in tau the local slope is measured over.
        const SLOPE_BASELINE: f64 = 2.0;

        let dev = |p: &AllanPoint| match sensor {
            Sensor::Accel => p.accel[axis],
            Sensor::Gyro => p.gyro[axis],
        };

        let usable: Vec<(f64, f64)> = self
            .points
            .iter()
            .map(|p| (p.tau, dev(p)))
            .filter(|(tau, d)| *tau > 0.0 && *d > 0.0)
            .collect();

        let start = if target > 0.0 {
            usable
                .iter()
                .enumerate()
                .min_by(|(_, (_, a)), (_, (_, b))| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            0
        };
        let usable = &usable[start.min(usable.len())..];

        let mut best: Vec<(f64, f64)> = Vec::new();
        let mut current: Vec<(f64, f64)> = Vec::new();

        for (i, &(tau, dev)) in usable.iter().enumerate() {
            // Measure the slope against a point at least SLOPE_BASELINE times
            // further out rather than the neighbour. Adjacent Allan points are
            // barely apart in tau and share most of their data, so the slope
            // between them is dominated by estimator noise and a real region
            // gets chopped into runs too short to recognise.
            let Some(far) = usable[i + 1..]
                .iter()
                .find(|(other, _)| *other >= tau * SLOPE_BASELINE)
            else {
                break;
            };

            let local = (far.1.ln() - dev.ln()) / (far.0.ln() - tau.ln());
            if (local - target).abs() <= SLOPE_TOLERANCE {
                current.push((tau, dev));
            } else if current.len() > best.len() {
                best = std::mem::take(&mut current);
            } else {
                current.clear();
            }
        }
        if current.len() > best.len() {
            best = current;
        }
        best
    }

    /// The Allan deviation table as CSV, ready for gnuplot or a spreadsheet.
    ///
    /// Columns: `tau,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,pairs`.
    pub fn to_csv_string(&self) -> String {
        let mut out = String::from("tau,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,pairs\n");
        for p in &self.points {
            out.push_str(&format!(
                "{:.6},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{}\n",
                p.tau, p.accel[0], p.accel[1], p.accel[2], p.gyro[0], p.gyro[1], p.gyro[2], p.pairs
            ));
        }
        out
    }
}

/// Which of the two sensors an Allan curve belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensor {
    /// Accelerometer.
    Accel,
    /// Gyroscope.
    Gyro,
}

impl Sensor {
    fn name(&self) -> &'static str {
        match self {
            Sensor::Accel => "accelerometer",
            Sensor::Gyro => "gyroscope",
        }
    }
}

const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];

/// Points a slope region needs before it is believed.
const MIN_REGION_POINTS: usize = 4;

/// Factor in tau a slope region has to span before it is believed.
const MIN_REGION_SPAN: f64 = 1.5;

/// Streaming Allan-variance estimator.
///
/// Push every sample of a long stationary recording, then call
/// [`finish`](Self::finish). Memory use is constant, so a multi-hour recording
/// is fine.
#[derive(Debug, Clone)]
pub struct AllanEstimator {
    cfg: AllanConfig,
    /// Cluster length in samples, one entry per τ. Filled on the first sample,
    /// once the sample rate is known.
    cluster_lens: Vec<usize>,
    /// `[cluster][channel]`, channels are `ax, ay, az, gx, gy, gz`.
    accumulators: Vec<[ClusterAccumulator; 6]>,
    first_time: Option<f64>,
    last_time: f64,
    samples: usize,
    /// Sample rate, either supplied up front or inferred from the timestamps.
    rate: Option<f64>,
    /// Samples buffered while the rate is still being inferred.
    pending: Vec<(f64, [f64; 3], [f64; 3])>,
}

/// How many samples are used to infer the update rate when it is not given.
const RATE_ESTIMATION_SAMPLES: usize = 500;

impl AllanEstimator {
    /// Create an estimator that infers the update rate from the timestamps of
    /// the first few hundred samples.
    pub fn new(cfg: AllanConfig) -> Self {
        Self {
            cfg,
            cluster_lens: Vec::new(),
            accumulators: Vec::new(),
            first_time: None,
            last_time: 0.0,
            samples: 0,
            rate: None,
            pending: Vec::new(),
        }
    }

    /// Create an estimator with a known update rate \[Hz\].
    pub fn with_rate(cfg: AllanConfig, rate_hz: f64) -> anyhow::Result<Self> {
        if rate_hz <= 0.0 || !rate_hz.is_finite() {
            return Err(anyhow::anyhow!(
                "update rate must be a positive, finite value"
            ));
        }
        let mut est = Self::new(cfg);
        est.set_rate(rate_hz);
        Ok(est)
    }

    fn set_rate(&mut self, rate_hz: f64) {
        let mut lens: Vec<usize> = Vec::with_capacity(self.cfg.num_clusters);
        let log_min = self.cfg.min_cluster_time.ln();
        let log_max = self.cfg.max_cluster_time.ln();
        let n = self.cfg.num_clusters.max(2);
        for i in 0..n {
            let t = log_min + (log_max - log_min) * (i as f64) / ((n - 1) as f64);
            let len = (t.exp() * rate_hz).round() as usize;
            let len = len.max(1);
            if lens.last() != Some(&len) {
                lens.push(len);
            }
        }
        self.accumulators = vec![Default::default(); lens.len()];
        self.cluster_lens = lens;
        self.rate = Some(rate_hz);
    }

    /// Push one sample.
    ///
    /// `t` is a monotonically increasing timestamp in seconds; `accel` is in
    /// m/s² and `gyro` in rad/s. The IMU must be completely stationary for the
    /// whole recording.
    pub fn push(&mut self, t: f64, accel: [f64; 3], gyro: [f64; 3]) {
        if self.first_time.is_none() {
            self.first_time = Some(t);
        }
        self.last_time = t;
        self.samples += 1;

        if self.rate.is_none() {
            self.pending.push((t, accel, gyro));
            if self.pending.len() >= RATE_ESTIMATION_SAMPLES {
                let first = self.pending.first().unwrap().0;
                let last = self.pending.last().unwrap().0;
                let span = last - first;
                let rate = if span > 0.0 {
                    (self.pending.len() - 1) as f64 / span
                } else {
                    200.0
                };
                self.set_rate(rate);
                let pending = std::mem::take(&mut self.pending);
                for (_, a, g) in pending {
                    self.accumulate(a, g);
                }
            }
            return;
        }

        self.accumulate(accel, gyro);
    }

    /// Convenience wrapper around [`push`](Self::push) for [`ImuMsg`](crate::ImuMsg).
    pub fn push_msg(&mut self, t: f64, msg: &crate::ImuMsg) {
        self.push(t, msg.linear_acceleration, msg.angular_velocity);
    }

    fn accumulate(&mut self, accel: [f64; 3], gyro: [f64; 3]) {
        let channels = [accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]];
        for (acc, &len) in self.accumulators.iter_mut().zip(self.cluster_lens.iter()) {
            for (c, &value) in acc.iter_mut().zip(channels.iter()) {
                c.push(value, len);
            }
        }
    }

    /// Number of samples pushed so far.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Length of the recording so far \[s\].
    pub fn duration(&self) -> f64 {
        match self.first_time {
            Some(first) => self.last_time - first,
            None => 0.0,
        }
    }

    /// The update rate, once it is known.
    pub fn update_rate(&self) -> Option<f64> {
        self.rate
    }

    /// Compute the Allan deviation curves.
    pub fn finish(&self) -> anyhow::Result<AllanResult> {
        let rate = self
            .rate
            .ok_or_else(|| anyhow::anyhow!("too few samples to infer the update rate"))?;

        let mut points = Vec::new();
        for (acc, &len) in self.accumulators.iter().zip(self.cluster_lens.iter()) {
            let mut dev = [0.0f64; 6];
            let mut pairs = usize::MAX;
            let mut usable = true;
            for (i, c) in acc.iter().enumerate() {
                match c.variance(self.cfg.min_pairs) {
                    Some(v) => {
                        dev[i] = v.sqrt();
                        pairs = pairs.min(c.pairs);
                    }
                    None => {
                        usable = false;
                        break;
                    }
                }
            }
            if !usable {
                continue;
            }
            points.push(AllanPoint {
                tau: len as f64 / rate,
                accel: [dev[0], dev[1], dev[2]],
                gyro: [dev[3], dev[4], dev[5]],
                pairs,
            });
        }

        if points.is_empty() {
            return Err(anyhow::anyhow!(
                "no usable Allan deviation points: recording covers {:.1} s, \
                 which is too short for cluster times from {} s",
                self.duration(),
                self.cfg.min_cluster_time
            ));
        }

        Ok(AllanResult {
            points,
            update_rate: rate,
            duration: self.duration(),
            samples: self.samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Gaussian noise, so the tests do not need a rng crate.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            // Box-Muller
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    #[test]
    fn recovers_white_noise_density() {
        let rate = 200.0;
        let dt: f64 = 1.0 / rate;
        // Continuous-time density we want to recover.
        let sigma_a = 2.0e-3;
        let sigma_g = 3.0e-4;
        // Discrete-time standard deviation of one sample.
        let sd_a = sigma_a / dt.sqrt();
        let sd_g = sigma_g / dt.sqrt();

        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut est = AllanEstimator::with_rate(
            AllanConfig {
                min_cluster_time: 0.1,
                max_cluster_time: 30.0,
                num_clusters: 40,
                min_pairs: 10,
            },
            rate,
        )
        .unwrap();

        let n = (rate * 1200.0) as usize; // 20 minutes
        for i in 0..n {
            let t = i as f64 * dt;
            est.push(
                t,
                [
                    sd_a * rng.normal(),
                    sd_a * rng.normal(),
                    9.81 + sd_a * rng.normal(),
                ],
                [
                    sd_g * rng.normal(),
                    sd_g * rng.normal(),
                    sd_g * rng.normal(),
                ],
            );
        }

        let result = est.finish().unwrap();
        let accel = result.noise_density_per_axis(Sensor::Accel).unwrap();
        let gyro = result.noise_density_per_axis(Sensor::Gyro).unwrap();

        for a in accel {
            assert!(
                (a / sigma_a - 1.0).abs() < 0.15,
                "accel noise density {a} vs expected {sigma_a}"
            );
        }
        for g in gyro {
            assert!(
                (g / sigma_g - 1.0).abs() < 0.15,
                "gyro noise density {g} vs expected {sigma_g}"
            );
        }
    }

    #[test]
    fn recovers_the_bias_random_walk() {
        let rate: f64 = 100.0;
        let dt = 1.0 / rate;
        let sigma_a = 2.0e-3;
        let sigma_g = 3.0e-4;
        // Continuous-time random-walk strengths we want back out.
        let rw_a = 1.0e-3;
        let rw_g = 2.0e-4;

        let mut rng = Rng(0xfeed_face_dead_beef);
        let mut est = AllanEstimator::with_rate(
            AllanConfig {
                min_cluster_time: 0.1,
                max_cluster_time: 60.0,
                num_clusters: 60,
                min_pairs: 10,
            },
            rate,
        )
        .unwrap();

        let (mut bias_a, mut bias_g) = ([0.0f64; 3], [0.0f64; 3]);
        let n = (rate * 3600.0) as usize; // one hour
        for i in 0..n {
            let mut accel = [0.0; 3];
            let mut gyro = [0.0; 3];
            for axis in 0..3 {
                bias_a[axis] += rw_a * dt.sqrt() * rng.normal();
                bias_g[axis] += rw_g * dt.sqrt() * rng.normal();
                accel[axis] = bias_a[axis] + sigma_a / dt.sqrt() * rng.normal();
                gyro[axis] = bias_g[axis] + sigma_g / dt.sqrt() * rng.normal();
            }
            est.push(i as f64 * dt, accel, gyro);
        }

        let noise = est.finish().unwrap().noise().unwrap();

        assert!(
            (noise.accel_noise_density / sigma_a - 1.0).abs() < 0.2,
            "accel density {:e} vs {:e}",
            noise.accel_noise_density,
            sigma_a
        );
        assert!(
            (noise.gyro_noise_density / sigma_g - 1.0).abs() < 0.2,
            "gyro density {:e} vs {:e}",
            noise.gyro_noise_density,
            sigma_g
        );
        assert!(
            (noise.accel_random_walk / rw_a - 1.0).abs() < 0.5,
            "accel random walk {:e} vs {:e}",
            noise.accel_random_walk,
            rw_a
        );
        assert!(
            (noise.gyro_random_walk / rw_g - 1.0).abs() < 0.5,
            "gyro random walk {:e} vs {:e}",
            noise.gyro_random_walk,
            rw_g
        );
    }

    /// White noise alone has no rising region, so there is no random walk to
    /// report. Inventing one from a couple of noisy points at the long-tau end
    /// would be worse than saying so.
    #[test]
    fn pure_white_noise_reports_no_random_walk() {
        let rate: f64 = 100.0;
        let dt = 1.0 / rate;
        let sd = 2.0e-3 / dt.sqrt();

        let mut rng = Rng(0x0bad_c0de_0bad_c0de);
        let mut est = AllanEstimator::with_rate(
            AllanConfig {
                min_cluster_time: 0.1,
                max_cluster_time: 60.0,
                num_clusters: 60,
                min_pairs: 10,
            },
            rate,
        )
        .unwrap();

        for i in 0..(rate * 1200.0) as usize {
            est.push(
                i as f64 * dt,
                [
                    sd * rng.normal(),
                    sd * rng.normal(),
                    9.81 + sd * rng.normal(),
                ],
                [sd * rng.normal(), sd * rng.normal(), sd * rng.normal()],
            );
        }

        let result = est.finish().unwrap();
        // The white-noise half is still perfectly measurable.
        let density = result.noise_density_per_axis(Sensor::Accel).unwrap();
        assert!((density[0] / 2.0e-3 - 1.0).abs() < 0.15, "{density:?}");

        let err = result.noise().unwrap_err().to_string();
        assert!(err.contains("slope +0.5"), "{err}");
    }

    #[test]
    fn infers_the_update_rate() {
        let mut est = AllanEstimator::new(AllanConfig {
            min_cluster_time: 0.05,
            max_cluster_time: 5.0,
            num_clusters: 20,
            min_pairs: 5,
        });
        let rate = 100.0;
        for i in 0..20_000 {
            let t = i as f64 / rate;
            est.push(t, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]);
        }
        let inferred = est.update_rate().unwrap();
        assert!((inferred - rate).abs() < 1.0, "inferred {inferred}");
    }

    #[test]
    fn constant_signal_has_zero_deviation() {
        let mut est = AllanEstimator::with_rate(
            AllanConfig {
                min_cluster_time: 0.1,
                max_cluster_time: 2.0,
                num_clusters: 10,
                min_pairs: 5,
            },
            100.0,
        )
        .unwrap();
        for i in 0..10_000 {
            est.push(i as f64 / 100.0, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]);
        }
        let r = est.finish().unwrap();
        assert!(!r.points.is_empty());
        for p in &r.points {
            assert!(p.accel.iter().all(|v| *v < 1e-12));
            assert!(p.gyro.iter().all(|v| *v < 1e-12));
        }
        // No slope regions exist, so the fit must report a clear error.
        assert!(r.noise().is_err());
    }

    #[test]
    fn csv_has_a_row_per_point() {
        let mut est = AllanEstimator::with_rate(
            AllanConfig {
                min_cluster_time: 0.1,
                max_cluster_time: 2.0,
                num_clusters: 10,
                min_pairs: 5,
            },
            100.0,
        )
        .unwrap();
        for i in 0..10_000 {
            est.push(i as f64 / 100.0, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]);
        }
        let r = est.finish().unwrap();
        let csv = r.to_csv_string();
        assert_eq!(csv.lines().count(), r.points.len() + 1);
    }
}
