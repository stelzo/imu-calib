//! Reading and writing Kalibr's IMU YAML files.
//!
//! Write one Kalibr input file with [`ImuIntrinsics::save_kalibr_input_yaml`] after running the
//! Allan-variance estimator.
//!
//! # Biases
//!
//! Kalibr models the accelerometer and gyroscope biases as time-varying
//! B-splines and does not export them, so a genuine Kalibr file has no bias
//! entries and this crate reads them as zero. When this crate writes a file it
//! adds `b` next to `M` under `accelerometers` and `gyroscopes`. Kalibr ignores
//! unknown keys, so such a file still works as Kalibr input.

use crate::conv::{Mat3, Mat4, Vec3, ZERO3};
use crate::{ImuIntrinsics, ImuModel, ImuNoise};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

type Matrix3Yaml = Vec<Vec<f64>>;

#[derive(Debug, Default, Serialize, Deserialize)]
struct AccelerometersYaml {
    #[serde(rename = "M", skip_serializing_if = "Option::is_none")]
    m: Option<Matrix3Yaml>,
    /// Bias. Not part of Kalibr's own output; see the module docs.
    #[serde(rename = "b", skip_serializing_if = "Option::is_none")]
    b: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rx_i: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ry_i: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rz_i: Option<Vec<f64>>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct GyroscopesYaml {
    #[serde(rename = "M", skip_serializing_if = "Option::is_none")]
    m: Option<Matrix3Yaml>,
    #[serde(rename = "A", skip_serializing_if = "Option::is_none")]
    a: Option<Matrix3Yaml>,
    #[serde(rename = "C_gyro_i", skip_serializing_if = "Option::is_none")]
    c_gyro_i: Option<Matrix3Yaml>,
    /// Bias. Not part of Kalibr's own output; see the module docs.
    #[serde(rename = "b", skip_serializing_if = "Option::is_none")]
    b: Option<Vec<f64>>,
}

/// One IMU block of a Kalibr YAML file.
#[derive(Debug, Default, Serialize, Deserialize)]
struct ImuYaml {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rostopic: Option<String>,
    #[serde(default = "default_update_rate")]
    update_rate: f64,

    #[serde(default)]
    accelerometer_noise_density: f64,
    #[serde(default)]
    accelerometer_random_walk: f64,
    #[serde(default)]
    gyroscope_noise_density: f64,
    #[serde(default)]
    gyroscope_random_walk: f64,

    #[serde(skip_serializing_if = "Option::is_none")]
    accelerometers: Option<AccelerometersYaml>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gyroscopes: Option<GyroscopesYaml>,

    #[serde(rename = "T_i_b", skip_serializing_if = "Option::is_none")]
    t_i_b: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_offset: Option<f64>,
}

fn default_update_rate() -> f64 {
    200.0
}


fn to_matrix3(rows: &[Vec<f64>], what: &str) -> anyhow::Result<Mat3> {
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
        return Err(anyhow::anyhow!(
            "`{what}` must be a 3x3 matrix given as three rows of three values"
        ));
    }
    let mut m = ZERO3;
    for (i, row) in rows.iter().enumerate() {
        m[i].copy_from_slice(row);
    }
    Ok(m)
}

fn to_matrix4(rows: &[Vec<f64>], what: &str) -> anyhow::Result<Mat4> {
    if rows.len() != 4 || rows.iter().any(|r| r.len() != 4) {
        return Err(anyhow::anyhow!(
            "`{what}` must be a 4x4 matrix given as four rows of four values"
        ));
    }
    let mut m = [[0.0; 4]; 4];
    for (i, row) in rows.iter().enumerate() {
        m[i].copy_from_slice(row);
    }
    Ok(m)
}

fn to_vector3(v: &[f64], what: &str) -> anyhow::Result<Vec3> {
    if v.len() != 3 {
        return Err(anyhow::anyhow!("`{what}` must have exactly 3 values"));
    }
    Ok([v[0], v[1], v[2]])
}

fn from_matrix3(m: &Mat3) -> Matrix3Yaml {
    m.iter()
        .map(|row| row.iter().map(|v| normalize_zero(*v)).collect())
        .collect()
}

/// Turn a negative zero into a plain one, so structurally zero entries do not
/// read as though something was estimated there.
fn normalize_zero(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

fn from_matrix4(m: &Mat4) -> Vec<Vec<f64>> {
    m.iter()
        .map(|row| row.iter().map(|v| normalize_zero(*v)).collect())
        .collect()
}

impl ImuYaml {
    fn into_intrinsics(self) -> anyhow::Result<ImuIntrinsics> {
        let mut intr = ImuIntrinsics::identity();

        if let Some(model) = &self.model {
            intr.model = ImuModel::from_str_kalibr(model)?;
        }
        intr.rostopic = self.rostopic;
        intr.time_offset = self.time_offset.unwrap_or(0.0);
        intr.noise = ImuNoise {
            accel_noise_density: self.accelerometer_noise_density,
            accel_random_walk: self.accelerometer_random_walk,
            gyro_noise_density: self.gyroscope_noise_density,
            gyro_random_walk: self.gyroscope_random_walk,
            update_rate: self.update_rate,
        };

        if let Some(t) = &self.t_i_b {
            intr.t_i_b = Some(to_matrix4(t, "T_i_b")?);
        }

        if let Some(accel) = &self.accelerometers {
            if let Some(m) = &accel.m {
                intr.accel_m = to_matrix3(m, "accelerometers.M")?;
            }
            if let Some(b) = &accel.b {
                intr.accel_bias = to_vector3(b, "accelerometers.b")?;
            }
            match (&accel.rx_i, &accel.ry_i, &accel.rz_i) {
                (Some(rx), Some(ry), Some(rz)) => {
                    intr.accel_lever_arms = Some([
                        to_vector3(rx, "accelerometers.rx_i")?,
                        to_vector3(ry, "accelerometers.ry_i")?,
                        to_vector3(rz, "accelerometers.rz_i")?,
                    ]);
                }
                (None, None, None) => {}
                _ => {
                    return Err(anyhow::anyhow!(
                        "accelerometers.rx_i, ry_i and rz_i must all be present or all absent"
                    ))
                }
            }
        }

        if let Some(gyro) = &self.gyroscopes {
            if let Some(m) = &gyro.m {
                intr.gyro_m = to_matrix3(m, "gyroscopes.M")?;
            }
            if let Some(a) = &gyro.a {
                intr.gyro_a = to_matrix3(a, "gyroscopes.A")?;
            }
            if let Some(c) = &gyro.c_gyro_i {
                intr.c_gyro_i = to_matrix3(c, "gyroscopes.C_gyro_i")?;
            }
            if let Some(b) = &gyro.b {
                intr.gyro_bias = to_vector3(b, "gyroscopes.b")?;
            }
        }

        Ok(intr)
    }

    fn from_intrinsics(intr: &ImuIntrinsics, full: bool) -> Self {
        let noise = &intr.noise;
        let mut out = Self {
            model: Some(intr.model.as_str().to_string()),
            rostopic: intr.rostopic.clone(),
            update_rate: noise.update_rate,
            accelerometer_noise_density: noise.accel_noise_density,
            accelerometer_random_walk: noise.accel_random_walk,
            gyroscope_noise_density: noise.gyro_noise_density,
            gyroscope_random_walk: noise.gyro_random_walk,
            ..Default::default()
        };

        if !full {
            // Kalibr's input file carries the noise parameters only.
            out.model = None;
            return out;
        }

        out.accelerometers = Some(AccelerometersYaml {
            m: Some(from_matrix3(&intr.accel_m)),
            b: Some(intr.accel_bias.to_vec()),
            rx_i: intr.accel_lever_arms.map(|r| r[0].to_vec()),
            ry_i: intr.accel_lever_arms.map(|r| r[1].to_vec()),
            rz_i: intr.accel_lever_arms.map(|r| r[2].to_vec()),
        });
        out.gyroscopes = Some(GyroscopesYaml {
            m: Some(from_matrix3(&intr.gyro_m)),
            a: Some(from_matrix3(&intr.gyro_a)),
            c_gyro_i: Some(from_matrix3(&intr.c_gyro_i)),
            b: Some(intr.gyro_bias.to_vec()),
        });
        out.t_i_b = intr.t_i_b.as_ref().map(from_matrix4);
        out.time_offset = Some(intr.time_offset);
        out
    }
}

/// Parse either layout: a bare IMU block, or a map of `imu0`, `imu1`, … blocks.
fn parse_document(yaml: &str) -> anyhow::Result<Vec<(String, ImuYaml)>> {
    // A results file is a map whose values are themselves maps of IMU fields.
    if let Ok(set) = serde_yaml::from_str::<BTreeMap<String, ImuYaml>>(yaml) {
        if !set.is_empty() {
            let mut imus: Vec<(String, ImuYaml)> = set.into_iter().collect();
            // BTreeMap already sorts imu0, imu1, ... lexicographically, which
            // matches Kalibr's numbering for any realistic IMU count.
            imus.sort_by(|a, b| a.0.cmp(&b.0));
            return Ok(imus);
        }
    }

    let single: ImuYaml = serde_yaml::from_str(yaml)
        .map_err(|e| anyhow::anyhow!("could not parse the file as a Kalibr IMU config: {e}"))?;
    Ok(vec![("imu0".to_string(), single)])
}

impl ImuIntrinsics {
    /// Load the first IMU from a Kalibr YAML file.
    ///
    /// Accepts both Kalibr's input format and its `imu-<bagname>.yaml` results
    /// format.
    pub fn from_kalibr_yaml(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("could not read {}: {e}", path.as_ref().display()))?;
        Self::from_kalibr_yaml_str(&contents)
    }

    /// Load the IMU named `name` (for example `"imu1"`) from a Kalibr YAML file.
    pub fn from_kalibr_yaml_named(path: impl AsRef<Path>, name: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("could not read {}: {e}", path.as_ref().display()))?;
        let imus = parse_document(&contents)?;
        let found = imus
            .into_iter()
            .find(|(key, _)| key == name)
            .ok_or_else(|| anyhow::anyhow!("no IMU named `{name}` in the file"))?;
        found.1.into_intrinsics()
    }

    /// Parse the first IMU from a Kalibr YAML string.
    pub fn from_kalibr_yaml_str(yaml: &str) -> anyhow::Result<Self> {
        let mut imus = parse_document(yaml)?;
        if imus.is_empty() {
            return Err(anyhow::anyhow!("the file contains no IMU configuration"));
        }
        imus.remove(0).1.into_intrinsics()
    }

    /// Parse every IMU of a Kalibr YAML string, keyed by name.
    pub fn all_from_kalibr_yaml_str(yaml: &str) -> anyhow::Result<Vec<(String, Self)>> {
        parse_document(yaml)?
            .into_iter()
            .map(|(name, imu)| Ok((name, imu.into_intrinsics()?)))
            .collect()
    }

    /// Serialise to Kalibr's results layout, nested under `name`.
    pub fn to_kalibr_yaml_string_named(&self, name: &str) -> anyhow::Result<String> {
        let mut doc: BTreeMap<String, ImuYaml> = BTreeMap::new();
        doc.insert(name.to_string(), ImuYaml::from_intrinsics(self, true));
        Ok(to_kalibr_flow_style(&serde_yaml::to_string(&doc)?))
    }

    /// Serialise to Kalibr's results layout, nested under `imu0`.
    pub fn to_kalibr_yaml_string(&self) -> anyhow::Result<String> {
        self.to_kalibr_yaml_string_named("imu0")
    }

    /// Serialise the noise parameters in the flat layout Kalibr reads as input.
    ///
    /// This is the file you pass to `kalibr_calibrate_imu_camera --imu`.
    pub fn to_kalibr_input_yaml_string(&self) -> anyhow::Result<String> {
        Ok(to_kalibr_flow_style(&serde_yaml::to_string(
            &ImuYaml::from_intrinsics(self, false),
        )?))
    }

    /// Write the full parameter set in Kalibr's results layout.
    ///
    /// Returns the canonical path of the file that was written.
    pub fn save_kalibr_yaml(&self, path: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
        write_string(path, &self.to_kalibr_yaml_string()?)
    }

    /// Write the noise parameters in the flat layout Kalibr reads as input.
    pub fn save_kalibr_input_yaml(&self, path: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
        write_string(path, &self.to_kalibr_input_yaml_string()?)
    }
}

/// Rewrite leaf sequences of scalars in flow style.
///
/// `serde_yaml` always emits block style, while Kalibr dumps its YAML with
/// PyYAML's `default_flow_style=None`, which puts sequences of plain scalars on
/// one line and leaves sequences of collections in block style. Matching that
/// keeps a file written here visually interchangeable with one written by
/// Kalibr; the content is identical either way.
fn to_kalibr_flow_style(yaml: &str) -> String {
    let lines: Vec<&str> = yaml.lines().collect();
    let inner = collapse_nested_sequences(&lines);
    let inner_refs: Vec<&str> = inner.iter().map(String::as_str).collect();
    let mut out = collapse_scalar_values(&inner_refs).join("\n");
    out.push('\n');
    out
}

/// Number of leading spaces.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// `    - - 1.0` / `      - 0.0` ... becomes `    - [1.0, 0.0, ...]`.
fn collapse_nested_sequences(lines: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let indent = indent_of(line);
        let trimmed = line.trim_start();

        let first = match trimmed.strip_prefix("- - ") {
            Some(v) if is_scalar(v) => v,
            _ => {
                out.push(line.to_string());
                i += 1;
                continue;
            }
        };

        // The remaining elements sit two spaces further in.
        let child_prefix = format!("{}  - ", " ".repeat(indent));
        let mut values = vec![first.to_string()];
        let mut j = i + 1;
        while j < lines.len() {
            match lines[j].strip_prefix(child_prefix.as_str()) {
                Some(v) if is_scalar(v) => {
                    values.push(v.to_string());
                    j += 1;
                }
                _ => break,
            }
        }

        out.push(format!("{}- [{}]", " ".repeat(indent), values.join(", ")));
        i = j;
    }
    out
}

/// `    b:` followed by `    - 0.1` ... becomes `    b: [0.1, ...]`.
///
/// Only applies when every element is a plain scalar, so sequences of rows keep
/// their block layout exactly as Kalibr writes them.
fn collapse_scalar_values(lines: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_end();

        if !trimmed.ends_with(':') || trimmed.trim_start().starts_with('-') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        let indent = indent_of(line);
        let item_prefix = format!("{}- ", " ".repeat(indent));
        let mut values: Vec<String> = Vec::new();
        let mut j = i + 1;
        while j < lines.len() {
            match lines[j].strip_prefix(item_prefix.as_str()) {
                Some(v) if is_scalar(v) => {
                    values.push(v.to_string());
                    j += 1;
                }
                _ => break,
            }
        }

        if values.is_empty() {
            out.push(line.to_string());
            i += 1;
        } else {
            out.push(format!("{trimmed} [{}]", values.join(", ")));
            i = j;
        }
    }
    out
}

/// A plain scalar: not a nested collection and not a mapping entry.
fn is_scalar(value: &str) -> bool {
    let v = value.trim();
    // A leading '-' is fine, that is just a negative number; "- " would be a
    // nested sequence item.
    !v.is_empty()
        && !v.starts_with("- ")
        && !v.starts_with('[')
        && !v.starts_with('{')
        && !v.contains(": ")
        && !v.ends_with(':')
}

fn write_string(path: impl AsRef<Path>, contents: &str) -> anyhow::Result<PathBuf> {
    let path = path.as_ref();
    let mut file = File::create(path)
        .map_err(|e| anyhow::anyhow!("could not create {}: {e}", path.display()))?;
    file.write_all(contents.as_bytes())?;
    file.flush()?;
    Ok(std::fs::canonicalize(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{IDENTITY3, ZERO_VEC3};

    const EPS: f64 = 1e-12;

    #[track_caller]
    fn assert_m3(a: &Mat3, b: &Mat3) {
        for (ra, rb) in a.iter().zip(b) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < EPS, "{a:?} != {b:?}");
            }
        }
    }

    #[track_caller]
    fn assert_v3(a: &Vec3, b: &Vec3) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < EPS, "{a:?} != {b:?}");
        }
    }

    fn sample() -> ImuIntrinsics {
        let mut intr = ImuIntrinsics::identity();
        intr.model = ImuModel::ScaleMisalignment;
        intr.accel_m = [[1.01, 0.0, 0.0], [0.001, 1.02, 0.0], [0.002, 0.003, 1.03]];
        intr.accel_bias = [0.1, 0.2, 0.3];
        intr.gyro_m = [[0.99, 0.0, 0.0], [-0.001, 0.98, 0.0], [-0.002, 0.001, 0.97]];
        intr.gyro_bias = [0.01, 0.02, 0.03];
        intr.gyro_a = [[1e-3, 0.0, 0.0], [0.0, 2e-3, 0.0], [0.0, 0.0, 3e-3]];
        intr.c_gyro_i = [
            [0.99999949, 0.00071135, 0.00071417],
            [-0.00071004, 0.99999806, -0.00183835],
            [-0.00071547, 0.00183784, 0.99999806],
        ];
        intr.noise = ImuNoise {
            accel_noise_density: 1.86e-3,
            accel_random_walk: 4.33e-4,
            gyro_noise_density: 1.87e-4,
            gyro_random_walk: 2.66e-5,
            update_rate: 200.0,
        };
        intr.rostopic = Some("/imu0".to_string());
        intr.time_offset = 0.0;
        intr
    }

    #[test]
    fn round_trip() {
        let original = sample();
        let yaml = original.to_kalibr_yaml_string().unwrap();
        let parsed = ImuIntrinsics::from_kalibr_yaml_str(&yaml).unwrap();

        assert_eq!(parsed.model, ImuModel::ScaleMisalignment);
        assert_m3(&original.accel_m, &parsed.accel_m);
        assert_v3(&original.accel_bias, &parsed.accel_bias);
        assert_m3(&original.gyro_m, &parsed.gyro_m);
        assert_v3(&original.gyro_bias, &parsed.gyro_bias);
        assert_m3(&original.gyro_a, &parsed.gyro_a);
        assert_m3(&original.c_gyro_i, &parsed.c_gyro_i);
        assert_eq!(original.noise, parsed.noise);
        assert_eq!(original.rostopic, parsed.rostopic);
    }

    #[test]
    fn output_is_nested_under_imu0() {
        let yaml = sample().to_kalibr_yaml_string().unwrap();
        assert!(yaml.starts_with("imu0:"), "{yaml}");
        assert!(yaml.contains("model: scale-misalignment"));
        assert!(yaml.contains("C_gyro_i"));
    }

    #[test]
    fn parses_a_real_kalibr_results_file() {
        // Layout and values as produced by kalibr_calibrate_imu_camera with
        // --imu-models scale-misalignment.
        let yaml = r#"
imu0:
  T_i_b:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 0.006
  accelerometer_random_walk: 0.0002
  accelerometers:
    M:
    - [1.00206, 0.0, 0.0]
    - [0.00123, 0.99871, 0.0]
    - [-0.00045, 0.00231, 1.00412]
  gyroscope_noise_density: 0.0004
  gyroscope_random_walk: 4.0e-06
  gyroscopes:
    A:
    - [-0.00075474, 0.00162226, 0.00019704]
    - [-0.00346093, -0.00023962, -0.00694434]
    - [-0.00216607, 0.00254173, 0.00037087]
    C_gyro_i:
    - [0.99999949, 0.00071135, 0.00071417]
    - [-0.00071004, 0.99999806, -0.00183835]
    - [-0.00071547, 0.00183784, 0.99999806]
    M:
    - [0.99732203, 0.0, 0.0]
    - [0.00373207, 0.99372249, 0.0]
    - [-0.00218998, -0.00186741, 0.99919668]
  model: scale-misalignment
  rostopic: /imu0
  time_offset: 0.0
  update_rate: 200.0
"#;
        let intr = ImuIntrinsics::from_kalibr_yaml_str(yaml).unwrap();
        assert_eq!(intr.model, ImuModel::ScaleMisalignment);
        assert_eq!(intr.rostopic.as_deref(), Some("/imu0"));
        assert!((intr.accel_m[0][0] - 1.00206).abs() < EPS);
        assert!((intr.gyro_m[2][2] - 0.99919668).abs() < EPS);
        assert!((intr.gyro_a[1][2] - (-0.00694434)).abs() < EPS);
        assert!((intr.c_gyro_i[0][1] - 0.00071135).abs() < EPS);
        assert!((intr.noise.update_rate - 200.0).abs() < EPS);
        // Kalibr does not export biases.
        assert_v3(&intr.accel_bias, &ZERO_VEC3);
        assert_v3(&intr.gyro_bias, &ZERO_VEC3);
        // And the matrices must be usable.
        assert!(intr.corrector().is_ok());
        assert!(intr.t_i_b.is_some());
    }

    #[test]
    fn parses_the_flat_input_format() {
        let yaml = r#"
rostopic: /imu0
update_rate: 200.0
accelerometer_noise_density: 1.86e-03
accelerometer_random_walk: 4.33e-04
gyroscope_noise_density: 1.87e-04
gyroscope_random_walk: 2.66e-05
"#;
        let intr = ImuIntrinsics::from_kalibr_yaml_str(yaml).unwrap();
        assert!((intr.noise.accel_noise_density - 1.86e-3).abs() < EPS);
        assert!((intr.noise.gyro_random_walk - 2.66e-5).abs() < EPS);
        assert_eq!(intr.model, ImuModel::Calibrated);
        // Everything deterministic defaults to pass-through.
        assert_m3(&intr.accel_m, &IDENTITY3);
        assert_m3(&intr.gyro_m, &IDENTITY3);
        assert_m3(&intr.gyro_a, &ZERO3);
    }

    #[test]
    fn input_yaml_is_flat_and_noise_only() {
        let yaml = sample().to_kalibr_input_yaml_string().unwrap();
        assert!(!yaml.contains("imu0:"), "{yaml}");
        assert!(!yaml.contains("accelerometers"), "{yaml}");
        assert!(yaml.contains("accelerometer_noise_density"));
        assert!(yaml.contains("rostopic: /imu0"));
        // Kalibr must be able to read it back as a config.
        let parsed = ImuIntrinsics::from_kalibr_yaml_str(&yaml).unwrap();
        assert_eq!(parsed.noise, sample().noise);
    }

    #[test]
    fn multi_imu_file_is_read_in_order() {
        let yaml = r#"
imu0:
  update_rate: 200.0
  rostopic: /imu0
  accelerometer_noise_density: 0.001
  accelerometer_random_walk: 0.0001
  gyroscope_noise_density: 0.0002
  gyroscope_random_walk: 2.0e-06
imu1:
  update_rate: 400.0
  rostopic: /imu1
  accelerometer_noise_density: 0.002
  accelerometer_random_walk: 0.0002
  gyroscope_noise_density: 0.0004
  gyroscope_random_walk: 4.0e-06
"#;
        let all = ImuIntrinsics::all_from_kalibr_yaml_str(yaml).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].0, "imu0");
        assert_eq!(all[1].0, "imu1");
        assert!((all[1].1.noise.update_rate - 400.0).abs() < EPS);

        let first = ImuIntrinsics::from_kalibr_yaml_str(yaml).unwrap();
        assert_eq!(first.rostopic.as_deref(), Some("/imu0"));
    }

    #[test]
    fn file_round_trip() {
        let original = sample();
        let path = std::env::temp_dir().join("imu_calib_kalibr_round_trip.yaml");
        original.save_kalibr_yaml(&path).unwrap();
        let loaded = ImuIntrinsics::from_kalibr_yaml(&path).unwrap();
        assert_m3(&original.accel_m, &loaded.accel_m);
        assert_m3(&original.gyro_a, &loaded.gyro_a);
        assert_eq!(original.noise, loaded.noise);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_malformed_matrix_is_rejected() {
        let yaml = r#"
imu0:
  update_rate: 200.0
  accelerometers:
    M:
    - [1.0, 0.0]
    - [0.0, 1.0]
"#;
        let err = ImuIntrinsics::from_kalibr_yaml_str(yaml).unwrap_err();
        assert!(err.to_string().contains("accelerometers.M"), "{err}");
    }

    #[test]
    fn matrix_rows_are_written_in_flow_style() {
        let yaml = sample().to_kalibr_yaml_string().unwrap();
        assert!(yaml.contains("    - [1.01, 0.0, 0.0]"), "{yaml}");
        assert!(yaml.contains("b: [0.1, 0.2, 0.3]"), "{yaml}");
        // No leftover block-style nesting.
        assert!(!yaml.contains("- - "), "{yaml}");
        // And it still parses.
        let parsed = ImuIntrinsics::from_kalibr_yaml_str(&yaml).unwrap();
        assert_m3(&parsed.accel_m, &sample().accel_m);
        assert_v3(&parsed.accel_bias, &sample().accel_bias);
    }

    #[test]
    fn structural_zeros_are_written_without_a_sign() {
        let mut intr = sample();
        // Round-tripping a lower-triangular matrix through an inverse leaves
        // -0.0 in the entries above the diagonal.
        intr.accel_m = [
            [1.01, -0.0, -0.0],
            [0.004, 0.99, -0.0],
            [-0.006, 0.003, 1.02],
        ];
        let yaml = intr.to_kalibr_yaml_string().unwrap();
        assert!(!yaml.contains("-0.0,"), "{yaml}");
        assert!(!yaml.contains("-0.0]"), "{yaml}");
    }

    #[test]
    fn flow_style_keeps_rows_of_a_matrix_in_block_style() {
        let block = "imu0:\n  T_i_b:\n  - - 1.0\n    - 0.0\n  - - 0.0\n    - 1.0\n";
        let flow = to_kalibr_flow_style(block);
        assert_eq!(flow, "imu0:\n  T_i_b:\n  - [1.0, 0.0]\n  - [0.0, 1.0]\n");
    }

    #[test]
    fn flow_style_leaves_scalars_and_strings_alone() {
        let doc = "imu0:\n  rostopic: /imu0\n  update_rate: 200.0\n";
        assert_eq!(to_kalibr_flow_style(doc), doc);
    }

    #[test]
    fn flow_style_handles_negative_numbers() {
        let block = "  b:\n  - -0.5\n  - 0.25\n";
        assert_eq!(to_kalibr_flow_style(block), "  b: [-0.5, 0.25]\n");
    }

    #[test]
    fn an_unknown_model_is_rejected() {
        let yaml = "imu0:\n  model: something-else\n  update_rate: 200.0\n";
        assert!(ImuIntrinsics::from_kalibr_yaml_str(yaml).is_err());
    }
}
