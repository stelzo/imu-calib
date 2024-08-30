use std::{
    collections::HashMap,
    fs::File,
    hash::Hash,
    io::{Read, Write},
    path::PathBuf,
};

use nalgebra as na;

#[derive(Debug, Default)]
pub struct ImuMsg {
    pub orientation: [f64; 4],
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: [f64; 3],
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: [f64; 3],
    pub linear_acceleration_covariance: [f64; 9],
}

impl ImuMsg {
    pub fn new(linear_accel: &[f64; 3], angular_vel: &[f64; 3]) -> Self {
        Self {
            linear_acceleration: *linear_accel,
            angular_velocity: *angular_vel,
            ..Default::default()
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum Orientation {
    Xpos,
    Xneg,
    Ypos,
    Yneg,
    Zpos,
    Zneg,
}

#[derive(Debug)]
struct AccelerometerCalibration {
    pub reference_index: [usize; 6],
    pub reference_sign: [i32; 6],
    pub calib_ready: bool,
    pub sm: na::Matrix3<f64>,   // combined scale and misalignment parameters
    pub bias: na::Vector3<f64>, // scaled and rotated bias parameters
    pub reference_acceleration: f64, // expected acceleration measurement (e.g. 1.0 for unit of g's, 9.80665 for unit of m/s^2)
    pub calib_initialized: bool,
    pub orientation_count: [usize; 6],
    pub meas: na::DMatrix<f64>,       // least squares measurements matrix
    pub ref_: na::DVector<f64>,       // least squares expected measurements vector
    pub num_measurements: usize,      // number of measurements expected for this calibration
    pub measurements_received: usize, // number of measurements received for this calibration
}

impl AccelerometerCalibration {
    pub fn create(calib_file_path: Option<String>) -> anyhow::Result<Self> {
        let reference_index = [0, 0, 1, 1, 2, 2];
        let reference_sign = [1, -1, 1, -1, 1, -1];
        let calib_ready = false;
        let sm = na::Matrix3::identity();
        let bias = na::Vector3::zeros();
        let reference_acceleration = 1.0;
        let calib_initialized = false;
        let orientation_count = [0, 0, 0, 0, 0, 0];
        let meas = na::DMatrix::zeros(0, 3);
        let ref_ = na::DVector::zeros(0);
        let num_measurements = 0;
        let measurements_received = 0;

        let mut s = Self {
            reference_index,
            reference_sign,
            calib_ready,
            sm,
            bias,
            reference_acceleration,
            calib_initialized,
            orientation_count,
            meas,
            ref_,
            num_measurements,
            measurements_received,
        };

        if let Some(calib_file_path) = calib_file_path {
            s.load_calib(calib_file_path)?;
        }

        Ok(s)
    }

    fn load_calib(&mut self, calib_file: String) -> anyhow::Result<()> {
        let mut file = File::open(calib_file)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let lines = contents.lines();
        let sm_str = "SM = [";
        let bias_str = "bias = [";
        let mut found_sm = false;
        let mut found_bias = false;

        for line in lines {
            if let Some(line) = line.strip_prefix(sm_str) {
                let mut iter = line.split(", ");
                for sm_val in self.sm.iter_mut() {
                    *sm_val = iter.next().unwrap().parse::<f64>()?;
                }
                found_sm = true;
            }

            if let Some(line) = line.strip_prefix(bias_str) {
                let mut iter = line.split(", ");
                for bias_val in self.bias.iter_mut() {
                    *bias_val = iter.next().unwrap().parse::<f64>()?;
                }
                found_bias = true;
            }
        }

        if !found_sm || !found_bias {
            return Err(anyhow::anyhow!("Calibration file not formatted correctly"));
        }

        self.calib_ready = true;

        Ok(())
    }

    fn save_calib(&self, calib_file: &str) -> anyhow::Result<PathBuf> {
        if !self.calib_ready {
            return Err(anyhow::anyhow!("Calibration not ready"));
        }

        let sm = self
            .sm
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        let bias = self
            .bias
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();

        let mut out_str = String::new();
        out_str.push_str("SM = [");
        sm.iter().for_each(|x| {
            out_str.push_str(x);
            out_str.push_str(", ");
        });
        out_str.push_str("]\n");

        out_str.push_str("bias = [");
        bias.iter().for_each(|x| {
            out_str.push_str(x);
            out_str.push_str(", ");
        });
        out_str.push_str("]\n");

        let mut buff = File::create(calib_file)?;
        buff.write_all(out_str.as_bytes())?;
        buff.flush()?;

        // get the absolute path of the file
        let path = std::fs::canonicalize(calib_file)?;

        Ok(path)
    }

    fn begin_calib(&mut self, measurements: usize, reference_acceleration: f64) {
        self.reference_acceleration = reference_acceleration;
        self.num_measurements = measurements;
        self.measurements_received = 0;
        self.meas = na::DMatrix::zeros(3 * measurements, 12);
        self.ref_ = na::DVector::zeros(3 * measurements);
        self.orientation_count = [0, 0, 0, 0, 0, 0];
        self.calib_initialized = true;
    }

    fn add_measurement(&mut self, orientation: &Orientation, ax: f64, ay: f64, az: f64) -> bool {
        if !self.calib_initialized || self.measurements_received >= self.num_measurements {
            return false;
        }

        let orientation = match orientation {
            Orientation::Xpos => 0,
            Orientation::Xneg => 1,
            Orientation::Ypos => 2,
            Orientation::Yneg => 3,
            Orientation::Zpos => 4,
            Orientation::Zneg => 5,
        };

        for i in 0..3 {
            self.meas[(3 * self.measurements_received + i, 3 * i)] = ax;
            self.meas[(3 * self.measurements_received + i, 3 * i + 1)] = ay;
            self.meas[(3 * self.measurements_received + i, 3 * i + 2)] = az;

            self.meas[(3 * self.measurements_received + i, 9 + i)] = -1.0;
        }

        self.ref_[3 * self.measurements_received + self.reference_index[orientation]] =
            self.reference_sign[orientation] as f64 * self.reference_acceleration;

        self.measurements_received += 1;
        self.orientation_count[orientation] += 1;

        true
    }

    fn compute_calib(&mut self) -> bool {
        if self.measurements_received < 12 {
            return false;
        }

        for i in 0..6 {
            if self.orientation_count[i] == 0 {
                return false;
            }
        }

        // solve least squares
        let xhat = self
            .meas
            .clone()
            .svd(true, true)
            .solve(&self.ref_, 1e-6)
            .unwrap();

        for i in 0..9 {
            self.sm[(i / 3, i % 3)] = xhat[i];
        }

        for i in 0..3 {
            self.bias[i] = xhat[9 + i];
        }

        self.calib_ready = true;

        true
    }

    fn apply_calib(&self, raw: [f64; 3]) -> [f64; 3] {
        let raw_accel = na::Vector3::new(raw[0], raw[1], raw[2]);
        let corrected_accel = self.sm * raw_accel - self.bias;

        [corrected_accel[0], corrected_accel[1], corrected_accel[2]]
    }
}

#[derive(Debug)]
pub enum CalibState {
    START,
    SWITCHING,
    RECEIVING,
    COMPUTING,
    DONE,
}

#[derive(Debug)]
pub struct CalibrateProcess {
    state: CalibState,
    measurements_per_orientation: usize,
    measurements_received: usize,
    reference_acceleration: f64,
    output_file: String,
    orientations: Vec<Orientation>,
    current_orientation: Orientation,
    orientation_labels: HashMap<Orientation, String>,
    calib: AccelerometerCalibration,
}

impl CalibrateProcess {
    pub fn new(
        measurements_per_orientation: usize,
        reference_acceleration: f64,
        output_file: String,
    ) -> Self {
        let state = CalibState::START;
        let measurements_received = 0;
        let orientations = vec![
            Orientation::Xpos,
            Orientation::Xneg,
            Orientation::Ypos,
            Orientation::Yneg,
            Orientation::Zpos,
            Orientation::Zneg,
        ];
        let current_orientation = Orientation::Xpos;
        let orientation_labels = {
            let mut map = HashMap::new();
            map.insert(Orientation::Xpos, "X+".to_string());
            map.insert(Orientation::Xneg, "X-".to_string());
            map.insert(Orientation::Ypos, "Y+".to_string());
            map.insert(Orientation::Yneg, "Y-".to_string());
            map.insert(Orientation::Zpos, "Z+".to_string());
            map.insert(Orientation::Zneg, "Z-".to_string());
            map
        };

        Self {
            state,
            measurements_per_orientation,
            measurements_received,
            reference_acceleration,
            output_file,
            orientations,
            current_orientation,
            orientation_labels,
            calib: AccelerometerCalibration::create(None).unwrap(),
        }
    }

    pub fn imu_callback(&mut self, imu: ImuMsg) -> anyhow::Result<()> {
        match self.state {
            CalibState::START => {
                self.calib.begin_calib(
                    6 * self.measurements_per_orientation,
                    self.reference_acceleration,
                );
                self.state = CalibState::SWITCHING;
            }
            CalibState::SWITCHING => {
                if self.orientations.is_empty() {
                    self.state = CalibState::COMPUTING;
                } else {
                    self.current_orientation = self.orientations.remove(0);
                    self.measurements_received = 0;
                    println!(
                        "Orient IMU with {} axis up and press Enter",
                        self.orientation_labels[&self.current_orientation]
                    );
                    let _ = std::io::stdin().read_line(&mut String::new())?;
                    self.state = CalibState::RECEIVING;
                }
            }
            CalibState::RECEIVING => {
                let accepted = self.calib.add_measurement(
                    &self.current_orientation,
                    imu.linear_acceleration[0],
                    imu.linear_acceleration[1],
                    imu.linear_acceleration[2],
                );
                self.measurements_received += if accepted { 1 } else { 0 };
                if self.measurements_received >= self.measurements_per_orientation {
                    println!(" Done.");
                    self.state = CalibState::SWITCHING;
                }
            }
            CalibState::COMPUTING => {
                println!("Computing calibration parameters...");
                if self.calib.compute_calib() {
                    println!(" Success!");

                    println!("Saving calibration file...");
                    let path = self.calib.save_calib(&self.output_file)?;
                    println!("Calibration saved to {}", path.display());
                } else {
                    println!("Calibration Failed.");
                }
                self.state = CalibState::DONE;
            }
            CalibState::DONE => {}
        }

        Ok(())
    }

    pub fn is_done(&self) -> bool {
        matches!(self.state, CalibState::DONE)
    }
}

#[derive(Debug)]
pub struct CalibrationApplication {
    calibrate_gyros: bool,
    printed_gyro_calib_warning: bool,
    gyro_calib_samples: usize,
    gyro_sample_count: usize,
    gyro_bias: [f64; 3],
    process: CalibrateProcess,
}

impl CalibrationApplication {
    pub fn create(
        calibrate_gyros: bool,
        gyro_calib_samples: usize,
        output_file: String,
        reference_acceleration: f64,
    ) -> anyhow::Result<Self> {
        let gyro_bias = [0.0, 0.0, 0.0];
        let gyro_sample_count = 0;
        let mut process = CalibrateProcess::new(
            gyro_calib_samples,
            reference_acceleration,
            output_file.clone(),
        );
        process.calib.load_calib(output_file)?;

        Ok(Self {
            calibrate_gyros,
            gyro_calib_samples,
            gyro_sample_count,
            gyro_bias,
            process,
            printed_gyro_calib_warning: false,
        })
    }

    pub fn imu_callback(&mut self, mut imu: ImuMsg) -> Option<ImuMsg> {
        if self.calibrate_gyros {
            if !self.printed_gyro_calib_warning {
                println!("Calibrating gyros - Do not move the IMU");
                self.printed_gyro_calib_warning = true;
            }

            self.gyro_sample_count += 1;
            self.gyro_bias[0] = ((self.gyro_sample_count - 1) as f64 * self.gyro_bias[0]
                + imu.angular_velocity[0])
                / self.gyro_sample_count as f64;
            self.gyro_bias[1] = ((self.gyro_sample_count - 1) as f64 * self.gyro_bias[1]
                + imu.angular_velocity[1])
                / self.gyro_sample_count as f64;
            self.gyro_bias[2] = ((self.gyro_sample_count - 1) as f64 * self.gyro_bias[2]
                + imu.angular_velocity[2])
                / self.gyro_sample_count as f64;

            if self.gyro_sample_count >= self.gyro_calib_samples {
                println!(
                    "Gyro calibration complete! (bias = [{}, {}, {}])",
                    self.gyro_bias[0], self.gyro_bias[1], self.gyro_bias[2]
                );
                self.calibrate_gyros = false;
            }

            return None;
        }

        let corrected_accel = self.process.calib.apply_calib(imu.linear_acceleration);
        imu.linear_acceleration = corrected_accel;

        imu.angular_velocity[0] -= self.gyro_bias[0];
        imu.angular_velocity[1] -= self.gyro_bias[1];
        imu.angular_velocity[2] -= self.gyro_bias[2];

        Some(imu)
    }
}
