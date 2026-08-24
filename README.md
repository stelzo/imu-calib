# imu-calib

Kalibr-compatible IMU intrinsic calibration for Rust: estimate the parameters, then apply them online.

Implements the model from Rehder et al., "Extending Kalibr: Calibrating the Extrinsics of Multiple IMUs and of Individual Axes", ICRA 2016, and reads and writes the YAML files [Kalibr](https://github.com/ethz-asl/kalibr) uses.

## Noise parameters

Record at least 3 hours with the IMU undisturbed, away from vibration.

```rust
use imu_calib::allan::{AllanConfig, AllanEstimator};

let mut estimator = AllanEstimator::new(AllanConfig::default());

// for every sample of the recording:
estimator.push(timestamp_seconds, raw_accel, raw_gyro);

let result = estimator.finish()?;
let noise = result.noise()?;

println!("accel noise density: {}", noise.accel_noise_density);
println!("gyro random walk:    {}", noise.gyro_random_walk);

std::fs::write("allan.csv", result.to_csv_string())?;
```

## Intrinsics

Needs no camera and no target. Hold the IMU still in at least nine well-spread
orientations, rotating about a different axis between each one. Start at rest:
the first seconds set the noise floor that tells a hold from a move.

`GuidedCalibration` prompts through it live:

```rust
use imu_calib::estimate::{GuidedCalibration, GuidedConfig, Status};

let mut session = GuidedCalibration::new(GuidedConfig::default());

// for every incoming sample:
match session.push(t, raw_accel, raw_gyro) {
    Status::Initialising { collected, total } => { /* hold still */ }
    Status::WaitingForMotion { next_pose, total } => { /* pick it up and turn it */ }
    Status::WaitingForStillness { .. }           => { /* put it down */ }
    Status::Collecting { collected, total, .. }  => { /* keep holding */ }
    Status::PoseCaptured { pose, total }         => { /* one down */ }
    Status::Done => {
        let calibration = session.finish()?;
        println!("{}", calibration.report.summary());
        calibration.intrinsics.save_kalibr_yaml("imu.yaml")?;
    }
}
```

## Apply

```rust
use imu_calib::ImuIntrinsics;

let intrinsics = ImuIntrinsics::from_kalibr_yaml("imu.yaml")?;
let corrector = intrinsics.corrector()?;   // build once

// per sample:
let (accel, gyro) = corrector.correct(raw_accel, raw_gyro);
```

## Kalibr interop

`from_kalibr_yaml` reads Kalibr's `imu-<bagname>.yaml` directly, including the
`imu0:` / `imu1:` nesting and every key of all three `--imu-models`:

```rust
let intrinsics = ImuIntrinsics::from_kalibr_yaml("imu-dynamic.yaml")?;
let second     = ImuIntrinsics::from_kalibr_yaml_named("imu-dynamic.yaml", "imu1")?;
let all        = ImuIntrinsics::all_from_kalibr_yaml_str(&yaml)?;
```

Files written by this crate use Kalibr's layout, so the two are interchangeable.

To feed a Kalibr run, write the flat input file after the Allan step:

```rust
intrinsics.save_kalibr_input_yaml("imu0.yaml")?;
```

```bash
rosrun kalibr kalibr_calibrate_imu_camera --target april_6x6_80x80cm.yaml --imu imu0.yaml --imu-models scale-misalignment --cam camchain.yaml --bag dynamic.bag
```

### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>
