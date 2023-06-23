# Control-By-Face
Control a robot by the shape of a face.

```bash
git clone https://github.com/sasekazu/control-by-face.git --recursive
```

# Dependencies
- dlib
- cereal
- OpenCV
- nana
- openFrameworks (for simulator)
- xArm-CPLUS-SDK (for controller)

# Apps
## Collector
Collect face images from a webcam with labels.
This exports collected data in `out.cereal`.
You need to put `shape_predictor_68_face_landmarks.dat` in the same directory.

## Trainer
Train a model to predict control signals from face images.
This reads `out.cereal` and exports a trained model in `decision_function.dat`.

## Predictor
Predict control signals from face images.
This reads `decision_function.dat`.
You can check the predicted control signals on a console.

## Simulator
Simulate a robot with the predicted control signals.
A robot end-effector is modeled as a sphere.
This requires openFrameworks.
Include `training_data_raw.h` in a openFrameworks project.
Replace `ofApp.h` and `ofApp.cpp` with these of this repository.

## Controller
Control a robot (Ufactory Lite6) with the predicted control signals.
This requires [xArm-CPLUS-SDK](https://github.com/xArm-Developer/xArm-CPLUS-SDK).
In an example lf xArm-CPLUS-SDK, such as `1009-cartesian_velocity_control.cc`, add an include path to `traning_data_raw.h` and replace a source file with `controller.cpp` of this repository.
