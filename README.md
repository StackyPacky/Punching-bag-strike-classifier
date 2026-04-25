# Punch Detector
VIDEO DEMO: https://drive.google.com/file/d/1ubXF2s9OXG1rroEfGz7wi9pADXxgrA-R/view?usp=sharing
A real-time punch classification system built for a heavy bag using Arduino, piezo sensors, an accelerometer, Python, and a machine learning model.

The goal of this project is to detect different strikes on a punching bag, classify them live, and display the result in a simple desktop UI.

## Parts Used

- Arduino Uno R3
- 3 piezo sensors
- MPU6050 accelerometer
- Breadboard
- Jumper wires / extended sensor wires (thermostat wire)
- 1M Resistors for the piezo sensors
- Heavy bag mounting setup(duct tape)
- Windows laptop running Python

## How It Works

The Arduino reads impact data from three piezo sensors and motion data from the MPU6050 accelerometer.

When the bag is hit, the sensor values are sent over serial to the computer. Python groups the raw sensor readings into punch events, extracts useful features from each event, and uses a trained machine learning model to classify the punch.

The UI then displays the detected punch live.

## Files

### `sketch_apr8a3piezo1acc`

Arduino code for reading the sensors.

It collects:
- Piezo sensor 1
- Piezo sensor 2
- Piezo sensor 3
- Accelerometer X, Y, and Z values

It sends the data to the computer through serial.

### `read_arduino.py`

Reads live serial data from the Arduino.

It:
- Receives sensor data
- Groups rows into punch events
- Extracts features from each punch
- Saves punch data
- Runs the trained model to predict the punch type

### `train_model.py`

Trains the machine learning model.

It:
- Loads the collected punch feature data
- Trains a classifier
- Tests accuracy
- Saves the trained model as a `.pkl` file

### `punch_ui.py`

Desktop interface for the punch detector.

It:
- Shows the latest detected punch
- Displays session time
- Tracks recent punches
- Gives a cleaner way to use the system live

## Current Punch Classes

The current model is trained to detect punches such as:

- Jab
- Cross
- Left hook
- Right hook

More strikes can be added by collecting more labeled data and retraining the model.


### `punch_features_3piezo.csv`

This file contains the processed training data used for the machine learning model.

Each row represents one punch event (not raw sensor data). Instead of individual readings, it stores extracted features such as:

- Total and peak values from each piezo sensor  
- Relative contribution of each sensor (helps determine punch direction)  
- Differences and ratios between sensors  
- Timing and duration of the punch  
- Some accelerometer-based values  

This condensed format allows the model to learn patterns in how different punches behave, rather than relying on raw sensor data.
## Project Status

This project is still in development. The system works, but accuracy depends heavily on the amount and quality of training data. Future improvements may include:

- More punch data
- Messy/swaying bag training data
- Uppercuts and kicks
- Better combo detection
- Improved power scoring
- Cleaner UI and audio feedback

## Why I Built This

I built this project to combine boxing, electronics, Python, and machine learning into a real physical system. Instead of just detecting that the bag was hit, the goal is to understand what kind of punch was thrown in real time.

## Important

You will likely need your own training data because your punching bag is different, so attach your sensors at the top half of the bag(back, left 45deg and right 45deg), and collect your own data. Once it's in the CSV, just replace "unlabeled" with the strike thrown and run train_model.py
