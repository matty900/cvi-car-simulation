# Self-Driving Car Simulation Using CNN
**Final Project**  
Developed for the Udacity Self-Driving Car Simulator using Python, OpenCV, and TensorFlow...

## Project Overview
This project implements a convolutional neural network (CNN) to control the steering angle of a simulated self-driving car. The model predicts steering angles in real-time using images from the car's front-facing camera, enabling autonomous navigation along a predefined track.

I used Udacity self-driving car simulator for data collection and testing. The CNN model was trained on images captured during manual driving and evaluated within the simulator to test its real-world performance in real-time.

## Technologies Used
* **Python**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **scikit-learn**
* **TensorFlow/Keras**
* **Flask + SocketIO**

## Setup Instructions
1. **Create a Virtual Environment:**
   ```
   1. pip install virtualenv
   2. python -m venv car-env
   3. cd car-env
   4. car-env/bin/activate  # On Windows: car-env\Scripts\activate
   ```

2. **Install Dependencies:**
   ```
   pip install -r package_list.txt
   ```

3. **Download the Udacity Simulator:**
   * [Simulator Link](https://github.com/udacity/self-driving-car-sim) ( prefer Version 1 over other versions)

## Data Collection
1. Launch the simulator and select **Training Mode**.
2. Manually drive the car using your keyboard.
3. Save the data by clicking Recording, which generates:
   * A folder of images (`IMG/`)
   * A CSV file (`driving_log.csv`)

Drive several laps in both directions to balance your dataset.

## Training the Model
Run the following to preprocess and train the model:

```
python train_model.py
```

The final model is saved as `model.h5`. A loss graph (`training_plot.png`) is also generated.

## Testing the Model
1. Start the simulator in **Autonomous Mode**.
2. Run the testing script:
   ```
   python TestSimulation.py
   ```
3. The car will start moving using predictions from your trained model.

## Challenges & Solutions

| **Challenge** | **Solution** |
|---------------|--------------|
| Uneven distribution of turning angles | Augmented data with flipped images to balance left/right turns |
| Model underfitting | Increased model complexity by adding convolutional layerst |
| Inconsistent steering behavior | Implemented smoothing function to average consecutive prediction |
| Input image noise | Applied Gaussian blur preprocessing to reduce unwanted features

## Demo
[Link to demo video](https://youtu.be/xxDrQUYasg0)
