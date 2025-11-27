#!/home/mecha/venvs/tflite/bin/python
# image_classification_test.py

import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ---------------- Model & Camera Setup ----------------
IMG = 240
color_model_path = "./color.tflite"
direction_model_path = "./direction.tflite"
color_labels = ["green", "red"]
direction_labels = ["forward", "left", "right"]

# Load color model
color_interpreter = tflite.Interpreter(model_path=color_model_path)
color_interpreter.allocate_tensors()
color_inp = color_interpreter.get_input_details()[0]
color_out = color_interpreter.get_output_details()[0]

# Load direction model
direction_interpreter = tflite.Interpreter(model_path=direction_model_path)
direction_interpreter.allocate_tensors()
direction_inp = direction_interpreter.get_input_details()[0]
direction_out = direction_interpreter.get_output_details()[0]

# Camera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# ---------------- GPIO Setup ----------------
DIR_PIN = 16
PWM_PIN = 12
SERVO_PIN = 13

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

SPEED_FORWARD = 65
SPEED_MIN = 50

GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# ---------------- Control Functions ----------------
def set_servo_angle(degree):
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.05)
    servo_pwm.ChangeDutyCycle(0)

def move_forward(speed):
    duty = max(0.0, min(100.0, speed))
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(duty)

def stop_motor():
    motor_pwm.ChangeDutyCycle(0)

# ---------------- Image Processing ----------------
def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

def get_color_prediction(frame):
    x = preprocess_frame(frame)
    color_interpreter.set_tensor(color_inp["index"], x)
    color_interpreter.invoke()
    probs = color_interpreter.get_tensor(color_out["index"])[0]
    pred_id = int(np.argmax(probs))
    confidence = probs[pred_id]
    return color_labels[pred_id], confidence

def get_direction_prediction(frame):
    x = preprocess_frame(frame)
    direction_interpreter.set_tensor(direction_inp["index"], x)
    direction_interpreter.invoke()
    probs = direction_interpreter.get_tensor(direction_out["index"])[0]
    pred_id = int(np.argmax(probs))
    confidence = probs[pred_id]
    return direction_labels[pred_id], confidence

# ---------------- Main Control Loop ----------------
try:
    print("Starting line tracking with color detection...")
    state = "TRACKING"  # States: TRACKING, RED_STOP, GREEN_GO
    red_detected_time = None

    frame_count = 0
    COLOR_CHECK_INTERVAL = 3  # 3프레임마다 색상 체크

    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Get predictions from both models
        
        direction_pred, direction_conf = get_direction_prediction(frame)

        if frame_count % COLOR_CHECK_INTERVAL == 0:
            color_pred, color_conf = get_color_prediction(frame)
        frame_count += 1

        print(f"Color: {color_pred}({color_conf:.2f}) | Direction: {direction_pred}({direction_conf:.2f}) | State: {state}")

        # State machine
        if state == "TRACKING":
            # Check for red color
            if color_pred == "red" and color_conf > 0.7:
                print("[RED DETECTED] Stopping...")
                stop_motor()
                set_servo_angle(90)
                state = "RED_STOP"
                red_detected_time = time.time()
            else:
                # Normal line tracking based on direction
                if direction_pred == "forward":
                    set_servo_angle(90)
                    move_forward(SPEED_FORWARD)
                elif direction_pred == "left":
                    set_servo_angle(60)
                    move_forward(SPEED_MIN)
                elif direction_pred == "right":
                    set_servo_angle(120)
                    move_forward(SPEED_MIN)

        elif state == "RED_STOP":
            # Wait for green color
            if color_pred == "green" and color_conf > 0.7:
                print("[GREEN DETECTED] Moving forward for 1 second...")
                state = "GREEN_GO"
                move_forward(SPEED_FORWARD)
                set_servo_angle(90)
                time.sleep(1.0)
                stop_motor()
                print("[MISSION COMPLETE] Stopped.")
                break
            else:
                # Stay stopped
                stop_motor()
                set_servo_angle(90)

        # Small delay for control loop
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

finally:
    print("Cleaning up...")
    stop_motor()
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    print("Done.")