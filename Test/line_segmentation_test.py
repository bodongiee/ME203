#!/home/mecha/venvs/tflite/bin/python
# line_segmentation_test.py
# Semantic Segmentation 기반 라인 트레이싱 (낮은 카메라 높이에 최적화)

import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from collections import deque

# ---------------- Model & Camera Setup ----------------
IMG_SIZE = 240
segmentation_model_path = "./line_segmentation.tflite"
color_model_path = "./color.tflite"
color_labels = ["green", "red"]

# Load segmentation model
seg_interpreter = tflite.Interpreter(model_path=segmentation_model_path)
seg_interpreter.allocate_tensors()
seg_inp = seg_interpreter.get_input_details()[0]
seg_out = seg_interpreter.get_output_details()[0]

# Load color model
color_interpreter = tflite.Interpreter(model_path=color_model_path)
color_interpreter.allocate_tensors()
color_inp = color_interpreter.get_input_details()[0]
color_out = color_interpreter.get_output_details()[0]

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
SPEED_TURN = 50

GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# ---------------- PID Controller ----------------
class PIDController:
    def __init__(self, kp=0.46, ki=0.0, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt=0.05):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -50, 50)  # Anti-windup

        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0
        self.integral = 0

pid_controller = PIDController(kp=1.8, ki=0.02, kd=1.0)

# ---------------- Temporal Smoothing ----------------
# 이전 프레임의 라인 위치를 기억하여 일시적 손실 대비
line_position_history = deque(maxlen=5)
line_detected_history = deque(maxlen=3)

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
    """프레임을 모델 입력 형식으로 변환"""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

def get_line_mask(frame):
    """Segmentation 모델로 라인 마스크 얻기"""
    x = preprocess_frame(frame)
    seg_interpreter.set_tensor(seg_inp["index"], x)
    seg_interpreter.invoke()
    mask = seg_interpreter.get_tensor(seg_out["index"])[0]

    # 출력이 (240, 240, 1) 형태라고 가정
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # 확률값을 0/1 마스크로 변환
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    return binary_mask

def get_color_prediction(frame):
    """색상 검출 (기존 모델 사용)"""
    x = preprocess_frame(frame)
    color_interpreter.set_tensor(color_inp["index"], x)
    color_interpreter.invoke()
    probs = color_interpreter.get_tensor(color_out["index"])[0]
    pred_id = int(np.argmax(probs))
    confidence = probs[pred_id]
    return color_labels[pred_id], confidence

def analyze_line_mask(mask):
    """
    마스크에서 라인의 위치와 방향을 분석

    Returns:
        line_detected (bool): 라인이 검출되었는지
        line_center_x (float): 라인 중심의 x 좌표 (0~1 정규화)
        line_angle (float): 라인의 각도 (도 단위)
        confidence (float): 검출 신뢰도
    """
    height, width = mask.shape

    # ROI 설정: 화면 하단 60%만 사용 (카메라가 낮으므로)
    roi_start = int(height * 0.4)
    roi_mask = mask[roi_start:, :]

    # 라인 픽셀 수 확인
    line_pixels = np.sum(roi_mask > 0)
    total_pixels = roi_mask.shape[0] * roi_mask.shape[1]
    confidence = line_pixels / total_pixels

    # 최소 픽셀 임계값 (매우 낮게 설정하여 부분 검출도 허용)
    if line_pixels < 50:  # 50픽셀만 있어도 추적 시도
        return False, 0.5, 0, confidence

    # Moments를 이용한 라인 중심 계산
    moments = cv2.moments(roi_mask)

    if moments['m00'] == 0:
        return False, 0.5, 0, confidence

    # 라인 중심점
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    # 정규화 (0~1 범위)
    line_center_x = cx / width

    # 라인 각도 계산 (contour fitting)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_angle = 90.0  # 기본값 (직진)

    if contours:
        # 가장 큰 contour 선택
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) >= 5:
            # 타원 피팅으로 방향 추정
            ellipse = cv2.fitEllipse(largest_contour)
            line_angle = ellipse[2]  # 각도

            # 각도 보정 (-90~90 범위를 0~180으로 변환)
            if line_angle < 0:
                line_angle += 180

    return True, line_center_x, line_angle, confidence

def calculate_steering_angle(line_center_x, line_angle, frame_width=IMG_SIZE):
    """
    라인 위치와 각도로부터 조향각 계산

    Args:
        line_center_x: 라인 중심 x 좌표 (0~1 정규화)
        line_angle: 라인 각도 (도 단위)

    Returns:
        servo_angle: 서보 모터 각도 (45~135)
    """
    # 화면 중심에서의 오차 계산
    center_error = (line_center_x - 0.5) * 2  # -1 ~ 1 범위

    # PID 제어로 조향각 계산
    correction = pid_controller.update(center_error)

    # 기본 각도 90도(직진)에서 보정값 추가
    # center_error > 0: 라인이 오른쪽 -> 왼쪽으로 회전 필요 (각도 감소)
    # center_error < 0: 라인이 왼쪽 -> 오른쪽으로 회전 필요 (각도 증가)
    servo_angle = 90 - (correction * 30)  # 최대 ±30도 보정

    # 라인 각도도 반영 (추가적인 미세 조정)
    angle_correction = (line_angle - 90) * 0.2
    servo_angle -= angle_correction

    # 범위 제한
    servo_angle = max(45, min(135, servo_angle))

    return servo_angle

def get_estimated_position():
    """
    이전 프레임 정보로 현재 라인 위치 추정
    (라인을 일시적으로 놓쳤을 때 사용)
    """
    if len(line_position_history) == 0:
        return 0.5  # 중앙값 반환

    # 최근 위치들의 가중 평균 (최신 프레임에 더 높은 가중치)
    weights = np.array([0.1, 0.15, 0.25, 0.5])[-len(line_position_history):]
    weights = weights / weights.sum()

    positions = np.array(list(line_position_history))
    estimated = np.average(positions, weights=weights)

    return estimated

# ---------------- Main Control Loop ----------------
try:
    print("Starting segmentation-based line tracking...")
    print("This system works even with low camera height!")

    state = "TRACKING"
    red_detected_time = None
    frame_count = 0
    COLOR_CHECK_INTERVAL = 5

    # 초기에 서보 중앙 정렬
    set_servo_angle(90)
    time.sleep(0.5)

    while True:
        frame = picam2.capture_array()

        # Segmentation으로 라인 검출
        line_mask = get_line_mask(frame)
        line_detected, line_center_x, line_angle, mask_confidence = analyze_line_mask(line_mask)

        # 색상 검출 (주기적으로)
        if frame_count % COLOR_CHECK_INTERVAL == 0:
            color_pred, color_conf = get_color_prediction(frame)
        frame_count += 1

        # 라인 검출 이력 업데이트
        line_detected_history.append(line_detected)

        # Temporal smoothing: 최근 3프레임 중 하나라도 검출되면 추적 가능
        any_detection = any(line_detected_history)

        if line_detected:
            line_position_history.append(line_center_x)

        # 디버깅 정보 출력
        print(f"Line: {'YES' if line_detected else 'NO'} | "
              f"Pos: {line_center_x:.2f} | "
              f"Angle: {line_angle:.1f}° | "
              f"Conf: {mask_confidence:.3f} | "
              f"Color: {color_pred}({color_conf:.2f}) | "
              f"State: {state}")

        # State machine
        if state == "TRACKING":
            # 빨간색 검출 확인
            if color_pred == "red" and color_conf > 0.7:
                print("[RED DETECTED] Stopping...")
                stop_motor()
                set_servo_angle(90)
                state = "RED_STOP"
                red_detected_time = time.time()
                continue

            # 라인 추적
            if line_detected:
                # 직접 검출된 경우
                servo_angle = calculate_steering_angle(line_center_x, line_angle)

                # 신뢰도에 따라 속도 조절
                if mask_confidence > 0.15:
                    speed = SPEED_FORWARD
                else:
                    speed = SPEED_TURN  # 신뢰도 낮으면 천천히

            elif any_detection:
                # 최근에 검출된 적 있으면 추정 위치 사용
                estimated_x = get_estimated_position()
                servo_angle = calculate_steering_angle(estimated_x, 90)
                speed = SPEED_TURN  # 천천히 이동
                print(f"[ESTIMATED] Using position: {estimated_x:.2f}")

            else:
                # 완전히 라인을 놓친 경우: 제자리에서 탐색
                print("[LOST] Searching for line...")
                stop_motor()

                # 마지막으로 본 방향으로 약간 회전
                if len(line_position_history) > 0:
                    last_pos = line_position_history[-1]
                    if last_pos < 0.5:
                        set_servo_angle(60)  # 왼쪽으로
                    else:
                        set_servo_angle(120)  # 오른쪽으로
                else:
                    set_servo_angle(90)

                time.sleep(0.2)
                continue

            # 모터 제어 실행
            set_servo_angle(servo_angle)
            move_forward(speed)

        elif state == "RED_STOP":
            # 초록불 대기
            if color_pred == "green" and color_conf > 0.7:
                print("[GREEN DETECTED] Moving forward...")
                state = "GREEN_GO"
                move_forward(SPEED_FORWARD)
                set_servo_angle(90)
                time.sleep(1.0)
                stop_motor()
                print("[MISSION COMPLETE]")
                break
            else:
                stop_motor()
                set_servo_angle(90)

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
