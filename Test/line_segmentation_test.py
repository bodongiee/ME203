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
IMG_SIZE = 160  # 경량 segmentation 모델용 (240 → 160)
COLOR_SIZE = 64  # 경량 color 모델용 (240 → 64)
segmentation_model_path = "./line_segmentation_light.tflite"  # 경량 모델
color_model_path = "./color_light.tflite"  # 경량 색상 모델
color_labels = ["green", "red"]

# Load segmentation model
seg_interpreter = tflite.Interpreter(model_path=segmentation_model_path)
seg_interpreter.allocate_tensors()
seg_inp = seg_interpreter.get_input_details()[0]
seg_out = seg_interpreter.get_output_details()[0]

# INT8 양자화 모델 확인
seg_is_int8 = (seg_inp['dtype'] == np.uint8)
print(f"[INFO] Segmentation model INT8: {seg_is_int8}")
print(f"  Input dtype: {seg_inp['dtype']}, Output dtype: {seg_out['dtype']}")

# Load color model
color_interpreter = tflite.Interpreter(model_path=color_model_path)
color_interpreter.allocate_tensors()
color_inp = color_interpreter.get_input_details()[0]
color_out = color_interpreter.get_output_details()[0]

color_is_int8 = (color_inp['dtype'] == np.uint8)
print(f"[INFO] Color model INT8: {color_is_int8}")
print(f"  Input dtype: {color_inp['dtype']}, Output dtype: {color_out['dtype']}")

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
    def __init__(self, kp=1.8, ki=0.02, kd=1.0):
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

pid_controller = PIDController(kp=1.2, ki=0.01, kd=0.6)  # 곡률 대응 강화

# ---------------- Angle Smoothing ----------------
from collections import deque

angle_buffer = deque(maxlen=5)  # 최근 5개 각도 저장

def smooth_angle(current_angle, buffer):
    """
    각도 스무딩 (0°/180° 점프 방지)

    핵심 원리:
    - 0°와 180°는 같은 방향 (수평선)
    - 큰 점프 발생 시 각도를 통일된 범위로 변환
    - 모든 버퍼 값을 같은 기준으로 유지
    """

    if len(buffer) == 0:
        buffer.append(current_angle)
        return current_angle

    # 이전 스무딩된 각도 (버퍼의 평균)
    prev_smoothed = sum(buffer) / len(buffer)

    # 현재 각도와 이전 평균의 차이
    diff = abs(current_angle - prev_smoothed)

    # 180도 점프 감지 및 보정
    if diff > 90:
        # 0° 근처와 180° 근처 중 어느 쪽에 가까운지 판단
        if current_angle < 90:
            # 현재 0° 근처, 이전이 180° 근처라면
            # → 현재를 180° 영역으로 변환
            current_angle += 180
        else:
            # 현재 180° 근처, 이전이 0° 근처라면
            # → 버퍼 전체를 180° 영역으로 변환
            for i in range(len(buffer)):
                if buffer[i] < 90:
                    buffer[i] += 180

    # 버퍼에 추가
    buffer.append(current_angle)

    # 이동 평균 계산
    smoothed = sum(buffer) / len(buffer)

    # 180° 이상이면 0~180 범위로 정규화
    while smoothed >= 180:
        smoothed -= 180

    return smoothed

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
def preprocess_frame(frame, target_size=IMG_SIZE, is_int8=False):
    """프레임을 모델 입력 형식으로 변환"""
    img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
    if is_int8:
        # INT8 양자화 모델: 0-255 uint8 그대로
        img = img.astype(np.uint8)
    else:
        # FLOAT32 모델: 0-1 정규화
        img = img.astype(np.float32) / 255.0
    return img[None, ...]

def get_line_mask(frame):
    """Segmentation 모델로 라인 마스크 얻기"""
    x = preprocess_frame(frame, target_size=IMG_SIZE, is_int8=seg_is_int8)
    seg_interpreter.set_tensor(seg_inp["index"], x)
    seg_interpreter.invoke()
    mask = seg_interpreter.get_tensor(seg_out["index"])[0]

    # 출력이 (160, 160, 1) 형태라고 가정
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    else:
        mask = mask.squeeze()

    # INT8 양자화 출력 역양자화
    if seg_out['dtype'] == np.uint8 or seg_out['dtype'] == np.int8:
        output_quant = seg_out.get('quantization_parameters', {})
        output_scale = output_quant.get('scales', [1.0])[0]
        output_zero_point = output_quant.get('zero_points', [0])[0]

        # 역양자화 + Sigmoid
        mask = (mask.astype(np.float32) - output_zero_point) * output_scale
        mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -10, 10)))

    # 확률값을 0/1 마스크로 변환
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    return binary_mask

def get_color_prediction(frame):
    """색상 검출 (경량 모델 사용)"""
    x = preprocess_frame(frame, target_size=COLOR_SIZE, is_int8=color_is_int8)  # 64x64로 리사이즈
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

    # ROI 설정: 화면 전체 사용
    roi_start = 0
    roi_mask = mask

    # 라인 픽셀 수 확인
    line_pixels = np.sum(roi_mask > 0)
    total_pixels = roi_mask.shape[0] * roi_mask.shape[1]
    confidence = line_pixels / total_pixels

    # 최소 픽셀 임계값 (매우 낮게 설정하여 부분 검출도 허용)
    if line_pixels < 50:  # 50픽셀만 있어도 추적 시도
        return False, 0.5, 0, confidence

    # Look-ahead: 화면을 상/하로 나누어 가중 평균
    # 상단(먼 곳) 70% + 하단(가까운 곳) 30% 가중치
    height = roi_mask.shape[0]
    upper_roi = roi_mask[:height//2, :]  # 상단 절반
    lower_roi = roi_mask[height//2:, :]  # 하단 절반

    # 상단 라인 중심 (Look-ahead)
    upper_moments = cv2.moments(upper_roi)
    upper_cx = 0.5 * width  # 기본값
    if upper_moments['m00'] > 0:
        upper_cx = upper_moments['m10'] / upper_moments['m00']

    # 하단 라인 중심 (현재 위치)
    lower_moments = cv2.moments(lower_roi)
    lower_cx = 0.5 * width  # 기본값
    if lower_moments['m00'] > 0:
        lower_cx = lower_moments['m10'] / lower_moments['m00']

    # 가중 평균: 상단 70% + 하단 30% (먼 곳을 더 중요하게)
    cx = upper_cx * 0.7 + lower_cx * 0.3

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

            # 각도 스무딩 적용 (0°/180° 점프 방지)
            line_angle = smooth_angle(line_angle, angle_buffer)

    return True, line_center_x, line_angle, confidence

def calculate_steering_angle(line_center_x, line_angle, frame_width=IMG_SIZE):

    center_error = (line_center_x - 0.5) * 2  # -1 ~ 1 범위

    correction = pid_controller.update(center_error)

    # 서보 각도 계산
    # center_error > 0 (라인이 오른쪽) → servo > 90 (오른쪽 회전)
    # center_error < 0 (라인이 왼쪽) → servo < 90 (왼쪽 회전)
    servo_angle = 90 + (correction * 25)  # 곡률 대응을 위해 증가  


    angle_deviation = line_angle - 90

    # 수평선 근처(0°/180°)는 각도 정보가 불안정하므로 무시
    # ±60도 이상 벗어나면(30도 미만 또는 150도 초과) 각도 보정 안 함
    if abs(angle_deviation) > 60:
        # 수평선에 가까움 → 각도 보정 사용 안 함
        angle_correction = 0
    else:
        # 수직선에 가까움 (30~150도 범위)
        # angle_deviation이 양수(135도 방향) → servo 감소(왼쪽)
        # angle_deviation이 음수(45도 방향) → servo 증가(오른쪽)

        # 급커브 대응: 각도 편차가 클수록 더 강하게 반응
        angle_weight = 0.08 + (abs(angle_deviation) / 60.0) * 0.12  # 0.08~0.20 동적 조정
        angle_correction = angle_deviation * angle_weight
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
