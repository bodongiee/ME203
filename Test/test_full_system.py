#!/usr/bin/env python3
# test_full_system.py
# TFLite 모델 + 실제 서보 모터로 전체 시스템 테스트

import time
import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
from collections import deque

# ---------------- 설정 ----------------
TFLITE_MODEL_PATH = "../line_segmentation_light.tflite"
IMG_SIZE = 160  # 경량 모델

# GPIO 설정
SERVO_PIN = 13
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 디스플레이 설정
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# FPS 계산용
frame_count = 0
start_time = time.time()
fps = 0

print("="*60)
print("전체 시스템 테스트 - TFLite + 서보")
print(f"모델: {TFLITE_MODEL_PATH}")
print(f"이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print("="*60)

# ---------------- GPIO Setup ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
servo_pwm.start(0)

def set_servo_angle(angle):
    """서보 각도 설정 (45~135도)"""
    angle = max(45, min(135, angle))
    duty = SERVO_MIN_DUTY + (angle - 45) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 90.0
    servo_pwm.ChangeDutyCycle(duty)
    return angle

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
        self.integral = np.clip(self.integral, -50, 50)

        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0
        self.integral = 0

pid_controller = PIDController(kp=0.8, ki=0.01, kd=0.5)

# ---------------- Angle Smoothing ----------------
angle_buffer = deque(maxlen=5)

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

# ---------------- TFLite 모델 로드 ----------------
print("\n[INFO] Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✓ Model loaded")
print(f"  Input:  {input_details[0]['shape']} {input_details[0]['dtype']}")
print(f"  Output: {output_details[0]['shape']} {output_details[0]['dtype']}")

# INT8 모델 확인
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
is_int8_model = (input_dtype == np.uint8)
print(f"  INT8 quantized: {is_int8_model}")

# 모델 입력 크기 확인
model_input_size = input_details[0]['shape'][1]
if model_input_size != IMG_SIZE:
    print(f"[WARNING] IMG_SIZE ({IMG_SIZE}) != model ({model_input_size})")
    IMG_SIZE = model_input_size
    print(f"[INFO] Updated IMG_SIZE to {IMG_SIZE}")

# ---------------- PiCamera2 초기화 ----------------
print("\n[INFO] Initializing PiCamera2...")
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (IMG_SIZE, IMG_SIZE), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("✓ Camera started")
print("\nControls:")
print("  'q' - Quit")
print("  's' - Save screenshot")
print("  'c' - Toggle control (AUTO/MANUAL)")
print("  Arrow keys - Manual control")
print("-"*60)

time.sleep(1)

# 초기 서보 중앙
set_servo_angle(90)
time.sleep(0.5)

# 제어 모드
auto_control = True

# ---------------- 조향 계산 함수 ----------------
def calculate_steering_angle(line_center_x, line_angle):
    """조향 각도 계산"""
    center_error = (line_center_x - 0.5) * 2
    correction = pid_controller.update(center_error)
    servo_angle = 90 + (correction * 20)  # 30 → 20으로 감소 (반응 완화)

    angle_deviation = line_angle - 90
    if abs(angle_deviation) > 60:
        angle_correction = 0
    else:
        angle_correction = angle_deviation * 0.08  # 0.15 → 0.08로 감소
        servo_angle -= angle_correction

    servo_angle = max(45, min(135, servo_angle))
    return servo_angle, center_error, correction, angle_correction

# ---------------- 메인 루프 ----------------
try:
    manual_servo = 90

    while True:
        loop_start = time.time()

        # 프레임 캡처
        frame = picam2.capture_array()

        # 전처리
        if is_int8_model:
            input_data = frame.astype(np.uint8)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            input_data = frame.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

        # 추론
        inference_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        inference_time = (time.time() - inference_start) * 1000

        # 출력 처리
        mask = output_data[0].squeeze()

        # INT8 역양자화
        if output_dtype == np.uint8 or output_dtype == np.int8:
            output_quant = output_details[0].get('quantization_parameters', {})
            output_scale = output_quant.get('scales', [1.0])[0]
            output_zero_point = output_quant.get('zero_points', [0])[0]
            mask = (mask.astype(np.float32) - output_zero_point) * output_scale
            mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -10, 10)))

        # 이진화
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # ---------------- 라인 분석 ----------------
        line_detected = False
        line_center_x = 0.5
        line_angle = 90.0

        line_pixels = np.sum(mask_binary > 0)

        if line_pixels > 50:
            moments = cv2.moments(mask_binary)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                line_center_x = cx / IMG_SIZE

                # 각도 계산
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 5:
                        ellipse = cv2.fitEllipse(largest_contour)
                        line_angle = ellipse[2]
                        if line_angle < 0:
                            line_angle += 180
                        line_angle = smooth_angle(line_angle, angle_buffer)

                line_detected = True

        # ---------------- 조향 제어 ----------------
        if auto_control and line_detected:
            servo_angle, center_error, correction, angle_correction = calculate_steering_angle(
                line_center_x, line_angle
            )
            actual_servo = set_servo_angle(servo_angle)
        else:
            actual_servo = set_servo_angle(manual_servo)
            center_error = (line_center_x - 0.5) * 2
            correction = 0
            angle_correction = 0

        # ---------------- 시각화 ----------------
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 오버레이
        overlay = frame_bgr.copy()
        green_overlay = np.zeros_like(frame_bgr)
        green_overlay[:, :, 1] = mask_binary
        overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)

        # 라인 중심점
        if line_detected:
            cx_px = int(line_center_x * IMG_SIZE)
            cy_px = IMG_SIZE // 2
            cv2.drawMarker(overlay, (cx_px, cy_px), (0, 0, 255),
                          cv2.MARKER_CROSS, 15, 2)

        # 화면 중앙선
        cv2.line(overlay, (IMG_SIZE//2, 0), (IMG_SIZE//2, IMG_SIZE),
                (0, 255, 255), 1, cv2.LINE_DASHED)

        # FPS 계산
        frame_count += 1
        if frame_count % 5 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed

        # 정보 패널 생성
        info_height = 120
        info_panel = np.zeros((info_height, IMG_SIZE, 3), dtype=np.uint8)

        # FPS (색상)
        fps_color = (0, 255, 0) if fps >= 10 else (0, 255, 255) if fps >= 5 else (0, 0, 255)
        cv2.putText(info_panel, f"FPS: {fps:.1f} | Inf: {inference_time:.1f}ms",
                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_color, 1)

        # 모드
        mode_text = "AUTO" if auto_control else "MANUAL"
        mode_color = (0, 255, 0) if auto_control else (255, 100, 0)
        cv2.putText(info_panel, f"Mode: {mode_text}",
                   (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1)

        # 라인 정보
        if line_detected:
            line_ratio = line_pixels / (IMG_SIZE * IMG_SIZE) * 100
            cv2.putText(info_panel, f"Line: {line_ratio:.1f}% | Pos: {line_center_x:.2f} | Ang: {line_angle:.0f}deg",
                       (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(info_panel, "No line detected",
                       (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 제어 정보
        cv2.putText(info_panel, f"center_err: {center_error:+.2f} | PID: {correction:+.2f} | ang: {angle_correction:+.2f}",
                   (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # 서보 각도
        servo_color = (0, 100, 255) if actual_servo < 85 else (0, 255, 100) if actual_servo > 95 else (100, 255, 255)
        cv2.putText(info_panel, f"SERVO: {actual_servo:.1f} deg",
                   (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, servo_color, 2)

        # 결합
        display = np.vstack([info_panel, overlay])

        # 리사이즈
        aspect = display.shape[1] / display.shape[0]
        new_height = DISPLAY_HEIGHT
        new_width = int(new_height * aspect)
        display = cv2.resize(display, (new_width, new_height))

        # 표시
        cv2.imshow("Full System Test - TFLite + Servo", display)

        # 키 입력
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quit requested")
            break

        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"test_{timestamp}.jpg", overlay)
            cv2.imwrite(f"mask_{timestamp}.jpg", mask_binary)
            print(f"[SAVE] Saved: test_{timestamp}.jpg, mask_{timestamp}.jpg")

        elif key == ord('c'):
            auto_control = not auto_control
            if auto_control:
                pid_controller.reset()
                print("[MODE] AUTO control")
            else:
                manual_servo = actual_servo
                print("[MODE] MANUAL control")

        # Manual control (화살표 키)
        elif not auto_control:
            if key == 81 or key == 2:  # 왼쪽
                manual_servo = max(45, manual_servo - 2)
                print(f"Manual servo: {manual_servo:.0f}°")
            elif key == 83 or key == 3:  # 오른쪽
                manual_servo = min(135, manual_servo + 2)
                print(f"Manual servo: {manual_servo:.0f}°")
            elif key == ord(' '):  # 중앙
                manual_servo = 90
                print("Manual servo: 90° (center)")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    # 서보 중앙으로
    set_servo_angle(90)
    time.sleep(0.3)

    # 정리
    servo_pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()

    # 최종 통계
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average inference: {1000/avg_fps:.1f}ms")

    if avg_fps >= 10:
        print(f"\n✓ SUCCESS: {avg_fps:.1f} FPS - 실시간 추론 가능!")
    elif avg_fps >= 5:
        print(f"\n⚠ OK: {avg_fps:.1f} FPS - 동작 가능")
    else:
        print(f"\n✗ SLOW: {avg_fps:.1f} FPS - 너무 느림")

    print("="*60)
    print("[INFO] Cleanup complete")
