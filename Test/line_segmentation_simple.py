#!/home/mecha/venvs/tflite/bin/python
# line_segmentation_simple.py
# 최소 기능만 있는 단순 라인 트레이싱

import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ---------------- 설정 ----------------
IMG_SIZE = 160
MODEL_PATH = "./line_segmentation_light.tflite"

# GPIO
SERVO_PIN = 13
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

DIR_PIN = 16
PWM_PIN = 12
MOTOR_FREQ = 1000
SPEED = 60  # 일정 속도

print("="*60)
print("단순 라인 트레이싱 테스트")
print("="*60)

# ---------------- GPIO 초기화 ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

def set_servo_angle(angle):
    """서보 각도 설정 (45~135도)"""
    angle = max(45, min(135, angle))
    duty = SERVO_MIN_DUTY + (angle - 45) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 90.0
    servo_pwm.ChangeDutyCycle(duty)
    return angle

def set_motor(speed):
    """모터 속도 설정"""
    GPIO.output(DIR_PIN, GPIO.HIGH)  # 전진
    motor_pwm.ChangeDutyCycle(speed)

# ---------------- TFLite 모델 로드 ----------------
print("\n[INFO] Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

is_int8 = (input_details['dtype'] == np.uint8)
print(f"✓ Model loaded (INT8: {is_int8})")

# ---------------- 카메라 초기화 ----------------
print("\n[INFO] Initializing camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)
print("✓ Camera ready")

# 초기 서보 중앙
set_servo_angle(90)
time.sleep(0.5)

print("\n[START] Press Ctrl+C to stop")
print("-"*60)

# ---------------- 메인 루프 ----------------
try:
    frame_count = 0
    start_time = time.time()

    while True:
        # 1. 프레임 캡처
        frame = picam2.capture_array()

        # 2. 전처리 (중앙 160x160 크롭)
        h, w = frame.shape[:2]
        y_start = (h - IMG_SIZE) // 2
        x_start = (w - IMG_SIZE) // 2
        cropped = frame[y_start:y_start+IMG_SIZE, x_start:x_start+IMG_SIZE]

        # 3. 모델 추론
        if is_int8:
            input_data = cropped.astype(np.uint8)
        else:
            input_data = cropped.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])

        # 4. 출력 처리
        mask = output_data[0].squeeze()

        # INT8 역양자화
        if output_details['dtype'] == np.uint8 or output_details['dtype'] == np.int8:
            quant = output_details.get('quantization_parameters', {})
            scale = quant.get('scales', [1.0])[0]
            zero_point = quant.get('zero_points', [0])[0]
            mask = (mask.astype(np.float32) - zero_point) * scale
            mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -10, 10)))

        # 이진화
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # 5. 라인 중심 계산
        line_pixels = np.sum(mask_binary > 0)

        if line_pixels > 50:
            moments = cv2.moments(mask_binary)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                line_center_x = cx / IMG_SIZE  # 0~1 범위

                # 6. 단순 비례 제어
                # 중심에서 벗어난 정도 (-1 ~ +1)
                error = (line_center_x - 0.5) * 2

                # 서보 각도 계산 (단순 비례)
                # 라인이 오른쪽(error > 0) → 오른쪽으로 회전(angle > 90)
                # 라인이 왼쪽(error < 0) → 왼쪽으로 회전(angle < 90)
                servo_angle = 90 + (error * 35)  # 비례 상수 35
                servo_angle = max(45, min(135, servo_angle))

                # 7. 서보 제어
                set_servo_angle(servo_angle)

                # 8. 모터 제어 (일정 속도)
                set_motor(SPEED)

                # 9. 출력
                if frame_count % 5 == 0:
                    print(f"Line: Pos={line_center_x:.2f} | Error={error:+.2f} | Servo={servo_angle:.1f}°")
            else:
                # 라인 없음
                set_servo_angle(90)
                set_motor(0)
        else:
            # 라인 없음
            set_servo_angle(90)
            set_motor(0)
            if frame_count % 10 == 0:
                print("No line detected")

        frame_count += 1

        # FPS 계산
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[FPS: {fps:.1f}]")

except KeyboardInterrupt:
    print("\n\n[INFO] Stopping...")

finally:
    # 정지
    set_servo_angle(90)
    set_motor(0)
    time.sleep(0.3)

    # 정리
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
    picam2.stop()

    # 통계
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print("="*60)
    print("[INFO] Cleanup complete")
