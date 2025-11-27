#!/home/mecha/venvs/tflite/bin/python
# step1_line_segmentation_simple.py
# 빨간불 정지 → 초록불 전진 후 종료

import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ---------------- 설정 ----------------
IMG_SIZE = 160
COLOR_SIZE = 64
MODEL_PATH = "./line_segmentation_light.tflite"
COLOR_MODEL_PATH = "./color_light.tflite"

# GPIO
SERVO_PIN = 13
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

DIR_PIN = 16
PWM_PIN = 12
MOTOR_FREQ = 1000
SPEED = 25  # 일정 속도

print("="*60)
print("Step 1: 빨간불 정지 → 초록불 전진")
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
print("\n[INFO] Loading segmentation model...")
seg_interpreter = tflite.Interpreter(model_path=MODEL_PATH)
seg_interpreter.allocate_tensors()

seg_input_details = seg_interpreter.get_input_details()[0]
seg_output_details = seg_interpreter.get_output_details()[0]

seg_is_int8 = (seg_input_details['dtype'] == np.uint8)
print(f"✓ Segmentation model loaded (INT8: {seg_is_int8})")

print("\n[INFO] Loading color model...")
color_interpreter = tflite.Interpreter(model_path=COLOR_MODEL_PATH)
color_interpreter.allocate_tensors()

color_input_details = color_interpreter.get_input_details()[0]
color_output_details = color_interpreter.get_output_details()[0]

color_is_int8 = (color_input_details['dtype'] == np.uint8)
print(f"✓ Color model loaded (INT8: {color_is_int8})")

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

print("\n[START] Tracking line until RED light...")
print("-"*60)

# ---------------- 상태 변수 ----------------
state = "TRACKING"  # TRACKING, WAITING_GREEN, MOVING_FORWARD, DONE

# ---------------- 메인 루프 ----------------
try:
    frame_count = 0
    start_time = time.time()
    green_light_time = None

    while True:
        # 1. 프레임 캡처
        frame = picam2.capture_array()

        # 2-1. 색상 감지용 전처리 (64x64 크롭)
        h, w = frame.shape[:2]
        y_color = (h - COLOR_SIZE) // 2
        x_color = (w - COLOR_SIZE) // 2
        color_cropped = frame[y_color:y_color+COLOR_SIZE, x_color:x_color+COLOR_SIZE]

        if color_is_int8:
            color_input = color_cropped.astype(np.uint8)
        else:
            color_input = color_cropped.astype(np.float32) / 255.0
        color_input = np.expand_dims(color_input, axis=0)

        color_interpreter.set_tensor(color_input_details['index'], color_input)
        color_interpreter.invoke()
        color_output = color_interpreter.get_tensor(color_output_details['index'])[0]

        # INT8 역양자화
        if color_output_details['dtype'] == np.uint8 or color_output_details['dtype'] == np.int8:
            color_quant = color_output_details.get('quantization_parameters', {})
            color_scale = color_quant.get('scales', [1.0])[0]
            color_zero = color_quant.get('zero_points', [0])[0]
            color_output = (color_output.astype(np.float32) - color_zero) * color_scale

        # Softmax
        exp_vals = np.exp(color_output - np.max(color_output))
        color_probs = exp_vals / np.sum(exp_vals)
        color_idx = np.argmax(color_probs)
        color_conf = color_probs[color_idx]
        color_name = ["green", "red"][color_idx]

        # 2-2. 라인 감지용 전처리 (160x160 크롭)
        y_start = (h - IMG_SIZE) // 2
        x_start = (w - IMG_SIZE) // 2
        cropped = frame[y_start:y_start+IMG_SIZE, x_start:x_start+IMG_SIZE]

        # 3. 세그멘테이션 모델 추론
        if seg_is_int8:
            input_data = cropped.astype(np.uint8)
        else:
            input_data = cropped.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        seg_interpreter.set_tensor(seg_input_details['index'], input_data)
        seg_interpreter.invoke()
        output_data = seg_interpreter.get_tensor(seg_output_details['index'])

        # 4. 출력 처리
        mask = output_data[0].squeeze()

        # INT8 역양자화
        if seg_output_details['dtype'] == np.uint8 or seg_output_details['dtype'] == np.int8:
            quant = seg_output_details.get('quantization_parameters', {})
            scale = quant.get('scales', [1.0])[0]
            zero_point = quant.get('zero_points', [0])[0]
            mask = (mask.astype(np.float32) - zero_point) * scale
            mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -10, 10)))

        # 이진화
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # 테두리 제거 (10픽셀씩 잘라내기)
        border = 10
        mask_binary[:border, :] = 0  # 상단
        mask_binary[-border:, :] = 0  # 하단
        mask_binary[:, :border] = 0  # 좌측
        mask_binary[:, -border:] = 0  # 우측

        # 5. 상태 머신
        if state == "TRACKING":
            # 빨간불 감지 (신뢰도 > 80%)
            if color_name == "red" and color_conf > 0.8:
                print(f"\nRED LIGHT DETECTED! Stopping...")
                set_servo_angle(90)
                set_motor(0)
                state = "WAITING_GREEN"
                continue

            # 라인 추적
            line_pixels = np.sum(mask_binary > 0)

            if line_pixels > 50:
                moments = cv2.moments(mask_binary)
                if moments['m00'] > 0:
                    cx = moments['m10'] / moments['m00']
                    line_center_x = cx / IMG_SIZE

                    error = (line_center_x - 0.5) * 2
                    servo_angle = 90 + (error * 25)
                    servo_angle = max(45, min(135, servo_angle))

                    set_servo_angle(servo_angle)
                    set_motor(SPEED)

                    if frame_count % 5 == 0:
                        print(f"TRACKING | Pos={line_center_x:.2f} | Servo={servo_angle:.1f}° | Color={color_name}")
                else:
                    set_servo_angle(90)
                    set_motor(0)
            else:
                set_servo_angle(90)
                set_motor(0)

        elif state == "WAITING_GREEN":
            # 초록불 감지 대기
            if color_name == "green" and color_conf > 0.8:
                print(f"\n🟢 GREEN LIGHT DETECTED! Moving forward...")
                set_servo_angle(90)
                set_motor(SPEED)
                green_light_time = time.time()
                state = "MOVING_FORWARD"
            else:
                # 정지 상태 유지
                if frame_count % 10 == 0:
                    print(f"WAITING | Color={color_name}({color_conf:.2f})")

        elif state == "MOVING_FORWARD":
            # 1초 전진
            elapsed = time.time() - green_light_time
            if elapsed >= 1.0:
                print(f"\n✓ Forward complete. Stopping...")
                set_servo_angle(90)
                set_motor(0)
                state = "DONE"
                break
            else:
                # 계속 직진
                set_servo_angle(90)
                set_motor(SPEED)
                if frame_count % 5 == 0:
                    print(f"FORWARD | Time={elapsed:.1f}s")

        frame_count += 1

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
    print("MISSION COMPLETE!")
    print("="*60)
    print(f"Final state: {state}")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print("="*60)
    print("[INFO] Cleanup complete")
