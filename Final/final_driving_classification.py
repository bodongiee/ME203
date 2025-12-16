import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2



IMG_SIZE = 160
COLOR_SIZE = 160
MODEL_PATH = "./line_segmentation_light.tflite"
COLOR_MODEL_PATH = "./color_light_160.tflite"

# GPIO
SERVO_PIN = 13
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

DIR_PIN = 16
PWM_PIN = 12
MOTOR_FREQ = 1000
SPEED = 18
ANGLE_SPEED_GAIN = 5  # 각도에 따른 속도 보정 최대값 (0 이상)

RED_ACCEPTANCE = 40
GREEN_ACCEPTANCE = 50

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

def angle_cor(servo_angle):
    # 90도(직진)에서 0, 45도/135도(최대 회전)에서 최대 보정값
    # 코너에서 마찰 보상을 위해 속도를 높임
    angle_diff = abs(servo_angle - 90)  # 0~45
    speed_correction = ANGLE_SPEED_GAIN * (angle_diff / 45.0)  # 0 ~ GAIN
    return speed_correction

def set_motor(speed, direction="forward"):
    if direction == "forward":
        GPIO.output(DIR_PIN, GPIO.HIGH)
    elif direction == "backward":
        GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)

def detect_color_classification(rgb_image, interpreter, input_details, output_details):

    if input_details['dtype'] == np.uint8:
        input_data = rgb_image.astype(np.uint8)
    else:
        input_data = rgb_image.astype(np.float32) / 255.0

    input_data = np.expand_dims(input_data, axis=0)

    # 추론
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    # 출력 처리 (INT8 역양자화)
    if output_details['dtype'] == np.uint8 or output_details['dtype'] == np.int8:
        quant = output_details.get('quantization_parameters', {})
        scale = quant.get('scales', [1.0])[0]
        zero_point = quant.get('zero_points', [0])[0]
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    # 소프트맥스 적용
    exp_output = np.exp(output_data - np.max(output_data))
    probabilities = exp_output / np.sum(exp_output)
    probabilities = probabilities[0]

    # 클래스 인덱스: 0=green, 1=red (모델 학습 순서에 따라 조정 필요)
    class_idx = np.argmax(probabilities)
    confidence = probabilities[class_idx]

    class_names = ["green", "red"]
    color_name = class_names[class_idx]

    return color_name, confidence


print("\n[INFO] Loading segmentation model...")
seg_interpreter = tflite.Interpreter(model_path=MODEL_PATH)
seg_interpreter.allocate_tensors()

seg_input_details = seg_interpreter.get_input_details()[0]
seg_output_details = seg_interpreter.get_output_details()[0]

seg_is_int8 = (seg_input_details['dtype'] == np.uint8)

print("\n[INFO] Loading color classification model...")
color_interpreter = tflite.Interpreter(model_path=COLOR_MODEL_PATH)
color_interpreter.allocate_tensors()

color_input_details = color_interpreter.get_input_details()[0]
color_output_details = color_interpreter.get_output_details()[0]

color_is_int8 = (color_input_details['dtype'] == np.uint8)

# ---------------- 카메라 초기화 ----------------
print("\n[INFO] Initializing camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)


set_servo_angle(90)

set_motor(20)
time.sleep(0.5)
set_motor(0)


print("-"*60)

state = "TRACKING"  # TRACKING, REVERSING, WAITING_GREEN, MOVING_FORWARD, DONE

try:
    frame_count = 0
    start_time = time.time()
    green_light_time = None
    reverse_start_time = None

    while True:
        # 1. 프레임 캡처
        frame = picam2.capture_array()

        # 2-1. 색상 분류용 전처리 (160x160 크롭)
        h, w = frame.shape[:2]
        y_color = (h - COLOR_SIZE) // 2
        x_color = (w - COLOR_SIZE) // 2
        color_cropped = frame[y_color:y_color+COLOR_SIZE, x_color:x_color+COLOR_SIZE]

        # TFLite 모델로 색상 감지
        color_name, color_conf = detect_color_classification(
            color_cropped, color_interpreter, color_input_details, color_output_details
        )

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

        if state == "TRACKING":
            # 빨간불 감지
            if color_name == "red" and color_conf > 1 - RED_ACCEPTANCE / 100:
                set_servo_angle(90)
                set_motor(SPEED, direction="backward")
                reverse_start_time = time.time()
                state = "REVERSING"
                continue

            # 라인 추적
            line_pixels = np.sum(mask_binary > 0)

            if line_pixels > 50:
                moments = cv2.moments(mask_binary)
                if moments['m00'] > 0:
                    cx = moments['m10'] / moments['m00']
                    line_center_x = cx / IMG_SIZE

                    error = (line_center_x - 0.5) * 2
                    servo_angle = 90 + (error * 22)
                    servo_angle = max(45, min(135, servo_angle))

                    set_servo_angle(servo_angle)
                    set_motor(SPEED)

                    if frame_count % 10 == 0:
                        print(f"TRACKING | Pos={line_center_x:.2f} | Servo={servo_angle:.1f}°")
                else:
                    set_servo_angle(90)
                    set_motor(0)
            else:
                set_servo_angle(90)
                set_motor(0)

        elif state == "REVERSING":
            elapsed = time.time() - reverse_start_time
            if elapsed >= 0.5:  # 0.5초 후진
                set_servo_angle(90)
                set_motor(0)
                state = "WAITING_GREEN"
            else:
                set_servo_angle(90)
                set_motor(SPEED, direction="backward")

        elif state == "WAITING_GREEN":
            if color_name == "green" and color_conf > 1 - GREEN_ACCEPTANCE / 100:
                set_servo_angle(80)
                set_motor(SPEED)
                green_light_time = time.time()
                state = "MOVING_FORWARD"
            else:
                if frame_count % 10 == 0:
                    print(f"WAITING | Color={color_name}({color_conf:.2f})")

        elif state == "MOVING_FORWARD":
            # 1초 전진
            elapsed = time.time() - green_light_time
            if elapsed >= 1.0:
                set_servo_angle(90)
                set_motor(0)
                state = "DONE"
                break
            else:
                # 계속 직진 (서보 80도 유지)
                set_servo_angle(80)
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
    print("MISSION 1 COMPLETE!")
    print("="*60)
    print(f"Final state: {state}")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print("="*60)
    print("[INFO] Cleanup complete")

    # 2 (PID 주행) 시작
    if state == "DONE":
        time.sleep(0.5)
        import subprocess
        subprocess.run(["python3", "../Ultrasonic_PID_driving/driving_edit.py"])
