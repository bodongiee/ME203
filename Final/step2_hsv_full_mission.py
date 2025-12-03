#!/home/mecha/venvs/tflite/bin/python
# step2_hsv_full_mission.py
# Step 1: 라인 추적 → 빨간불 후진 → 초록불 1.5초 직진
# Step 2: PID 벽추종 주행

import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ---------------- 설정 ----------------
IMG_SIZE = 160
COLOR_SIZE = 160
MODEL_PATH = "./line_segmentation_light.tflite"

# GPIO 핀 설정
SERVO_PIN = 13
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

DIR_PIN = 16
PWM_PIN = 12
MOTOR_FREQ = 1000

# 초음파 센서
TRIG_LEFT = 17
ECHO_LEFT = 27
TRIG_RIGHT = 5
ECHO_RIGHT = 6

# 속도 설정
SPEED_LINE_TRACKING = 25  # 라인 추적 속도
SPEED_MIN = 50  # PID 주행 최소 속도
SPEED_MAX = 80  # PID 주행 최대 속도

# HSV 색상 범위
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([80, 255, 255])
RED_LOWER1 = np.array([0, 50, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 50, 50])
RED_UPPER2 = np.array([180, 255, 255])

# PID 파라미터
Kp = 0.46
Ki = 0.0
Kd = 0.2
base_angle = 90

# 초음파 파라미터
MIN_CM, MAX_CM = 3.0, 150.0
ALPHA = 0.35
SOUND_CM_S = 34300.0
TRIG_PULSE_S = 10e-6
MARGIN_S = 0.002
ECHO_TIMEOUT_S = (2 * MAX_CM / SOUND_CM_S) + MARGIN_S
EMERGENCY_CM = 6.0
LOST_TIMEOUT = 0.5
V_SAFE = 0.5
SPEED_SLEW = 15.0

print("="*60)
print("Full Mission: 신호등 대응 → PID 벽추종")
print("="*60)

# ---------------- GPIO 초기화 ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# ---------------- 서보/모터 제어 함수 ----------------
def set_servo_angle(angle, hold=True):
    """서보 각도 설정 (45~135도)
    hold=True: duty 유지 (라인 추적용)
    hold=False: duty 0으로 (PID 주행용, 떨림 방지)
    """
    angle = max(45, min(135, angle))
    duty = SERVO_MIN_DUTY + (angle - 45) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 90.0
    servo_pwm.ChangeDutyCycle(duty)
    if not hold:
        time.sleep(0.05)
        servo_pwm.ChangeDutyCycle(0)
    return angle

def set_motor(speed, direction="forward"):
    """모터 속도 및 방향 설정"""
    if direction == "forward":
        GPIO.output(DIR_PIN, GPIO.HIGH)
    elif direction == "backward":
        GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)

def stop_motor():
    """모터 정지"""
    motor_pwm.ChangeDutyCycle(0)

# ---------------- HSV 색상 감지 ----------------
def detect_color_hsv(rgb_image):
    """HSV 색상 감지"""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_pixels = np.sum(green_mask > 0)

    red_mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_pixels = np.sum(red_mask > 0)

    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    green_ratio = green_pixels / total_pixels
    red_ratio = red_pixels / total_pixels

    green_conf = min(green_ratio / 0.1, 1.0)
    red_conf = min(red_ratio / 0.1, 1.0)

    if green_conf > red_conf and green_conf > 0.3:
        return "green", green_conf
    elif red_conf > green_conf and red_conf > 0.3:
        return "red", red_conf
    else:
        return "none", 0.0

# ---------------- 초음파 센서 함수 ----------------
def sample_distance(trig, echo):
    """초음파 거리 측정"""
    GPIO.output(trig, False)
    time.sleep(2e-6)
    GPIO.output(trig, True)
    time.sleep(TRIG_PULSE_S)
    GPIO.output(trig, False)

    t0 = time.perf_counter()
    while GPIO.input(echo) == 0:
        if time.perf_counter() - t0 > ECHO_TIMEOUT_S:
            return None

    start = time.perf_counter()
    while GPIO.input(echo) == 1:
        if time.perf_counter() - start > ECHO_TIMEOUT_S:
            return None

    end = time.perf_counter()
    dist = (end - start) * SOUND_CM_S / 2.0
    if dist < MIN_CM or dist > MAX_CM:
        return None
    return dist

def read_stable(trig, echo, k=5):
    """안정적인 거리 측정 (중간값 사용)"""
    vals = []
    for _ in range(k):
        v = sample_distance(trig, echo)
        if v is not None:
            vals.append(v)
        time.sleep(0.001)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals)//2]

def smooth(prev_value, new_value, alpha=ALPHA):
    """EMA 필터링"""
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value

def slew(prev_val, target_val, max_delta):
    """속도 변화 제한"""
    if target_val > prev_val:
        return min(target_val, prev_val + max_delta)
    else:
        return max(target_val, prev_val - max_delta)

def speed_from_angle_safe(angle, amin=45, amid=90, amax=135,
                          vmin=SPEED_MIN, vmax=SPEED_MAX, gamma=1.3):
    """조향 각도에 따른 안전 속도 계산"""
    if angle <= amid:
        r = (amid - angle) / (amid - amin)
    else:
        r = (angle - amid) / (amax - amid)
    r = max(0.0, min(1.0, r))
    k = (1.0 - r) ** gamma
    v = vmin + (vmax - vmin) * k
    return max(vmin, min(vmax, v))

# ---------------- TFLite 모델 로드 ----------------
print("\n[INFO] Loading segmentation model...")
seg_interpreter = tflite.Interpreter(model_path=MODEL_PATH)
seg_interpreter.allocate_tensors()

seg_input_details = seg_interpreter.get_input_details()[0]
seg_output_details = seg_interpreter.get_output_details()[0]
seg_is_int8 = (seg_input_details['dtype'] == np.uint8)
print(f"✓ Segmentation model loaded (INT8: {seg_is_int8})")

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

print("\n[START] Mission Start!")
print("-"*60)

# ---------------- 메인 루프 ----------------
state = "TRACKING"  # TRACKING, REVERSING, WAITING_GREEN, MOVING_FORWARD, PID_DRIVING
frame_count = 0
start_time = time.time()
green_light_time = None
reverse_start_time = None

# PID 변수
prev_error = 0.0
integral = 0.0
last_left = None
last_right = None
last_valid_ts = time.time()
MOTOR_SPEED = SPEED_MIN
pid_state = 'TRACKING'

try:
    while True:
        frame_count += 1

        # ============ STEP 1: 라인 추적 및 신호등 대응 ============
        if state in ["TRACKING", "REVERSING", "WAITING_GREEN", "MOVING_FORWARD"]:
            # 프레임 캡처
            frame = picam2.capture_array()

            # HSV 색상 감지용 전처리
            h, w = frame.shape[:2]
            y_color = (h - COLOR_SIZE) // 2
            x_color = (w - COLOR_SIZE) // 2
            color_cropped = frame[y_color:y_color+COLOR_SIZE, x_color:x_color+COLOR_SIZE]
            color_name, color_conf = detect_color_hsv(color_cropped)

            # 라인 감지용 전처리
            y_start = (h - IMG_SIZE) // 2
            x_start = (w - IMG_SIZE) // 2
            cropped = frame[y_start:y_start+IMG_SIZE, x_start:x_start+IMG_SIZE]

            # 세그멘테이션 추론
            if seg_is_int8:
                input_data = cropped.astype(np.uint8)
            else:
                input_data = cropped.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            seg_interpreter.set_tensor(seg_input_details['index'], input_data)
            seg_interpreter.invoke()
            output_data = seg_interpreter.get_tensor(seg_output_details['index'])
            mask = output_data[0].squeeze()

            # INT8 역양자화
            if seg_output_details['dtype'] == np.uint8 or seg_output_details['dtype'] == np.int8:
                quant = seg_output_details.get('quantization_parameters', {})
                scale = quant.get('scales', [1.0])[0]
                zero_point = quant.get('zero_points', [0])[0]
                mask = (mask.astype(np.float32) - zero_point) * scale
                mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -10, 10)))

            mask_binary = (mask > 0.5).astype(np.uint8) * 255

            # 테두리 제거
            border = 10
            mask_binary[:border, :] = 0
            mask_binary[-border:, :] = 0
            mask_binary[:, :border] = 0
            mask_binary[:, -border:] = 0

            # 상태별 처리
            if state == "TRACKING":
                # 빨간불 감지
                if color_name == "red" and color_conf > 0.5:
                    print(f"\n🔴 RED LIGHT DETECTED! (conf={color_conf:.3f})")
                    print(f"Reversing...")
                    set_servo_angle(90)
                    set_motor(SPEED_LINE_TRACKING, direction="backward")
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
                        set_motor(SPEED_LINE_TRACKING)
                        if frame_count % 10 == 0:
                            print(f"TRACKING | Pos={line_center_x:.2f} | Servo={servo_angle:.1f}°")
                    else:
                        set_servo_angle(90)
                        stop_motor()
                else:
                    set_servo_angle(90)
                    stop_motor()

            elif state == "REVERSING":
                elapsed = time.time() - reverse_start_time
                if elapsed >= 0.5:
                    print(f"\n✓ Reverse complete. Stopping...")
                    set_servo_angle(90)
                    stop_motor()
                    state = "WAITING_GREEN"
                else:
                    set_servo_angle(90)
                    set_motor(SPEED_LINE_TRACKING, direction="backward")

            elif state == "WAITING_GREEN":
                if color_name == "green" and color_conf > 0.5:
                    print(f"\nGREEN LIGHT DETECTED! Moving forward...")
                    set_servo_angle(80)
                    set_motor(SPEED_LINE_TRACKING)
                    green_light_time = time.time()
                    state = "MOVING_FORWARD"
                else:
                    if frame_count % 10 == 0:
                        print(f"WAITING | Color={color_name}({color_conf:.2f})")

            elif state == "MOVING_FORWARD":
                elapsed = time.time() - green_light_time
                if elapsed >= 1.5:
                    print(f"\n✓ Forward complete. Switching to PID mode...")
                    set_servo_angle(90, hold=True)
                    stop_motor()
                    time.sleep(0.5)

                    # 카메라 정리
                    print("[INFO] Stopping camera for PID mode...")
                    picam2.stop()
                    time.sleep(0.3)

                    # PID 모드로 전환 준비
                    prev_error = 0.0
                    integral = 0.0
                    last_left = None
                    last_right = None
                    last_valid_ts = time.time()
                    pid_state = 'TRACKING'
                    MOTOR_SPEED = SPEED_MIN
                    last_time = time.time()

                    state = "PID_DRIVING"
                    print("\n" + "="*60)
                    print("Entering PID Wall-Following Mode")
                    print("="*60)
                else:
                    set_servo_angle(80)
                    set_motor(SPEED_LINE_TRACKING)
                    if frame_count % 5 == 0:
                        print(f"FORWARD | Time={elapsed:.1f}s")

        # ============ STEP 2: PID 벽추종 주행 ============
        elif state == "PID_DRIVING":
            # 초음파 센서 읽기
            raw_left = read_stable(TRIG_LEFT, ECHO_LEFT)
            raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)
            left = smooth(last_left, raw_left)
            right = smooth(last_right, raw_right)
            last_left, last_right = left, right

            now = time.time()
            dt = max(1e-3, now - last_time)
            last_time = now

            # 상태 전이
            if raw_left is not None and raw_right is not None:
                last_valid_ts = now
                valid = True
            else:
                valid = False

            if not valid and (now - last_valid_ts) > LOST_TIMEOUT:
                pid_state = 'LOST'
            elif valid:
                pid_state = 'TRACKING'

            # 응급 회피
            if left is not None and left <= EMERGENCY_CM:
                set_servo_angle(120, hold=False)
                target_speed = SPEED_MIN
                MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
                set_motor(MOTOR_SPEED)
                print(f"[EMERGENCY-L] L:{left:.1f} R:{(right or -1):.1f} spd:{MOTOR_SPEED:.0f}")
                continue

            if right is not None and right <= EMERGENCY_CM:
                set_servo_angle(60, hold=False)
                target_speed = SPEED_MIN
                MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
                set_motor(MOTOR_SPEED)
                print(f"[EMERGENCY-R] L:{(left or -1):.1f} R:{right:.1f} spd:{MOTOR_SPEED:.0f}")
                continue

            # LOST 상태 처리
            if pid_state == 'LOST':
                integral = 0.0
                angle_cmd = base_angle
                target_speed = speed_from_angle_safe(angle_cmd) * V_SAFE
                MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
                set_servo_angle(angle_cmd, hold=False)
                set_motor(MOTOR_SPEED)
                print(f"[LOST] L:{left} R:{right} angle:{angle_cmd:.1f} spd:{MOTOR_SPEED:.0f}")
                continue

            # 정상 PID 제어
            if left is None or right is None:
                angle_cmd = base_angle
                target_speed = speed_from_angle_safe(angle_cmd) * 0.8
                MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
                set_servo_angle(angle_cmd, hold=False)
                set_motor(MOTOR_SPEED)
                print(f"[HOLD] L:{left} R:{right} angle:{angle_cmd:.1f} spd:{MOTOR_SPEED:.0f}")
                continue

            # PID 계산
            error = left - right
            integral += error * dt
            integral = max(-200.0, min(200.0, integral))
            derivative = (error - prev_error) / dt
            prev_error = error

            output = Kp * error + Ki * integral + Kd * derivative
            angle_cmd = max(45.0, min(135.0, base_angle - output))
            target_speed = speed_from_angle_safe(angle_cmd)
            MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)

            set_servo_angle(round(angle_cmd, 0), hold=False)
            set_motor(MOTOR_SPEED)

            if frame_count % 10 == 0:
                print(f"L:{left:.1f} R:{right:.1f} err:{error:.2f} ang:{angle_cmd:.1f} v:{MOTOR_SPEED:.0f}")

            time.sleep(0.0001)

except KeyboardInterrupt:
    print("\n\n[INFO] Stopping...")

finally:
    # 정지
    set_servo_angle(90, hold=True)
    stop_motor()
    time.sleep(0.3)

    # 정리
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()

    # 카메라 정리 (PID 모드에서는 이미 종료됨)
    try:
        if state != "PID_DRIVING":
            picam2.stop()
    except:
        pass

    # 통계
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print("MISSION COMPLETE!")
    print("="*60)
    print(f"Final state: {state}")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    if state != "PID_DRIVING":
        print(f"Average FPS: {avg_fps:.1f}")
    print("="*60)
    print("[INFO] Cleanup complete")
