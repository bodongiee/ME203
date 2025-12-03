import RPi.GPIO as GPIO
import time
import sys
import tty
import termios

# ---------------- PID & System Params ----------------
Kp = 0.46
Ki = 0.0
Kd = 0.2

base_angle = 90
prev_error = 0.0
integral = 0.0

DIR_PIN = 16
PWM_PIN = 12
SERVO_PIN = 13
TRIG_LEFT = 17
ECHO_LEFT = 27
TRIG_RIGHT = 5
ECHO_RIGHT = 6

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 속도(듀티, 0~100)
SPEED_MIN = 50
SPEED_MAX = 80
MOTOR_SPEED = SPEED_MIN

# 응급/안전 파라미터
EMERGENCY_CM = 6.0            # 응급 회피 임계
LOST_TIMEOUT = 0.5             # 연속 미검출 시
V_SAFE = 0.5                   # LOST에서 순항 대비 배율(0~1)
SPEED_SLEW = 15.0               # 한 루프당 최대 듀티 변화(가감속 제한)

# ---------------- GPIO Init ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)
motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# ---------------- Ultrasonic Params ----------------
MIN_CM, MAX_CM = 3.0, 150.0
ALPHA = 0.35
SOUND_CM_S = 34300.0
TRIG_PULSE_S = 10e-6
MARGIN_S = 0.002
ECHO_TIMEOUT_S = (2 * MAX_CM / SOUND_CM_S) + MARGIN_S  # ≈ 0.0107 s


def sample_distance(trig, echo):
    # Valid -> float(cm), else -> None (미검출)
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
    # 유효값만 EMA 업데이트, None이면 홀드
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value


# ---------------- Actuators ----------------
def set_servo_angle(degree):
    # 하우징 물리범위 보호
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.05)
    # 버징/발열 줄이기
    servo_pwm.ChangeDutyCycle(0)

def move_forward(speed):
    # 듀티 클램프
    duty = max(0.0, min(100.0, speed))
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(duty)

def stop_motor():
    motor_pwm.ChangeDutyCycle(0)


# ---------------- Utility ----------------
def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def slew(prev_val, target_val, max_delta):
    if target_val > prev_val:
        return min(target_val, prev_val + max_delta)
    else:
        return max(target_val, prev_val - max_delta)

# 안전하고 단조로운 속도 매핑(직진 빠름, 큰 조향 느림)
def speed_from_angle_safe(angle, amin=45, amid=90, amax=135,
                          vmin=SPEED_MIN, vmax=SPEED_MAX, gamma=1.3):
    if angle <= amid:
        r = (amid - angle) / (amid - amin)
    else:
        r = (angle - amid) / (amax - amid)
    r = max(0.0, min(1.0, r))
    k = (1.0 - r) ** gamma                     # 직진 1.0 → 조향 클수록 감소
    v = vmin + (vmax - vmin) * k
    return max(vmin, min(vmax, v))

# ---------------- Main Control ----------------
try:
    print("="*60)
    print("PID Autonomous Mode Starting...")
    print("Press 'q' at any time to quit")
    print("="*60)
    time.sleep(0.5)

    # 자동 시작 (키 입력 불필요)
    print("PID Autonomous mode activated.")
    prev_error = 0.0
    integral = 0.0
    last_left = None
    last_right = None
    last_valid_ts = time.time()
    state = 'TRACKING'
    MOTOR_SPEED = SPEED_MIN
    last_time = time.time()

    for _ in range(100000):
        # 1) 센서 읽기 + 필터
        raw_left  = read_stable(TRIG_LEFT,  ECHO_LEFT)
        raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)
        left  = smooth(last_left,  raw_left)
        right = smooth(last_right, raw_right)
        last_left, last_right = left, right

        now = time.time()
        dt = max(1e-3, now - last_time)    # 제어 dt
        last_time = now

        # 상태 전이
        if raw_left is not None and raw_right is not None:
            last_valid_ts = now
            valid = True
        else:
            valid = False

        if not valid and (now - last_valid_ts) > LOST_TIMEOUT:
            state = 'LOST'
        elif valid:
            state = 'TRACKING'

        # 2) 응급 회피
        if left is not None and left <= EMERGENCY_CM:
            set_servo_angle(120)
            target_speed = SPEED_MIN       # 감속
            MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
            move_forward(MOTOR_SPEED)
            print(f"[EMERGENCY-L] L:{left:.1f} R:{(right or -1):.1f} spd:{MOTOR_SPEED:.0f}")
            continue

        if right is not None and right <= EMERGENCY_CM:
            set_servo_angle(60)            # 왼쪽으로 회피
            target_speed = SPEED_MIN
            MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
            move_forward(MOTOR_SPEED)
            print(f"[EMERGENCY-R] L:{(left or -1):.1f} R:{right:.1f} spd:{MOTOR_SPEED:.0f}")
            continue

        # 3) 상태별 제어
        if state == 'LOST':
            integral = 0.0
            angle_cmd = base_angle
            target_speed = speed_from_angle_safe(angle_cmd) * V_SAFE
            MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
            set_servo_angle(angle_cmd)
            move_forward(MOTOR_SPEED)
            print(f"[LOST] L:{left} R:{right} angle:{angle_cmd:.1f} spd:{MOTOR_SPEED:.0f}")
            continue

        # 4) 정상 벽추종(PID)
        if left is None or right is None:
            angle_cmd = base_angle
            target_speed = speed_from_angle_safe(angle_cmd) * 0.8
            MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)
            set_servo_angle(angle_cmd)
            move_forward(MOTOR_SPEED)
            print(f"[HOLD] L:{left} R:{right} angle:{angle_cmd:.1f} spd:{MOTOR_SPEED:.0f}")
            continue

        # PID 오차(좌-우)
        error = left - right
        integral += error * dt
        integral = max(-200.0, min(200.0, integral))
        derivative = (error - prev_error) / dt
        prev_error = error

        output = Kp * error + Ki * integral + Kd * derivative
        # 조향 계산 및 클램프
        angle_cmd = max(45.0, min(135.0, base_angle - output))
        # 속도는 조향 의존(안전한 단조형)
        target_speed = speed_from_angle_safe(angle_cmd)
        MOTOR_SPEED = slew(MOTOR_SPEED, target_speed, SPEED_SLEW)

        # 출력 적용
        set_servo_angle(round(angle_cmd, 0))
        move_forward(MOTOR_SPEED)

        # 모니터링
        print(f"L:{left:.1f} R:{right:.1f} err:{error:.2f} ang:{angle_cmd:.1f} v:{MOTOR_SPEED:.0f}")

        # 제어주기
        time.sleep(0.0001)

finally:
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
