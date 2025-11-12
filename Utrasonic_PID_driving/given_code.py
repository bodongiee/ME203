import RPi.GPIO as GPIO
import time
import sys
import tty
import termios
import select

Kp = 0.6
Ki = 0.0
Kd = 0.0

base_angle = 90
prev_error = 0
integral = 0

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
# Speed
SPEED_MIN = 30
SPEED_MAX = 40
MOTOR_SPEED = SPEED_MIN

GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)
motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

MIN_CM, MAX_CM = 3.0, 150.0
ALPHA = 0.35
SOUND_CM_S = 34300.0
TRIG_PULSE_S = 10e-6           # 10 µs
MARGIN_S = 0.002               # 2 ms 여유
ECHO_TIMEOUT_S = (2*MAX_CM / SOUND_CM_S) + MARGIN_S  # ≈ 0.0107 s

def sample_distance(trig, echo):
    #Valid -> CM, ELSE -> None
    # Trigger
    GPIO.output(trig, False)
    time.sleep(2e-6)
    GPIO.output(trig, True)
    time.sleep(TRIG_PULSE_S)
    GPIO.output(trig, False)

    t0 = time.perf_counter()

    # 상승 에지 대기
    while GPIO.input(echo) == 0:
        if time.perf_counter() - t0 > ECHO_TIMEOUT_S:
            return None  # 미검출(너무 멂/배선/자세 등)

    start = time.perf_counter()

    # 하강 에지 대기
    while GPIO.input(echo) == 1:
        if time.perf_counter() - start > ECHO_TIMEOUT_S:
            return None  # 반사 미약/너무 멂

    end = time.perf_counter()

    dist = (end - start) * SOUND_CM_S / 2.0  # cm

    # 유효 범위 체크 (유효만 클리핑)
    if dist < MIN_CM or dist > MAX_CM:
        return None
    return max(MIN_CM, min(dist, MAX_CM))

def smooth(prev_value, new_value, alpha=ALPHA):

    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha*new_value + (1-alpha)*prev_value

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

def set_servo_angle(degree):
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.1)

def move_forward(speed):
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def move_backward():
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(MOTOR_SPEED)

def stop_motor():
    motor_pwm.ChangeDutyCycle(0)

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def speed_from_angle(angle, amin=45, amid=90, amax=135, vmin=SPEED_MIN, vmax=SPEED_MAX):
# Dividing cases
    if angle <= amid:

        t = (angle- amin) / (amid - amin) # Smaller value calculated with bigger steering angle
        t = max(0.0, min(1.0, t)) # Normalization
        if t != 0:
            t = 1 / t * 3 # New value t which has bigger value with bigger steering angle
        t = min(15, t) # Clipped with 15 to limit speed increase
        
        return vmin + (vmax - vmin) * t * 0.25 # Speed increased with bigger steering angle but tuned to avoid too much increase in speed
    else: #Same as above except steering rotation
        t = (amax - angle) / (amax - amid)
        t = max(0.0, min(1.0, t))
        if t != 0:
            t = 1 / t * 3
        t = min(15, t)
        return vmin + (vmax - vmin) * t* 0.25




try:
    print("Press 'a' to enter PID autonomous mode, 'q' to quit.")
    while True:
        key = get_key()
        if key == 'a':
            print("PID Autonomous mode activated.")
            prev_error = 0
            integral = 0
            last_left = None
            last_right = None

            for _ in range (100000):
                raw_left = read_stable(TRIG_LEFT, ECHO_LEFT)
                raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)
                left = smooth(last_left, raw_left)
                right = smooth(last_right, raw_right)
                last_left, last_right = left, right

                if left is None or right is None:
                    continue
                error = left - right
                integral += error
                derivative = error - prev_error
                output = Kp*error + Ki*integral + Kd*derivative # Feedback control of angles with left and right ultrasonic sensor
                angle = max(45, min(135, base_angle - output))
                MOTOR_SPEED = speed_from_angle(angle) # New speed of motor considering steering
                
                print(f"L: {left:.1f} R: {right:.1f} Err: {error:.1f} "f"Angle: {angle:.1f} Speed: {MOTOR_SPEED:.0f}")
                
                angle1 = max(50, min(130, base_angle - output)) # Clipping angle value to avoid too much steering
                angle = round (angle1, 0)

                if left <= 10 :
                    set_servo_angle(120)

                elif right <= 10 :
                    set_servo_angle(6)
                
                else :
                    set_servo_angle(angle)
                
                move_forward(MOTOR_SPEED)
                time.sleep(0.0001)

                prev_error = error

finally:
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()