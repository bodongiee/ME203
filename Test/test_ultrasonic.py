import RPi.GPIO as GPIO
import time
import sys
import tty
import termios


TRIG_LEFT = 17
ECHO_LEFT = 27
TRIG_RIGHT = 5
ECHO_RIGHT = 6





# ---------------- GPIO Init ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)



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

while(True):
    raw_left  = read_stable(TRIG_LEFT,  ECHO_LEFT)
    raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)
    print("[LEFT] : " + str(raw_left) + "| [RIGHT] : " + str(raw_right))
