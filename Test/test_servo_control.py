#!/usr/bin/env python3
# test_servo_control.py
# 라즈베리파이에서 실제 서보 모터를 움직이며 조향 테스트

import numpy as np
import cv2
import time
import RPi.GPIO as GPIO

# ---------------- GPIO 설정 ----------------
SERVO_PIN = 13
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
servo_pwm.start(0)

# ---------------- 서보 제어 함수 ----------------
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

# ---------------- 조향 각도 계산 ----------------
def calculate_steering_angle(line_center_x, line_angle, pid_controller):
    """조향 각도 계산"""
    # Center error
    center_error = (line_center_x - 0.5) * 2

    # PID 제어
    correction = pid_controller.update(center_error)

    # 서보 각도 (Center 기반)
    servo_angle = 90 + (correction * 30)

    # 각도 보정 (Angle 기반)
    angle_deviation = line_angle - 90

    # 수평선 근처는 무시
    if abs(angle_deviation) > 60:
        angle_correction = 0
    else:
        angle_correction = angle_deviation * 0.15
        servo_angle -= angle_correction

    # 범위 제한
    servo_angle = max(45, min(135, servo_angle))

    return servo_angle, center_error, correction, angle_correction

# ---------------- 시각화 ----------------
def create_visualization(line_center_x, line_angle, servo_angle, center_error, correction, angle_correction, scenario_name):
    """시각화 이미지 생성"""

    # 이미지 생성 (480x640)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)

    # 카메라 뷰
    cam_x, cam_y = 30, 80
    cam_size = 200

    cv2.rectangle(img, (cam_x, cam_y), (cam_x + cam_size, cam_y + cam_size), (100, 100, 100), 2)
    cv2.putText(img, "Camera View", (cam_x + 10, cam_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 중앙선
    center_x = cam_x + cam_size // 2
    cv2.line(img, (center_x, cam_y), (center_x, cam_y + cam_size), (0, 255, 255), 1, cv2.LINE_DASHED)

    # 라인 그리기
    line_x = int(cam_x + cam_size * line_center_x)
    angle_rad = np.deg2rad(line_angle)

    for y_offset in range(-50, 51, 8):
        y = cam_y + cam_size // 2 + y_offset
        x_offset = int(-y_offset * np.tan(np.pi/2 - angle_rad))
        x = line_x + x_offset

        if cam_y <= y <= cam_y + cam_size and cam_x <= x <= cam_x + cam_size:
            cv2.circle(img, (x, y), 6, (255, 255, 255), -1)

    # 라인 중심
    cv2.circle(img, (line_x, cam_y + cam_size // 2), 5, (0, 0, 255), -1)

    # Center error 표시
    error_y = cam_y + cam_size + 15
    if abs(line_x - center_x) > 5:
        cv2.arrowedLine(img, (center_x, error_y), (line_x, error_y), (255, 100, 100), 2, tipLength=0.3)

    cv2.putText(img, f"error={center_error:+.2f}",
                ((center_x + line_x) // 2 - 40, error_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

    # 각도 표시
    cv2.putText(img, f"angle={line_angle:.0f}°",
                (line_x + 15, cam_y + cam_size // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)

    # 서보 시각화 (Top View)
    servo_x = 350
    servo_y = 200

    # 차량
    car_w, car_h = 50, 75
    cv2.rectangle(img, (servo_x - car_w//2, servo_y - car_h//2),
                  (servo_x + car_w//2, servo_y + car_h//2), (150, 150, 150), -1)
    cv2.putText(img, "Robot", (servo_x - 20, servo_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 서보 범위
    radius = 100

    # 90도 (직진)
    angle_90_rad = np.deg2rad(90)
    x_90 = int(servo_x + radius * np.cos(angle_90_rad))
    y_90 = int(servo_y - radius * np.sin(angle_90_rad))
    cv2.line(img, (servo_x, servo_y), (x_90, y_90), (0, 255, 255), 1, cv2.LINE_DASHED)
    cv2.putText(img, "90°", (x_90 - 15, y_90 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 현재 서보 각도
    angle_rad = np.deg2rad(servo_angle)
    x_servo = int(servo_x + radius * np.cos(angle_rad))
    y_servo = int(servo_y - radius * np.sin(angle_rad))

    if servo_angle < 85:
        color = (255, 100, 0)  # 파란색 (왼쪽)
    elif servo_angle > 95:
        color = (0, 255, 100)  # 초록색 (오른쪽)
    else:
        color = (100, 255, 255)  # 노란색 (직진)

    cv2.arrowedLine(img, (servo_x, servo_y), (x_servo, y_servo), color, 3, tipLength=0.2)
    cv2.putText(img, f"{servo_angle:.1f}°", (x_servo + 10, y_servo),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 정보 패널
    info_x = 30
    info_y = 320

    cv2.rectangle(img, (info_x - 5, info_y - 5), (info_x + 230, info_y + 145), (60, 60, 60), -1)
    cv2.rectangle(img, (info_x - 5, info_y - 5), (info_x + 230, info_y + 145), (150, 150, 150), 2)

    cv2.putText(img, scenario_name, (info_x, info_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    info_texts = [
        f"line_center: {line_center_x:.2f}",
        f"line_angle: {line_angle:.0f} deg",
        f"",
        f"center_error: {center_error:+.2f}",
        f"PID corr: {correction:+.2f}",
        f"angle_corr: {angle_correction:+.2f}",
        f"",
        f"SERVO: {servo_angle:.1f} deg",
    ]

    for i, text in enumerate(info_texts):
        cv2.putText(img, text, (info_x, info_y + 40 + i * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 판단
    if servo_angle < 85:
        judgment = "LEFT"
        j_color = (255, 100, 0)
    elif servo_angle > 95:
        judgment = "RIGHT"
        j_color = (0, 255, 100)
    else:
        judgment = "STRAIGHT"
        j_color = (100, 255, 255)

    cv2.putText(img, judgment, (info_x + 50, info_y + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, j_color, 2)

    # 컨트롤 안내
    control_y = 450
    cv2.putText(img, "Controls: Arrow Keys | Space=Center | Q=Quit", (30, control_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img

# ---------------- 메인 루프 ----------------
def main():
    print("="*60)
    print("서보 제어 테스트")
    print("="*60)
    print("\n키 조작:")
    print("  ← → : 라인 좌우 이동")
    print("  ↑ ↓ : 라인 각도 조정")
    print("  Space : 중앙 정렬")
    print("  Q : 종료")
    print("="*60)

    # 초기 서보 중앙
    set_servo_angle(90)
    time.sleep(0.5)

    # 초기값
    line_center_x = 0.5
    line_angle = 90.0

    pid = PIDController()

    try:
        while True:
            # 조향 각도 계산
            servo_angle, center_error, correction, angle_correction = calculate_steering_angle(
                line_center_x, line_angle, pid
            )

            # 실제 서보 제어
            actual_servo = set_servo_angle(servo_angle)

            # 시각화
            scenario_name = f"Manual Control"
            img = create_visualization(
                line_center_x, line_angle, actual_servo,
                center_error, correction, angle_correction, scenario_name
            )

            cv2.imshow("Servo Control Test", img)

            # 키 입력
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                print("\n[INFO] Quit")
                break

            elif key == 81 or key == 2:  # 왼쪽 화살표
                line_center_x = max(0.0, line_center_x - 0.05)
                pid.reset()
                print(f"Line center: {line_center_x:.2f}")

            elif key == 83 or key == 3:  # 오른쪽 화살표
                line_center_x = min(1.0, line_center_x + 0.05)
                pid.reset()
                print(f"Line center: {line_center_x:.2f}")

            elif key == 82 or key == 0:  # 위쪽 화살표
                line_angle = (line_angle + 5) % 180
                print(f"Line angle: {line_angle:.0f}°")

            elif key == 84 or key == 1:  # 아래쪽 화살표
                line_angle = (line_angle - 5) % 180
                print(f"Line angle: {line_angle:.0f}°")

            elif key == ord(' '):  # 스페이스
                line_center_x = 0.5
                line_angle = 90.0
                pid.reset()
                print("Reset to center")

            elif key == ord('1'):  # 시나리오 1
                line_center_x = 0.75
                line_angle = 90
                pid.reset()
                print("Scenario: Right + Vertical")

            elif key == ord('2'):  # 시나리오 2
                line_center_x = 0.25
                line_angle = 90
                pid.reset()
                print("Scenario: Left + Vertical")

            elif key == ord('3'):  # 시나리오 3
                line_center_x = 0.5
                line_angle = 45
                pid.reset()
                print("Scenario: Center + Right Tilt")

            elif key == ord('4'):  # 시나리오 4
                line_center_x = 0.5
                line_angle = 135
                pid.reset()
                print("Scenario: Center + Left Tilt")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # 서보 중앙으로
        set_servo_angle(90)
        time.sleep(0.3)
        servo_pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == '__main__':
    main()
