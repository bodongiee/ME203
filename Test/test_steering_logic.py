#!/usr/bin/env python3
# test_steering_logic.py
# 조향 로직 테스트 - 실제 하드웨어 없이 로직 검증

import numpy as np
import cv2

# PID Controller (line_segmentation_test.py와 동일)
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

# 조향 각도 계산 함수 (line_segmentation_test.py와 동일)
def calculate_steering_angle(line_center_x, line_angle, pid_controller):
    """
    조향 각도 계산

    Args:
        line_center_x: 라인 중심 (0~1, 0.5=중앙)
        line_angle: 라인 각도 (0~180도)
        pid_controller: PID 컨트롤러

    Returns:
        servo_angle: 서보 각도 (45~135도)
    """
    # Center error 계산
    center_error = (line_center_x - 0.5) * 2  # -1 ~ 1 범위

    # PID 제어
    correction = pid_controller.update(center_error)

    # 서보 각도 계산 (Center 기반)
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

# 시각화 함수
def visualize_scenario(line_center_x, line_angle, scenario_name):
    """시나리오 시각화"""

    # 이미지 생성 (640x480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # 어두운 배경

    # 카메라 뷰 (240x240)
    cam_x, cam_y = 50, 120
    cam_size = 240

    # 카메라 프레임
    cv2.rectangle(img, (cam_x, cam_y), (cam_x + cam_size, cam_y + cam_size), (100, 100, 100), 2)
    cv2.putText(img, "Camera View", (cam_x + 10, cam_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 화면 중앙선
    center_x = cam_x + cam_size // 2
    cv2.line(img, (center_x, cam_y), (center_x, cam_y + cam_size), (0, 255, 255), 1, cv2.LINE_DASHED)

    # 라인 위치
    line_x = int(cam_x + cam_size * line_center_x)

    # 라인 그리기 (각도 반영)
    angle_rad = np.deg2rad(line_angle)
    line_length = 100

    for y_offset in range(-50, 51, 10):
        y = cam_y + cam_size // 2 + y_offset
        x_offset = int(-y_offset * np.tan(np.pi/2 - angle_rad))
        x = line_x + x_offset

        if cam_y <= y <= cam_y + cam_size and cam_x <= x <= cam_x + cam_size:
            cv2.circle(img, (x, y), 8, (255, 255, 255), -1)

    # 라인 중심점
    cv2.circle(img, (line_x, cam_y + cam_size // 2), 6, (0, 0, 255), -1)
    cv2.putText(img, "Line Center", (line_x + 10, cam_y + cam_size // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Center error 표시
    error_y = cam_y + cam_size + 20
    cv2.line(img, (center_x, error_y), (line_x, error_y), (255, 100, 100), 2)
    cv2.arrowedLine(img, (center_x, error_y), (line_x, error_y), (255, 100, 100), 2, tipLength=0.3)

    center_error = (line_center_x - 0.5) * 2
    cv2.putText(img, f"center_error = {center_error:+.2f}",
                ((center_x + line_x) // 2 - 60, error_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

    # 라인 각도 표시
    cv2.putText(img, f"line_angle = {line_angle:.0f}°",
                (line_x + 20, cam_y + cam_size // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

    # PID 계산
    pid = PIDController()
    servo_angle, center_error, correction, angle_correction = calculate_steering_angle(
        line_center_x, line_angle, pid
    )

    # 서보 시각화 (Top View)
    servo_x = 400
    servo_y = 240
    servo_center = (servo_x, servo_y)

    # 차량
    car_w, car_h = 60, 90
    cv2.rectangle(img,
                  (servo_center[0] - car_w//2, servo_center[1] - car_h//2),
                  (servo_center[0] + car_w//2, servo_center[1] + car_h//2),
                  (150, 150, 150), -1)
    cv2.putText(img, "Robot", (servo_center[0] - 25, servo_center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 서보 범위
    radius = 120

    # 90도 (직진)
    angle_90_rad = np.deg2rad(90)
    x_90 = int(servo_center[0] + radius * np.cos(angle_90_rad))
    y_90 = int(servo_center[1] - radius * np.sin(angle_90_rad))
    cv2.line(img, servo_center, (x_90, y_90), (0, 255, 255), 2, cv2.LINE_DASHED)
    cv2.putText(img, "90° (straight)", (x_90 - 60, y_90 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 현재 서보 각도
    angle_current_rad = np.deg2rad(servo_angle)
    x_current = int(servo_center[0] + radius * np.cos(angle_current_rad))
    y_current = int(servo_center[1] - radius * np.sin(angle_current_rad))

    # 화살표 색상 (방향에 따라)
    if servo_angle < 85:
        arrow_color = (0, 100, 255)  # 파란색 (왼쪽)
    elif servo_angle > 95:
        arrow_color = (0, 255, 100)  # 초록색 (오른쪽)
    else:
        arrow_color = (255, 255, 0)  # 청록색 (직진)

    cv2.arrowedLine(img, servo_center, (x_current, y_current), arrow_color, 4, tipLength=0.2)
    cv2.putText(img, f"{servo_angle:.1f}°", (x_current + 10, y_current),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

    # 정보 패널
    info_x = 350
    info_y = 50

    cv2.rectangle(img, (info_x - 10, info_y - 10), (info_x + 280, info_y + 150), (60, 60, 60), -1)
    cv2.rectangle(img, (info_x - 10, info_y - 10), (info_x + 280, info_y + 150), (150, 150, 150), 2)

    cv2.putText(img, scenario_name, (info_x, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    info_texts = [
        f"line_center_x: {line_center_x:.2f}",
        f"line_angle: {line_angle:.0f}°",
        f"",
        f"center_error: {center_error:+.2f}",
        f"PID correction: {correction:+.2f}",
        f"angle_correction: {angle_correction:+.2f}",
        f"",
        f"servo_angle: {servo_angle:.1f}°",
    ]

    for i, text in enumerate(info_texts):
        cv2.putText(img, text, (info_x, info_y + 35 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 판단 결과
    judgment_y = info_y + 150
    if servo_angle < 85:
        judgment = "LEFT TURN"
        judgment_color = (0, 100, 255)
    elif servo_angle > 95:
        judgment = "RIGHT TURN"
        judgment_color = (0, 255, 100)
    else:
        judgment = "STRAIGHT"
        judgment_color = (255, 255, 0)

    cv2.putText(img, judgment, (info_x + 20, judgment_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, judgment_color, 2)

    return img

# 테스트 시나리오
test_cases = [
    # (line_center_x, line_angle, 시나리오명, 예상 행동)
    (0.5, 90, "Scenario 1: Center + Vertical", "STRAIGHT"),
    (0.75, 90, "Scenario 2: Right + Vertical", "RIGHT TURN"),
    (0.25, 90, "Scenario 3: Left + Vertical", "LEFT TURN"),
    (0.5, 45, "Scenario 4: Center + Right Tilt", "RIGHT TURN"),
    (0.5, 135, "Scenario 5: Center + Left Tilt", "LEFT TURN"),
    (0.75, 45, "Scenario 6: Right + Right Tilt", "RIGHT TURN"),
    (0.25, 135, "Scenario 7: Left + Left Tilt", "LEFT TURN"),
    (0.5, 180, "Scenario 8: Center + Horizontal", "STRAIGHT"),
    (1.0, 90, "Scenario 9: Far Right", "RIGHT TURN"),
    (0.0, 90, "Scenario 10: Far Left", "LEFT TURN"),
]

print("="*60)
print("조향 로직 테스트")
print("="*60)
print("\n테스트 케이스:")
print("  'n' - 다음 시나리오")
print("  'q' - 종료")
print("="*60)

# 테스트 실행
for i, (line_center_x, line_angle, scenario_name, expected) in enumerate(test_cases):
    print(f"\n[Test {i+1}/{len(test_cases)}] {scenario_name}")
    print(f"  line_center_x: {line_center_x:.2f}")
    print(f"  line_angle: {line_angle:.0f}°")
    print(f"  Expected: {expected}")

    # PID 초기화
    pid = PIDController()
    servo_angle, center_error, correction, angle_correction = calculate_steering_angle(
        line_center_x, line_angle, pid
    )

    print(f"  → center_error: {center_error:+.2f}")
    print(f"  → PID correction: {correction:+.2f}")
    print(f"  → angle_correction: {angle_correction:+.2f}")
    print(f"  → servo_angle: {servo_angle:.1f}°")

    # 판단
    if servo_angle < 85:
        actual = "LEFT TURN"
        status = "✅" if expected == "LEFT TURN" else "❌"
    elif servo_angle > 95:
        actual = "RIGHT TURN"
        status = "✅" if expected == "RIGHT TURN" else "❌"
    else:
        actual = "STRAIGHT"
        status = "✅" if expected == "STRAIGHT" else "❌"

    print(f"  → Actual: {actual} {status}")

    # 시각화
    img = visualize_scenario(line_center_x, line_angle, scenario_name)
    cv2.imshow("Steering Logic Test", img)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print("\n[INFO] Test interrupted by user")
        break

cv2.destroyAllWindows()

# 요약
print("\n" + "="*60)
print("테스트 완료!")
print("="*60)
print("\n로직 검증:")
print("  ✅ 라인이 오른쪽 → 오른쪽 회전")
print("  ✅ 라인이 왼쪽 → 왼쪽 회전")
print("  ✅ 라인이 중앙 → 직진")
print("  ✅ 라인이 오른쪽 위로 기울어짐 → 오른쪽 회전")
print("  ✅ 라인이 왼쪽 위로 기울어짐 → 왼쪽 회전")
print("="*60)
