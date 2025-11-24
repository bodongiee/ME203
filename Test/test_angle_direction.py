#!/usr/bin/env python3
# test_angle_direction.py
# OpenCV fitEllipse 각도 방향 테스트

import cv2
import numpy as np

# 테스트 이미지 생성 (240x240)
img_size = 240

def draw_line_and_test(angle_deg, description):
    """특정 각도로 라인을 그리고 fitEllipse로 각도 확인"""

    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # 중심점
    cx, cy = img_size // 2, img_size // 2

    # 각도를 라디안으로
    angle_rad = np.deg2rad(angle_deg)

    # 라인 길이
    length = 100

    # 시작점과 끝점 계산
    x1 = int(cx - length * np.cos(angle_rad))
    y1 = int(cy - length * np.sin(angle_rad))
    x2 = int(cx + length * np.cos(angle_rad))
    y2 = int(cy + length * np.sin(angle_rad))

    # 라인 그리기
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 10)

    # Contour 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours[0]) >= 5:
        ellipse = cv2.fitEllipse(contours[0])
        detected_angle = ellipse[2]

        # 음수 각도 보정
        if detected_angle < 0:
            detected_angle += 180

        # 타원 그리기 (검증용)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

        # 정보 표시
        cv2.putText(img, f"Input: {angle_deg}deg", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"Detected: {detected_angle:.1f}deg", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, description, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 좌표계 표시
        cv2.arrowedLine(img, (20, 220), (60, 220), (0, 0, 255), 2)  # X축
        cv2.arrowedLine(img, (20, 220), (20, 180), (0, 255, 0), 2)  # Y축
        cv2.putText(img, "X", (65, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, "Y", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return img, detected_angle

    return img, None

# 테스트 케이스
test_cases = [
    (90, "Vertical (Up) - Should go straight"),
    (0, "Horizontal (Right) - Almost horizontal"),
    (180, "Horizontal (Left) - Almost horizontal"),
    (45, "Diagonal (Down-Right)"),
    (135, "Diagonal (Down-Left)"),
    (60, "Tilted Right-Up"),
    (120, "Tilted Left-Up"),
]

print("OpenCV fitEllipse Angle Test")
print("="*60)
print("OpenCV Coordinate System: X→right, Y→down")
print("Angle: 0° = horizontal right, 90° = vertical up")
print("="*60)

results = []

for angle, desc in test_cases:
    img, detected = draw_line_and_test(angle, desc)

    if detected is not None:
        results.append((angle, detected, desc, img))
        print(f"{angle:3d}° (Input) → {detected:6.1f}° (Detected) | {desc}")
    else:
        print(f"{angle:3d}° (Input) → Failed to detect")

# 이미지 표시
print("\n" + "="*60)
print("Press any key to see each test case...")
print("="*60)

for angle, detected, desc, img in results:
    cv2.imshow(f"Angle Test: {desc}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# 요약
print("\n" + "="*60)
print("SUMMARY - OpenCV Angle Convention")
print("="*60)
print("90° = Vertical line (↑) - Robot should go straight")
print("0° = Horizontal line (→) - Almost flat")
print("180° = Horizontal line (←) - Almost flat")
print("45° = Diagonal (↗) - Line tilts to upper-right")
print("135° = Diagonal (↖) - Line tilts to upper-left")
print("="*60)
