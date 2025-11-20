#!/usr/bin/env python3
# test_mouse.py - 마우스 드래그 테스트

import cv2
import numpy as np

print("=" * 60)
print("마우스 드래그 테스트")
print("=" * 60)
print("Mask 창에서 마우스를 드래그해보세요.")
print("초록색으로 그려지면 성공입니다.")
print("'q' 키를 누르면 종료합니다.")
print("=" * 60)

# 전역 변수
drawing = False
prev_x, prev_y = -1, -1
brush_size = 20

# 빈 캔버스
img = np.zeros((480, 640, 3), np.uint8)
mask = np.zeros((480, 640), np.uint8)

def mouse_callback(event, x, y, flags, param):
    global drawing, mask, prev_x, prev_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), brush_size, 255, -1)
        prev_x, prev_y = x, y
        print(f"마우스 다운: ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if prev_x >= 0 and prev_y >= 0:
                # 두꺼운 선으로 연결
                cv2.line(mask, (prev_x, prev_y), (x, y), 255, brush_size * 2)
                # 원도 추가
                cv2.circle(mask, (x, y), brush_size, 255, -1)
            prev_x, prev_y = x, y
            # 출력 줄이기 (너무 많으면 느려짐)
            if prev_x % 5 == 0:
                print(f"드래그 중: ({x}, {y})")

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_x, prev_y = -1, -1
        print("마우스 업")

# 윈도우 생성
cv2.namedWindow('Mask')
cv2.setMouseCallback('Mask', mouse_callback)

print("\n대기 중... Mask 창에서 마우스를 드래그하세요.\n")

while True:
    # 디스플레이용 이미지
    display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 오버레이
    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]  # 초록색
    display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

    # 상태 표시
    cv2.putText(display, f"Brush: {brush_size}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if drawing:
        cv2.putText(display, "DRAWING!", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "Ready", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Mask', display)

    # 화면 업데이트를 더 빠르게
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n종료합니다.")
        break
    elif key == ord('c'):
        mask = np.zeros((480, 640), np.uint8)
        print("초기화")

cv2.destroyAllWindows()
print("\n테스트 완료!")
print("만약 드래그가 작동했다면, label_segmentation.py도 작동해야 합니다.")
print("만약 드래그가 안 되었다면, OpenCV 설치에 문제가 있을 수 있습니다.")
