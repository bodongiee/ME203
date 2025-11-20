#!/usr/bin/env python3
# label_segmentation.py
# Segmentation을 위한 마스크 라벨링 도구

import cv2
import numpy as np
import os
import glob

"""
사용 방법:
1. 원본 이미지를 ./data/images/ 폴더에 저장
2. 이 스크립트 실행
3. 마우스로 라인 위에 드래그하여 마스킹
4. 키 조작:
   - 's': 저장하고 다음 이미지로
   - 'c': 현재 마스크 초기화
   - 'u': 이전 작업 취소
   - 'q': 종료
   - '+/-': 브러시 크기 조절
"""

# ---------------- 설정 ----------------
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"
BRUSH_SIZE = 15
BRUSH_COLOR = 255  # 흰색 (라인 영역)

# 마스크 디렉토리 생성
os.makedirs(MASKS_DIR, exist_ok=True)

# ---------------- 전역 변수 ----------------
drawing = False
brush_size = BRUSH_SIZE
mask = None
mask_history = []

# ---------------- 마우스 콜백 ----------------
def draw_mask(event, x, y, flags, param):
    global drawing, mask, mask_history

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # 새로운 작업 시작시 히스토리 저장
        mask_history.append(mask.copy())
        if len(mask_history) > 10:  # 최대 10단계까지 undo
            mask_history.pop(0)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, BRUSH_COLOR, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# ---------------- 메인 함수 ----------------
def main():
    global mask, brush_size, mask_history

    # 이미지 파일 리스트
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
                        glob.glob(os.path.join(IMAGES_DIR, "*.png")))

    if len(image_files) == 0:
        print(f"Error: No images found in {IMAGES_DIR}")
        print("Please add images to label first!")
        return

    print("=" * 60)
    print("Segmentation Mask Labeling Tool")
    print("=" * 60)
    print(f"Found {len(image_files)} images to label")
    print("\nControls:")
    print("  - Mouse drag: Draw mask on line")
    print("  - 's': Save and next")
    print("  - 'c': Clear mask")
    print("  - 'u': Undo last drawing")
    print("  - '+/-': Increase/decrease brush size")
    print("  - 'q': Quit")
    print("=" * 60)

    cv2.namedWindow('Original')
    cv2.namedWindow('Mask')
    cv2.setMouseCallback('Mask', draw_mask)

    idx = 0

    while idx < len(image_files):
        img_path = image_files[idx]
        img_name = os.path.basename(img_path)

        print(f"\n[{idx+1}/{len(image_files)}] Labeling: {img_name}")

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading {img_path}, skipping...")
            idx += 1
            continue

        height, width = img.shape[:2]

        # 기존 마스크가 있으면 로드, 없으면 새로 생성
        mask_path = os.path.join(MASKS_DIR, img_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print("  (Loaded existing mask)")
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        mask_history = []

        while True:
            # 디스플레이용 이미지
            display_img = img.copy()
            display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # 마스크를 원본 이미지에 오버레이
            overlay = display_img.copy()
            overlay[mask > 0] = [0, 255, 0]  # 초록색으로 표시
            display_img = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)

            # 브러시 크기 표시
            cv2.putText(display_img, f"Brush: {brush_size}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_mask, f"Brush: {brush_size}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Original', display_img)
            cv2.imshow('Mask', display_mask)

            key = cv2.waitKey(1) & 0xFF

            # 's': 저장 및 다음
            if key == ord('s'):
                cv2.imwrite(mask_path, mask)
                print(f"  Saved: {mask_path}")
                idx += 1
                break

            # 'c': 초기화
            elif key == ord('c'):
                mask_history.append(mask.copy())
                mask = np.zeros_like(mask)
                print("  Mask cleared")

            # 'u': Undo
            elif key == ord('u'):
                if mask_history:
                    mask = mask_history.pop()
                    print("  Undo")
                else:
                    print("  Nothing to undo")

            # '+': 브러시 크기 증가
            elif key == ord('+') or key == ord('='):
                brush_size = min(50, brush_size + 2)
                print(f"  Brush size: {brush_size}")

            # '-': 브러시 크기 감소
            elif key == ord('-') or key == ord('_'):
                brush_size = max(5, brush_size - 2)
                print(f"  Brush size: {brush_size}")

            # 'q': 종료
            elif key == ord('q'):
                print("Quitting...")
                cv2.destroyAllWindows()
                return

            # 'n': 저장하지 않고 다음 (skip)
            elif key == ord('n'):
                print("  Skipped (not saved)")
                idx += 1
                break

            # 'p': 이전 이미지
            elif key == ord('p'):
                if idx > 0:
                    idx -= 1
                    break
                else:
                    print("  Already at first image")

    print("\n" + "=" * 60)
    print("All images labeled!")
    print(f"Masks saved in: {MASKS_DIR}")
    print("=" * 60)

    cv2.destroyAllWindows()

# ---------------- 간단한 자동 라벨링 (옵션) ----------------
def auto_label_by_color(img_path, mask_path, target_color='white'):
    """
    색상 기반 자동 라벨링 (흰색/검은색 라인)
    완벽하지 않으므로 수동 보정 필요
    """
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if target_color == 'black':
        # 검은색 라인 검출
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
    elif target_color == 'white':
        # 흰색 라인 검출
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
    else:
        raise ValueError("Unsupported color")

    mask = cv2.inRange(hsv, lower, upper)

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(mask_path, mask)
    print(f"Auto-labeled: {mask_path}")

def batch_auto_label(color='white'):
    """배치 자동 라벨링"""
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    print(f"Auto-labeling {len(image_files)} images for {color} line...")

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(MASKS_DIR, img_name)

        if not os.path.exists(mask_path):
            auto_label_by_color(img_path, mask_path, color)

    print("Auto-labeling complete! Please review and correct manually.")

# ---------------- 실행 ----------------
if __name__ == '__main__':
    import sys

    # 자동 라벨링 옵션
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        color = sys.argv[2] if len(sys.argv) > 2 else 'black'
        batch_auto_label(color)
    else:
        main()
