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
   - 'e': 지우개 모드 토글 (검은색으로 칠하기)
   - 'd': 현재 이미지와 마스크 삭제
   - 'q': 종료
   - '+/-': 브러시 크기 조절
"""

# ---------------- 설정 ----------------
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"
BRUSH_SIZE = 20  # 기본 브러시 크기를 크게 (끊김 방지)
BRUSH_COLOR = 255  # 흰색 (라인 영역)

# 마스크 디렉토리 생성
os.makedirs(MASKS_DIR, exist_ok=True)

# ---------------- 전역 변수 ----------------
drawing = False
brush_size = BRUSH_SIZE
mask = None
mask_history = []
prev_x, prev_y = -1, -1
eraser_mode = False  # 지우개 모드

# ---------------- 마우스 콜백 ----------------
def draw_mask(event, x, y, flags, param):
    global drawing, mask, mask_history, brush_size, prev_x, prev_y, eraser_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # 새로운 작업 시작시 히스토리 저장
        if mask is not None:
            mask_history.append(mask.copy())
            if len(mask_history) > 10:  # 최대 10단계까지 undo
                mask_history.pop(0)
            # 첫 점 그리기 (지우개 모드면 0, 아니면 255)
            color = 0 if eraser_mode else BRUSH_COLOR
            cv2.circle(mask, (x, y), brush_size, color, -1)
            prev_x, prev_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and mask is not None:
            # 이전 위치에서 현재 위치까지 선 그리기 (더 부드럽게)
            if prev_x >= 0 and prev_y >= 0:
                # 지우개 모드면 검은색(0), 아니면 흰색(255)
                color = 0 if eraser_mode else BRUSH_COLOR
                # 두꺼운 선으로 연결 (끊김 방지)
                cv2.line(mask, (prev_x, prev_y), (x, y), color, brush_size * 2)
                # 추가로 원도 그려서 완전히 메우기
                cv2.circle(mask, (x, y), brush_size, color, -1)
            prev_x, prev_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_x, prev_y = -1, -1

# ---------------- 메인 함수 ----------------
def main():
    global mask, brush_size, mask_history, eraser_mode

    # 이미지 파일 리스트
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
                        glob.glob(os.path.join(IMAGES_DIR, "*.png")))

    if len(image_files) == 0:
        print(f"Error: No images found in {IMAGES_DIR}")
        print("Please add images to label first!")
        return

    # 진행 상황 파악
    labeled_count = 0
    for img_file in image_files:
        img_name = os.path.basename(img_file)
        mask_path = os.path.join(MASKS_DIR, img_name)
        if os.path.exists(mask_path):
            labeled_count += 1

    print("=" * 60)
    print("Segmentation Mask Labeling Tool")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Already labeled: {labeled_count}")
    print(f"Unlabeled: {len(image_files) - labeled_count}")

    if labeled_count == len(image_files):
        print("\n✓ All images have masks. You can review and edit them.")

    print("\nControls:")
    print("  - Mouse drag: Draw mask on line")
    print("  - 's': Save and next")
    print("  - 'c': Clear mask")
    print("  - 'u': Undo last drawing")
    print("  - 'e': Toggle eraser mode (draw with black)")
    print("  - 'd': Delete current image and mask")
    print("  - '+/-': Increase/decrease brush size")
    print("  - 'n': Skip without saving")
    print("  - 'p': Go to previous image")
    print("  - 'q': Quit")
    print("=" * 60)

    # 항상 처음부터 시작 (이미 있는 마스크도 수정 가능)
    start_idx = 0

    cv2.namedWindow('Original')
    cv2.namedWindow('Mask')
    cv2.setMouseCallback('Mask', draw_mask)

    idx = start_idx

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

        # 마우스 콜백 재설정 (각 이미지마다)
        cv2.setMouseCallback('Mask', draw_mask)

        # 디스플레이용 이미지를 한 번만 생성 (루프 밖)
        # 원본은 변하지 않으므로 미리 준비
        base_img = img.copy()

        # 프레임 카운터 (화면 업데이트 최적화)
        frame_count = 0

        while True:
            frame_count += 1

            # 매 프레임마다 업데이트하지 않고 3프레임마다 업데이트 (렉 감소)
            if frame_count % 3 == 0 or drawing:
                # 디스플레이용 이미지
                display_img = base_img.copy()
                display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # 마스크를 원본 이미지에 오버레이
                overlay = display_img.copy()
                overlay[mask > 0] = [0, 255, 0]  # 초록색으로 표시
                display_img = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)

                # 브러시 크기 및 안내 표시
                mode_text = "ERASER" if eraser_mode else "DRAW"
                mode_color = (0, 0, 255) if eraser_mode else (0, 255, 0)

                cv2.putText(display_img, f"Brush: {brush_size} | Mode: {mode_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_img, f"[{idx+1}/{len(image_files)}]", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(display_mask, f"Brush: {brush_size} | Mode: {mode_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 드래그 중일 때만 상태 표시 (텍스트 렌더링 줄이기)
                if drawing:
                    cv2.putText(display_mask, "DRAWING!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                else:
                    cv2.putText(display_mask, "Ready", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

                cv2.imshow('Original', display_img)
                cv2.imshow('Mask', display_mask)

            # 화면 업데이트 속도 조절 (10ms로 증가하여 CPU 부하 감소)
            key = cv2.waitKey(10) & 0xFF

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

            # 'e': 지우개 모드 토글
            elif key == ord('e'):
                eraser_mode = not eraser_mode
                mode_name = "ERASER" if eraser_mode else "DRAW"
                print(f"  Mode changed: {mode_name}")

            # 'd': 이미지와 마스크 삭제
            elif key == ord('d'):
                print(f"  Delete '{img_name}'? Press 'y' to confirm, any other key to cancel.")
                confirm_key = cv2.waitKey(0) & 0xFF
                if confirm_key == ord('y'):
                    # 마스크 삭제
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
                        print(f"    Deleted mask: {mask_path}")
                    # 이미지 삭제
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"    Deleted image: {img_path}")
                    print("  File deleted. Moving to next image...")
                    idx += 1
                    break
                else:
                    print("  Deletion cancelled.")

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

# ---------------- 개선된 자동 라벨링 (정확도 향상) ----------------
def auto_label_by_color(img_path, mask_path, target_color='white'):
    """
    고정밀 자동 라벨링 파이프라인 (85-95% 정확도 목표)

    개선 사항:
    - Multi-scale HSV detection (다양한 조명 대응)
    - Adaptive thresholding per region (지역별 최적화)
    - Probabilistic Line Continuity (끊어진 라인 연결)
    - Skeleton-based line width estimation (정확한 두께)
    - Weighted mask fusion (신뢰도 기반 결합)

    Args:
        img_path: 원본 이미지 경로
        mask_path: 저장할 마스크 경로
        target_color: 'white' 또는 'black'
    """
    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load {img_path}")
        return

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ========== Step 1: 향상된 전처리 ==========
    # 1.1 Bilateral Filter (엣지 보존하면서 노이즈 제거)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 1.2 CLAHE with larger tile size (더 부드러운 조명 보정)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(denoised)

    # ========== Step 2: Multi-scale HSV Detection ==========
    # 다양한 조명 조건에 대응하기 위해 3단계 임계값 사용
    masks_hsv = []

    if target_color == 'white':
        # 흰색: 밝은 정도에 따라 3단계
        hsv_ranges = [
            ([0, 0, 220], [180, 25, 255]),  # 매우 밝음
            ([0, 0, 180], [180, 40, 255]),  # 밝음
            ([0, 0, 150], [180, 50, 240]),  # 중간 밝음
        ]
    else:  # black
        hsv_ranges = [
            ([0, 0, 0], [180, 255, 40]),    # 매우 어두움
            ([0, 0, 0], [180, 255, 60]),    # 어두움
            ([0, 0, 40], [180, 255, 80]),   # 중간 어두움
        ]

    for lower, upper in hsv_ranges:
        mask_temp = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masks_hsv.append(mask_temp)

    # 가중 평균 결합 (밝은/어두운 것에 더 높은 가중치)
    mask_hsv = cv2.addWeighted(masks_hsv[0], 0.5, masks_hsv[1], 0.3, 0)
    mask_hsv = cv2.addWeighted(mask_hsv, 1.0, masks_hsv[2], 0.2, 0)
    mask_hsv = (mask_hsv > 127).astype(np.uint8) * 255

    # ========== Step 3: Region-based Adaptive Thresholding ==========
    # 이미지를 3개 영역으로 나누어 각각 최적화
    roi_height = height // 3
    mask_adaptive = np.zeros_like(gray)

    for i in range(3):
        y_start = i * roi_height
        y_end = (i + 1) * roi_height if i < 2 else height
        region = enhanced[y_start:y_end, :]

        if target_color == 'white':
            # 흰색: 각 영역별로 다른 파라미터
            block_size = 15 if i == 0 else 13  # 상단은 더 큰 블록
            c_value = -3 if i == 0 else -5     # 상단은 덜 공격적
            mask_region = cv2.adaptiveThreshold(
                region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_value
            )
        else:  # black
            block_size = 15 if i == 0 else 13
            c_value = 3 if i == 0 else 5
            mask_region = cv2.adaptiveThreshold(
                region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, c_value
            )

        mask_adaptive[y_start:y_end, :] = mask_region

    # ========== Step 4: Enhanced Edge Detection ==========
    # Otsu's thresholding을 이용한 자동 임계값 설정
    high_thresh, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = high_thresh * 0.4
    edges = cv2.Canny(enhanced, low_thresh, high_thresh)

    # Edge 확장 (더 연속적으로)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)

    # ========== Step 5: 초기 마스크 결합 (가중치 적용) ==========
    # HSV에 가장 높은 가중치 (색상 정보가 가장 정확)
    mask_combined = cv2.addWeighted(mask_hsv, 0.5, mask_adaptive, 0.3, 0)
    mask_combined = cv2.addWeighted(mask_combined, 1.0, edges_dilated, 0.2, 0)
    mask_combined = (mask_combined > 127).astype(np.uint8) * 255

    # ========== Step 6: Advanced Hough Line with Continuity ==========
    # ROI 설정: 전체 화면 사용 (상단도 포함)
    roi_mask = mask_combined.copy()

    # Probabilistic Hough Line with optimized parameters
    lines = cv2.HoughLinesP(
        roi_mask,
        rho=1,
        theta=np.pi/180,
        threshold=30,        # 임계값 낮춤 (더 많은 라인 검출)
        minLineLength=20,    # 최소 길이 줄임
        maxLineGap=40        # 갭 증가 (끊어진 라인 연결)
    )

    # 라인 클러스터링 및 연결
    mask_lines = np.zeros_like(gray)

    if lines is not None:
        # 라인을 각도별로 그룹화
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 각도 계산
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)

            # 수평에 가까운 라인만 (0~25도 or 155~180도)
            if angle < 25 or angle > 155:
                horizontal_lines.append((x1, y1, x2, y2))

        # 라인 연결: 가까운 라인끼리 연결
        connected_lines = []
        used = [False] * len(horizontal_lines)

        for i, (x1, y1, x2, y2) in enumerate(horizontal_lines):
            if used[i]:
                continue

            # 현재 라인 세그먼트 시작
            segment = [(x1, y1), (x2, y2)]
            used[i] = True

            # 가까운 라인 찾기
            changed = True
            while changed:
                changed = False
                for j, (x3, y3, x4, y4) in enumerate(horizontal_lines):
                    if used[j]:
                        continue

                    # 끝점들 간 거리 체크
                    seg_start = segment[0]
                    seg_end = segment[-1]

                    dist1 = np.sqrt((seg_end[0] - x3)**2 + (seg_end[1] - y3)**2)
                    dist2 = np.sqrt((seg_end[0] - x4)**2 + (seg_end[1] - y4)**2)
                    dist3 = np.sqrt((seg_start[0] - x3)**2 + (seg_start[1] - y3)**2)
                    dist4 = np.sqrt((seg_start[0] - x4)**2 + (seg_start[1] - y4)**2)

                    min_dist = min(dist1, dist2, dist3, dist4)

                    # 60픽셀 이내면 연결
                    if min_dist < 60:
                        if min_dist == dist1:
                            segment.append((x3, y3))
                            segment.append((x4, y4))
                        elif min_dist == dist2:
                            segment.append((x4, y4))
                            segment.append((x3, y3))
                        elif min_dist == dist3:
                            segment.insert(0, (x4, y4))
                            segment.insert(0, (x3, y3))
                        else:
                            segment.insert(0, (x3, y3))
                            segment.insert(0, (x4, y4))

                        used[j] = True
                        changed = True

            connected_lines.append(segment)

        # 연결된 라인 그리기
        for segment in connected_lines:
            pts = np.array(segment, dtype=np.int32)
            # Polylines로 부드럽게 연결
            cv2.polylines(mask_lines, [pts], False, 255, thickness=20)

    # Hough 라인을 원래 마스크와 결합 (높은 가중치)
    mask_final = cv2.addWeighted(mask_combined, 0.5, mask_lines, 0.5, 0)
    mask_final = (mask_final > 100).astype(np.uint8) * 255

    # ========== Step 7: Morphology-based Line Width Estimation ==========
    # Distance Transform으로 라인 폭 추정
    dist_transform = cv2.distanceTransform(mask_final, cv2.DIST_L2, 5)

    # 거리 변환의 중앙값으로 라인 폭 추정
    nonzero_dists = dist_transform[mask_final > 0]
    if len(nonzero_dists) > 0:
        avg_width = int(np.median(nonzero_dists) * 2.5)  # 약간 더 두껍게
        avg_width = max(12, min(28, avg_width))  # 12~28 픽셀 범위로 제한
    else:
        avg_width = 18  # 기본값

    # 스켈레톤화 (Zhang-Suen algorithm 대체)
    # Erosion 후 재구성하는 방식
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(mask_final, kernel_thin, iterations=3)

    # 얇아진 마스크를 추정된 두께로 확장
    kernel_width = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (avg_width, avg_width))
    mask_final = cv2.dilate(eroded, kernel_width, iterations=1)

    # ========== Step 8: Advanced Post-processing ==========
    # 8.1 Closing (구멍 메우기) - 더 큰 커널
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 8.2 Opening (노이즈 제거)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel_open)

    # 8.3 Connected Components with Area and Aspect Ratio Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_final, connectivity=8
    )

    # 배경 제외
    if num_labels > 1:
        valid_labels = []

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]

            # 필터링 조건
            # 1. 최소 면적: 500픽셀
            if area < 500:
                continue

            # 2. 가로세로 비율: 라인은 가로로 길어야 함 (w > h)
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 1.5:  # 가로가 세로의 1.5배 이상
                continue

            # 3. 위치 필터링 제거 - 전체 영역 허용

            valid_labels.append(label_id)

        # 유효한 컴포넌트만 유지 (최대 2개)
        if valid_labels:
            # 면적 기준 상위 2개 선택
            areas = [stats[lid, cv2.CC_STAT_AREA] for lid in valid_labels]
            sorted_indices = np.argsort(areas)[::-1]
            top_labels = [valid_labels[i] for i in sorted_indices[:2]]

            mask_filtered = np.zeros_like(mask_final)
            for label_id in top_labels:
                mask_filtered[labels == label_id] = 255

            mask_final = mask_filtered
        else:
            # 유효한 컴포넌트가 없으면 가장 큰 것 사용
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            mask_final = np.zeros_like(mask_final)
            mask_final[labels == largest_label] = 255

    # 8.4 Final smoothing (부드러운 경계)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel_smooth)

    # ROI 제한 없음 (전체 영역 사용)

    # ========== 저장 ==========
    cv2.imwrite(mask_path, mask_final)

    # 디버그 정보 출력
    line_pixels = np.sum(mask_final > 0)
    coverage = (line_pixels / (height * width)) * 100
    print(f"✓ Auto-labeled: {os.path.basename(img_path)} (coverage: {coverage:.2f}%, width: {avg_width}px)")

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
    print("\nNext step:")
    print("  python3 label_segmentation.py --review")
    print("Or:")
    print("  python3 label_segmentation.py")

# ---------------- 실행 ----------------
if __name__ == '__main__':
    import sys

    # 자동 라벨링 옵션
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        color = sys.argv[2] if len(sys.argv) > 2 else 'white'
        batch_auto_label(color)
    else:
        main()
