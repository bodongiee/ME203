#!/usr/bin/env python3
# test_lightweight_camera.py
# 초경량 INT8 모델 테스트 (라즈베리파이 전용)

import time
import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ---------------- 설정 ----------------
TFLITE_MODEL_PATH = "../line_segmentation_light.tflite"
IMG_SIZE = 160  # 경량 모델은 160x160

# 디스플레이 설정
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 480

# FPS 계산용
frame_count = 0
start_time = time.time()
fps = 0

print("="*60)
print("초경량 Segmentation 모델 테스트")
print(f"모델: {TFLITE_MODEL_PATH}")
print(f"이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print("="*60)

# ---------------- TFLite 모델 로드 ----------------
print("\n[INFO] Loading TFLite INT8 model...")
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✓ Model loaded")
print(f"  Input:  {input_details[0]['shape']} {input_details[0]['dtype']}")
print(f"  Output: {output_details[0]['shape']} {output_details[0]['dtype']}")

# INT8 모델 체크
is_int8_model = input_details[0]['dtype'] == np.uint8

if is_int8_model:
    print(f"  ✓ INT8 quantized model detected")
    # Quantization 파라미터
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    print(f"  Input scale: {input_scale:.6f}, zero_point: {input_zero_point}")
    print(f"  Output scale: {output_scale:.6f}, zero_point: {output_zero_point}")
else:
    print(f"  Warning: Not an INT8 model (using float)")

# ---------------- PiCamera2 초기화 ----------------
print("\n[INFO] Initializing PiCamera2...")
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (IMG_SIZE, IMG_SIZE), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("✓ Camera started")
print("\nControls:")
print("  'q' - Quit")
print("  's' - Save screenshot")
print("  't' - Toggle view mode (overlay/split)")
print("-"*60)

time.sleep(1)

# 뷰 모드
view_mode = "overlay"  # "overlay" or "split"

# ---------------- 메인 루프 ----------------
try:
    while True:
        loop_start = time.time()

        # 프레임 캡처
        frame = picam2.capture_array()  # (160, 160, 3) RGB

        # 전처리 (INT8 모델용)
        if is_int8_model:
            # 0~255 uint8 그대로 사용
            input_data = frame.astype(np.uint8)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            # Float 모델용
            input_data = frame.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

        # 추론
        inference_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        inference_time = (time.time() - inference_start) * 1000

        # 출력 처리
        mask = output_data[0].squeeze()  # (160, 160)

        if is_int8_model:
            # INT8 출력 (0~255) → 이진화
            mask_binary = (mask > 127).astype(np.uint8) * 255
        else:
            # Float 출력 (0~1) → 이진화
            mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # BGR 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ---------------- 시각화 ----------------
        if view_mode == "overlay":
            # 오버레이 모드 (라인을 초록색으로)
            display = frame_bgr.copy()
            green_overlay = np.zeros_like(frame_bgr)
            green_overlay[:, :, 1] = mask_binary
            display = cv2.addWeighted(display, 0.7, green_overlay, 0.3, 0)

        else:
            # 분할 모드 (원본 | 마스크)
            mask_bgr = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
            display = np.hstack([frame_bgr, mask_bgr])

        # 라인 중심점 계산 및 표시
        if np.sum(mask_binary) > 50:
            moments = cv2.moments(mask_binary)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                # 중심점
                cv2.drawMarker(display if view_mode == "overlay" else frame_bgr,
                              (cx, cy), (0, 0, 255),
                              cv2.MARKER_CROSS, 15, 2)

                # 중심에서의 오차
                center_error = (cx - IMG_SIZE // 2) / (IMG_SIZE // 2)
                error_px = cx - IMG_SIZE // 2

                # 오차선
                if view_mode == "overlay":
                    cv2.line(display, (IMG_SIZE//2, cy), (cx, cy), (255, 0, 0), 2)

        # 화면 중앙선
        mid_x = IMG_SIZE // 2 if view_mode == "overlay" else IMG_SIZE // 2
        cv2.line(display if view_mode == "overlay" else frame_bgr,
                (mid_x, 0), (mid_x, IMG_SIZE), (0, 255, 255), 1, cv2.LINE_DASHED)

        # FPS 계산
        frame_count += 1
        if frame_count % 5 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed

        # 정보 패널 생성
        info_height = 80
        info_panel = np.zeros((info_height, IMG_SIZE * (2 if view_mode == "split" else 1), 3), dtype=np.uint8)

        # FPS (색상: 초록 > 10, 노랑 5~10, 빨강 < 5)
        fps_color = (0, 255, 0) if fps >= 10 else (0, 255, 255) if fps >= 5 else (0, 0, 255)
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # Inference time (색상: 초록 < 100ms, 노랑 100~200, 빨강 > 200)
        inf_color = (0, 255, 0) if inference_time < 100 else (0, 255, 255) if inference_time < 200 else (0, 0, 255)
        cv2.putText(info_panel, f"Inference: {inference_time:.1f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, inf_color, 1)

        # 라인 정보
        if np.sum(mask_binary) > 50:
            line_ratio = np.sum(mask_binary > 0) / (IMG_SIZE * IMG_SIZE) * 100
            cv2.putText(info_panel, f"Line: {line_ratio:.1f}% | Error: {error_px:+d}px", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(info_panel, "No line detected", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 결합
        final_display = np.vstack([info_panel, display])

        # 리사이즈 (디스플레이용)
        final_display = cv2.resize(final_display, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # 표시
        cv2.imshow(f"Lightweight Segmentation [{view_mode.upper()}]", final_display)

        # 키 입력
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quit requested")
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"test_{timestamp}.jpg", display)
            cv2.imwrite(f"mask_{timestamp}.jpg", mask_binary)
            print(f"[SAVE] Saved: test_{timestamp}.jpg, mask_{timestamp}.jpg")
        elif key == ord('t'):
            view_mode = "split" if view_mode == "overlay" else "overlay"
            print(f"[VIEW] Mode: {view_mode}")

        # 루프 시간
        loop_time = (time.time() - loop_start) * 1000

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()

    # 최종 통계
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average inference time: {1000/avg_fps:.1f}ms")

    if avg_fps >= 10:
        print(f"\n✓ SUCCESS: {avg_fps:.1f} FPS - 실시간 추론 가능!")
    elif avg_fps >= 5:
        print(f"\n⚠ OK: {avg_fps:.1f} FPS - 동작 가능하지만 개선 필요")
    else:
        print(f"\n✗ SLOW: {avg_fps:.1f} FPS - 너무 느림")

    print("="*60)
