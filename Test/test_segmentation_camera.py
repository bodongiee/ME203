#!/usr/bin/env python3
# test_segmentation_camera.py
# PiCamera로 실시간 Segmentation 모델 테스트

import time
import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ---------------- 설정 ----------------
TFLITE_MODEL_PATH = "../line_segmentation.tflite"
IMG_SIZE = 160  # 모델 입력 크기 (자동으로 조정됨)

# 디스플레이 설정
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# FPS 계산용
frame_count = 0
start_time = time.time()
fps = 0

# ---------------- TFLite 모델 로드 ----------------
print("[INFO] Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"[INFO] Model loaded successfully")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# INT8 양자화 모델 확인
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
is_int8_model = (input_dtype == np.uint8)
print(f"[INFO] INT8 quantized model: {is_int8_model}")

# 모델 입력 크기 자동 확인
model_input_size = input_details[0]['shape'][1]  # [1, height, width, 3]
if model_input_size != IMG_SIZE:
    print(f"[WARNING] IMG_SIZE ({IMG_SIZE}) != model input size ({model_input_size})")
    IMG_SIZE = model_input_size
    print(f"[INFO] Updated IMG_SIZE to {IMG_SIZE}")

# ---------------- PiCamera2 초기화 ----------------
print("[INFO] Initializing PiCamera2...")
picam2 = Picamera2()

# 카메라 설정
config = picam2.create_preview_configuration(
    main={"size": (IMG_SIZE, IMG_SIZE), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("[INFO] Camera started")
print("[INFO] Press 'q' to quit")
print("[INFO] Press 's' to save current frame and mask")

# 카메라 워밍업
time.sleep(2)

# ---------------- 메인 루프 ----------------
try:
    while True:
        loop_start = time.time()

        # 프레임 캡처
        frame = picam2.capture_array()  # (240, 240, 3) RGB

        # 전처리 (INT8/FLOAT32 구분)
        if is_int8_model:
            # INT8 양자화 모델: 0-255 uint8 그대로 사용
            input_data = frame.astype(np.uint8)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            # FLOAT32 모델: 0-1 정규화
            input_data = frame.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

        # 추론
        inference_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        inference_time = (time.time() - inference_start) * 1000

        # 출력 처리
        mask = output_data[0].squeeze()  # (240, 240)

        # INT8 양자화 출력 역양자화
        if output_dtype == np.uint8 or output_dtype == np.int8:
            output_quant = output_details[0].get('quantization_parameters', {})
            output_scale = output_quant.get('scales', [1.0])[0]
            output_zero_point = output_quant.get('zero_points', [0])[0]

            # 역양자화 + Sigmoid
            mask = (mask.astype(np.float32) - output_zero_point) * output_scale
            mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -10, 10)))

        # 이진화 (threshold=0.5)
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # ---------------- 시각화 ----------------
        # 원본 이미지 (BGR로 변환)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 마스크를 컬러로 변환
        mask_color = cv2.applyColorMap(mask_binary, cv2.COLORMAP_JET)

        # 오버레이 (원본 + 마스크)
        overlay = cv2.addWeighted(frame_bgr, 0.6, mask_color, 0.4, 0)

        # 라인 중심점 계산
        if np.sum(mask_binary) > 100:  # 최소 픽셀 수
            moments = cv2.moments(mask_binary)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                # 중심점 표시
                cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
                cv2.circle(frame_bgr, (cx, cy), 5, (0, 255, 0), -1)

                # 중심선에서의 오차
                center_error = (cx - IMG_SIZE // 2) / (IMG_SIZE // 2)
                cv2.putText(overlay, f"Error: {center_error:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 화면 중앙선 표시
        cv2.line(frame_bgr, (IMG_SIZE//2, 0), (IMG_SIZE//2, IMG_SIZE), (0, 255, 255), 1)
        cv2.line(overlay, (IMG_SIZE//2, 0), (IMG_SIZE//2, IMG_SIZE), (0, 255, 255), 1)

        # FPS 계산
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed

        # 정보 표시
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame_bgr, f"Inference: {inference_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"Inference: {inference_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 마스크 정보
        line_pixels = np.sum(mask_binary > 0)
        line_ratio = line_pixels / (IMG_SIZE * IMG_SIZE) * 100
        cv2.putText(overlay, f"Line: {line_ratio:.1f}%", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 3개 화면을 나란히 배치
        # 1. 원본, 2. 마스크, 3. 오버레이
        mask_bgr = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)

        # 수평으로 결합
        combined = np.hstack([frame_bgr, mask_bgr, overlay])

        # 리사이즈 (디스플레이용)
        display = cv2.resize(combined, (DISPLAY_WIDTH * 3, DISPLAY_HEIGHT))

        # 화면 표시
        cv2.imshow("Segmentation Test: [Original | Mask | Overlay]", display)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quit requested")
            break
        elif key == ord('s'):
            # 현재 프레임 저장
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"test_frame_{timestamp}.jpg", frame_bgr)
            cv2.imwrite(f"test_mask_{timestamp}.jpg", mask_binary)
            cv2.imwrite(f"test_overlay_{timestamp}.jpg", overlay)
            print(f"[SAVE] Saved: test_*_{timestamp}.jpg")

        # 루프 시간 계산
        loop_time = (time.time() - loop_start) * 1000
        # print(f"Loop time: {loop_time:.1f}ms")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    # 정리
    print("[INFO] Cleaning up...")
    picam2.stop()
    cv2.destroyAllWindows()

    # 최종 통계
    total_time = time.time() - start_time
    print(f"\n[STATS] Total frames: {frame_count}")
    print(f"[STATS] Average FPS: {frame_count / total_time:.2f}")
    print("[INFO] Done")
