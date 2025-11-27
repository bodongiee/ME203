#!/home/mecha/venvs/tflite/bin/python
# test_color_only.py
# 색상 감지만 테스트 (모터 없이)

import numpy as np
import cv2
import time
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ---------------- 설정 ----------------
COLOR_SIZE = 160  # 64 → 160 (새 모델)
COLOR_MODEL_PATH = "./color_light_160.tflite"  # 새 모델

print("="*60)
print("색상 감지 테스트 (160x160 모델)")
print("="*60)

# ---------------- TFLite 모델 로드 ----------------
print("\n[INFO] Loading color model...")
color_interpreter = tflite.Interpreter(model_path=COLOR_MODEL_PATH)
color_interpreter.allocate_tensors()

color_input_details = color_interpreter.get_input_details()[0]
color_output_details = color_interpreter.get_output_details()[0]

color_is_int8 = (color_input_details['dtype'] == np.uint8)
print(f"✓ Color model loaded (INT8: {color_is_int8})")
print(f"  Input shape: {color_input_details['shape']}")
print(f"  Output shape: {color_output_details['shape']}")

# ---------------- 카메라 초기화 ----------------
print("\n[INFO] Initializing camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)
print("✓ Camera ready")

print("\n[START] Press Ctrl+C to stop")
print("-"*60)

# ---------------- 메인 루프 ----------------
try:
    frame_count = 0
    start_time = time.time()

    while True:
        # 1. 프레임 캡처
        frame = picam2.capture_array()

        # 2. 색상 감지용 전처리 (64x64 크롭)
        h, w = frame.shape[:2]
        y_color = (h - COLOR_SIZE) // 2
        x_color = (w - COLOR_SIZE) // 2
        color_cropped = frame[y_color:y_color+COLOR_SIZE, x_color:x_color+COLOR_SIZE]

        if color_is_int8:
            color_input = color_cropped.astype(np.uint8)
        else:
            color_input = color_cropped.astype(np.float32) / 255.0
        color_input = np.expand_dims(color_input, axis=0)

        # 3. 추론
        color_interpreter.set_tensor(color_input_details['index'], color_input)
        color_interpreter.invoke()
        color_output = color_interpreter.get_tensor(color_output_details['index'])[0]

        # INT8 역양자화
        if color_output_details['dtype'] == np.uint8 or color_output_details['dtype'] == np.int8:
            color_quant = color_output_details.get('quantization_parameters', {})
            color_scale = color_quant.get('scales', [1.0])[0]
            color_zero = color_quant.get('zero_points', [0])[0]
            color_output = (color_output.astype(np.float32) - color_zero) * color_scale

        # Softmax
        exp_vals = np.exp(color_output - np.max(color_output))
        color_probs = exp_vals / np.sum(exp_vals)
        color_idx = np.argmax(color_probs)
        color_conf = color_probs[color_idx]
        color_name = ["green", "red"][color_idx]

        # 4. 출력
        print(f"Frame {frame_count:4d} | Color: {color_name:5s} ({color_conf:.3f}) | green={color_probs[0]:.3f}, red={color_probs[1]:.3f}")

        # 빨간불/초록불 감지 알림 (신뢰도 > 70%)
        if color_name == "red" and color_conf > 0.7:
            print("  >>> 🔴 RED DETECTED! <<<")
        elif color_name == "green" and color_conf > 0.7:
            print("  >>> 🟢 GREEN DETECTED! <<<")

        frame_count += 1
        time.sleep(0.1)  # 10Hz

except KeyboardInterrupt:
    print("\n\n[INFO] Stopping...")

finally:
    picam2.stop()

    # 통계
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print("="*60)
