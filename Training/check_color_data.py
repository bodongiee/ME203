#!/usr/bin/env python3
# check_color_data.py
# 색상 데이터 검증 스크립트

import cv2
import numpy as np
import os

DATA_DIR = "../lab8_materials/data_color"

print("="*60)
print("색상 데이터 검증")
print("="*60)

green_dir = os.path.join(DATA_DIR, "green")
red_dir = os.path.join(DATA_DIR, "red")

# 샘플 이미지 로드
print("\n[Green 샘플]")
green_files = sorted([f for f in os.listdir(green_dir) if f.endswith('.png')])[:5]
green_samples = []
for f in green_files:
    img = cv2.imread(os.path.join(green_dir, f))
    if img is not None:
        green_samples.append(img)
        # RGB 평균값 계산
        mean_color = img.mean(axis=(0, 1))  # BGR
        print(f"  {f}: BGR평균 = {mean_color}")

print("\n[Red 샘플]")
red_files = sorted([f for f in os.listdir(red_dir) if f.endswith('.png')])[:5]
red_samples = []
for f in red_files:
    img = cv2.imread(os.path.join(red_dir, f))
    if img is not None:
        red_samples.append(img)
        # RGB 평균값 계산
        mean_color = img.mean(axis=(0, 1))  # BGR
        print(f"  {f}: BGR평균 = {mean_color}")

# HSV 분석
print("\n[HSV 분석]")
print("Green HSV 평균:")
for i, img in enumerate(green_samples[:3]):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:,:,0].mean()
    s_mean = hsv[:,:,1].mean()
    v_mean = hsv[:,:,2].mean()
    print(f"  Sample {i}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")

print("\nRed HSV 평균:")
for i, img in enumerate(red_samples[:3]):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:,:,0].mean()
    s_mean = hsv[:,:,1].mean()
    v_mean = hsv[:,:,2].mean()
    print(f"  Sample {i}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")

print("\n[결론]")
print("Green: H는 40~80 범위여야 함 (초록색)")
print("Red: H는 0~10 또는 160~180 범위여야 함 (빨간색)")
print("\n만약 두 클래스의 H 값이 비슷하면 데이터가 잘못됨!")
