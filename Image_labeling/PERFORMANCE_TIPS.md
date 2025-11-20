# 라벨링 도구 성능 최적화 팁

## 이미 적용된 최적화

✅ 선택적 화면 업데이트 (3프레임마다)
✅ 원본 이미지 캐싱
✅ waitKey 시간 증가 (10ms)
✅ 불필요한 렌더링 제거

## 추가 최적화 방법

### 1. 이미지 리사이즈 (권장)

이미지가 너무 크면 렉이 발생합니다.

**label_segmentation.py 수정:**

```python
# 이미지 로드 부분 (143번째 줄 근처)
img = cv2.imread(img_path)

# 다음을 추가:
max_size = 800  # 최대 크기 제한
h, w = img.shape[:2]
if max(h, w) > max_size:
    scale = max_size / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    # 마스크도 동일하게 리사이즈 필요
```

### 2. 윈도우 크기 제한

```python
# 윈도우 생성 시
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original', 640, 480)
cv2.resizeWindow('Mask', 640, 480)
```

### 3. 오버레이 업데이트 빈도 조절

**현재 코드 수정:**

```python
# 대기 중: 5프레임마다
if frame_count % 5 == 0 or drawing:
    # 렌더링
```

### 4. Original 창 제거 (가장 효과적)

마스크만 보고 작업:

```python
# Original 창 주석 처리
# cv2.imshow('Original', display_img)
cv2.imshow('Mask', display_mask)
```

### 5. 마스크 해상도 낮추기

학습용 마스크는 240x240이므로 미리 낮춰서 작업:

```python
# 이미지 로드 후
img = cv2.resize(img, (240, 240))
mask = np.zeros((240, 240), dtype=np.uint8)
```

## 성능 측정

현재 FPS 확인:

```python
import time

fps_start = time.time()
fps_frames = 0

while True:
    fps_frames += 1

    if time.time() - fps_start >= 1.0:
        print(f"FPS: {fps_frames}")
        fps_frames = 0
        fps_start = time.time()
```

## 하드웨어별 권장 설정

### 고사양 (최근 Mac/PC)
- 프레임 건너뛰기: 3프레임마다
- waitKey: 10ms
- 이미지 크기: 원본

### 중사양
- 프레임 건너뛰기: 5프레임마다
- waitKey: 15ms
- 이미지 크기: 최대 800px

### 저사양 (오래된 PC)
- 프레임 건너뛰기: 10프레임마다
- waitKey: 20ms
- 이미지 크기: 최대 640px
- Original 창 제거

## 라즈베리파이에서는?

라벨링은 PC에서 하고, 모델만 라즈베리파이에서 사용하세요!
