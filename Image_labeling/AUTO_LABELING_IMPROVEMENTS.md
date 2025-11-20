# 자동 라벨링 정확도 개선 (v2.0)

## 개선 목표
기존 60-70% → **85-95% 정확도**로 향상하여 수동 수정 시간 최소화

## 주요 개선 사항

### 1. Multi-scale HSV Detection
**문제**: 조명 조건에 따라 흰색 라인의 밝기가 달라짐
**해결**: 3단계 밝기 임계값 사용 + 가중 평균 결합
```
- 매우 밝음: V=220~255 (가중치 0.5)
- 밝음: V=180~255 (가중치 0.3)
- 중간 밝음: V=150~240 (가중치 0.2)
```

### 2. Region-based Adaptive Thresholding
**문제**: 전체 이미지에 동일한 임계값 적용 시 일부 영역 누락
**해결**: 이미지를 상/중/하 3개 영역으로 나누어 각각 최적화
- 상단: 더 큰 블록 크기 (15x15), 덜 공격적 임계값
- 하단: 더 작은 블록 (13x13), 더 공격적 임계값

### 3. Probabilistic Line Continuity
**문제**: 라인이 끊어져 있을 때 검출 실패
**해결**:
- Hough Line 파라미터 최적화 (minLineLength=20, maxLineGap=40)
- 60픽셀 이내의 라인 세그먼트 자동 연결
- Polylines로 부드럽게 이어그리기

### 4. Skeleton-based Line Width Estimation
**문제**: 고정된 두께로 라벨링하면 실제 라인과 차이 발생
**해결**:
- Distance Transform으로 각 픽셀까지의 거리 계산
- 스켈레톤 상의 중앙값으로 실제 라인 폭 추정 (10~30픽셀)
- 추정된 폭으로 정확하게 확장

### 5. Advanced Connected Components Filtering
**문제**: 노이즈가 라인으로 잘못 검출됨
**해결**: 3가지 조건으로 필터링
- 최소 면적: 500픽셀 이상
- 가로세로 비율: 1.5:1 이상 (가로로 긴 형태만)
- 위치: 화면 하단 75%에만 존재

### 6. Weighted Mask Fusion
**문제**: 여러 검출 방법의 결과를 단순 OR 연산하면 노이즈 증가
**해결**: 신뢰도 기반 가중치 적용
- HSV: 0.5 (가장 신뢰)
- Adaptive: 0.3
- Edge: 0.2

### 7. Enhanced Preprocessing
**기존**: Gaussian Blur + CLAHE
**개선**:
- Bilateral Filter (엣지 보존하면서 노이즈 제거)
- 더 큰 CLAHE 타일 크기 (16x16) - 부드러운 조명 보정

### 8. Otsu's Auto-thresholding for Canny
**문제**: 고정 임계값은 이미지마다 다른 결과
**해결**: Otsu 알고리즘으로 자동 임계값 계산 후 Canny 적용

## 사용 방법

### 자동 라벨링 실행
```bash
cd Image_labeling
python3 label_segmentation.py --auto white
```

### 수동 수정
```bash
python3 label_segmentation.py
```

### 출력 예시
```
✓ Auto-labeled: frame_1004.png (coverage: 8.45%, width: 18px)
✓ Auto-labeled: frame_1005.png (coverage: 9.12%, width: 22px)
✓ Auto-labeled: frame_1006.png (coverage: 7.89%, width: 16px)
```

## 기대 효과

| 항목 | 기존 (v1.0) | 개선 (v2.0) |
|------|-------------|-------------|
| 정확도 | 60-70% | 85-95% |
| 수동 수정 필요 | 대부분 | 10-15% |
| 라인 연속성 | 자주 끊김 | 거의 연속 |
| 조명 강건성 | 낮음 | 높음 |
| 라인 폭 정확도 | 고정 (부정확) | 자동 추정 (정확) |

## 문제 해결

### 여전히 정확도가 낮다면?

1. **이미지 품질 확인**
   - 흐릿하거나 심하게 흔들린 이미지는 검출 어려움
   - 가능하면 재촬영 권장

2. **조명이 너무 어두운 경우**
   - `label_segmentation.py` 452줄의 HSV 범위 조정
   - V 값 하한을 150 → 120으로 낮춤

3. **라인 폭이 너무 얇거나 두꺼운 경우**
   - 461줄의 폭 제한 범위 수정
   - `max(10, min(30, avg_width))` → 원하는 범위로

4. **화면 상단에 라인이 있는 경우**
   - 357줄의 ROI 시작 위치 조정
   - `int(height * 0.25)` → 0.15 등으로 변경

## 성능

- 처리 속도: 약 0.5~1초/이미지 (PC 기준)
- 메모리 사용: 이미지당 약 50MB
- 병렬 처리 가능 (멀티코어 활용 시 더 빠름)

## 다음 단계

1. 자동 라벨링 실행
2. 결과 확인 및 필요시 수동 수정
3. 모델 학습: `python3 train_segmentation.py`
4. 라즈베리파이에 배포
