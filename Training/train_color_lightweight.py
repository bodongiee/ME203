#!/usr/bin/env python3
# train_color_lightweight.py
# 초경량 Green/Red 색상 분류 모델 (라즈베리파이 실시간 추론용)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split

# ---------------- 설정 ----------------
IMG_SIZE = 160  # 64 → 160으로 증가 (더 많은 색상 정보)
BATCH_SIZE = 16  # 32 → 16 (이미지가 커져서)
EPOCHS = 50  # 30 → 50 (더 충분히 학습)
LEARNING_RATE = 0.0001  # 간단한 모델이므로 학습률 높임

# 데이터 경로
DATA_DIR = "../../lab8_materials/data_color"  # green/, red/ 폴더가 있는 디렉토리
CSV_PATH = "./data/color_annotations.csv"  # filepath, label_id, label (CSV가 있으면 사용)

# 모델 저장 경로
MODEL_SAVE_PATH = "./color_light_160.h5"
TFLITE_SAVE_PATH = "./color_light_160.tflite"

# ---------------- 데이터 로드 ----------------
def load_data_from_folders(data_dir, img_size=IMG_SIZE):
    """폴더 구조에서 데이터 로드 (green/, red/)"""

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    green_dir = os.path.join(data_dir, "green")
    red_dir = os.path.join(data_dir, "red")

    if not os.path.exists(green_dir) or not os.path.exists(red_dir):
        raise FileNotFoundError(f"green/ or red/ folder not found in {data_dir}")

    images = []
    labels = []

    # Green 이미지 로드 (label_id = 0)
    print("Loading green images...")
    green_files = [f for f in os.listdir(green_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for i, filename in enumerate(green_files):
        if i % 50 == 0:
            print(f"  Green: {i}/{len(green_files)}", end='\r')

        img_path = os.path.join(green_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(0)  # green = 0

    print(f"\n  Loaded {len(green_files)} green images")

    # Red 이미지 로드 (label_id = 1)
    print("Loading red images...")
    red_files = [f for f in os.listdir(red_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for i, filename in enumerate(red_files):
        if i % 50 == 0:
            print(f"  Red: {i}/{len(red_files)}", end='\r')

        img_path = os.path.join(red_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(1)  # red = 1

    print(f"\n  Loaded {len(red_files)} red images")

    if len(images) == 0:
        raise ValueError("No images loaded!")

    # NumPy 배열로 변환
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # One-hot encoding
    num_classes = 2
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)

    print(f"\nTotal images: {len(images)}")
    print(f"  Green: {np.sum(labels == 0)}")
    print(f"  Red: {np.sum(labels == 1)}")

    return images, labels_onehot, num_classes

def load_data_from_csv(csv_path, img_size=IMG_SIZE):
    """CSV에서 이미지 경로와 라벨 로드"""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if not {"filepath", "label_id"}.issubset(df.columns):
        raise ValueError("CSV must have columns: filepath, label_id")

    filepaths = df["filepath"].tolist()
    label_ids = df["label_id"].astype(int).tolist()

    num_classes = df["label_id"].nunique()

    print(f"Found {len(filepaths)} images")
    print(f"Number of classes: {num_classes}")

    if "label" in df.columns:
        label_names = df.groupby("label_id")["label"].first().to_dict()
        print(f"Classes: {label_names}")

    # 이미지 로드
    images = []
    labels = []

    for i, (path, label_id) in enumerate(zip(filepaths, label_ids)):
        if i % 100 == 0:
            print(f"Loading images: {i}/{len(filepaths)}", end='\r')

        # 이미지 읽기
        img = cv2.imread(path)
        if img is None:
            print(f"\nWarning: Failed to load {path}")
            continue

        # RGB 변환 및 리사이즈
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(label_id)

    print(f"\nLoaded {len(images)} images")

    # NumPy 배열로 변환
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # One-hot encoding
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)

    return images, labels_onehot, num_classes

# ---------------- Data Augmentation ----------------
def augment_color(image, label):
    """색상 분류를 위한 데이터 증강"""

    # 랜덤 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)

    # 랜덤 상하 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)

    # 랜덤 밝기 조정 (색상 분류에 중요)
    image = tf.image.random_brightness(image, 0.3)

    # 랜덤 대비 조정
    image = tf.image.random_contrast(image, 0.7, 1.4)

    # 랜덤 색조 조정 (색상 일반화)
    image = tf.image.random_hue(image, 0.1)

    # 랜덤 채도 조정
    image = tf.image.random_saturation(image, 0.7, 1.3)

    # 값 범위 클리핑
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def create_dataset(images, labels, batch_size, augment_data=True):
    """TensorFlow Dataset 생성"""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if augment_data:
        dataset = dataset.map(augment_color, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ---------------- 초경량 CNN 모델 ----------------
def build_tiny_color_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=2, dropout_rate=0.2):
    """
    초간단 색상 분류 CNN
    - Green/Red는 매우 단순한 색상 차이 → 얕은 네트워크로 충분
    - 2-block만 사용
    """

    inputs = layers.Input(shape=input_shape)

    # Block 1: 160 → 80
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)

    # Block 2: 80 → 40
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layer
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='SimpleCNN')

    return model

# ---------------- 학습 함수 ----------------
def train_model():
    print("="*60)
    print("초경량 색상 분류 모델 학습")
    print(f"IMG_SIZE: {IMG_SIZE}x{IMG_SIZE}")
    print(f"목표: Green/Red 실시간 분류")
    print("="*60)

    # 데이터 로드
    print("\nLoading data from folders...")
    images, labels, num_classes = load_data_from_folders(DATA_DIR)

    if len(images) == 0:
        print("Error: No data found!")
        return

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels.argmax(axis=1)
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Class distribution (train): {np.bincount(y_train.argmax(axis=1))}")
    print(f"Class distribution (val): {np.bincount(y_val.argmax(axis=1))}")

    # Dataset 생성
    train_dataset = create_dataset(X_train, y_train, BATCH_SIZE, augment_data=True)
    val_dataset = create_dataset(X_val, y_val, BATCH_SIZE, augment_data=False)

    # 모델 생성
    print("\nBuilding lightweight model...")
    model = build_tiny_color_cnn(num_classes=num_classes)
    model.summary()

    # 파라미터 수 출력
    total_params = model.count_params()
    print(f"\n총 파라미터 수: {total_params:,} (목표: < 100K)")

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 학습
    print("\nTraining...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 결과 출력
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")

    # TFLite 변환 (INT8 양자화)
    print("\n" + "="*60)
    print("Converting to TFLite (INT8)...")
    print("="*60)
    convert_to_tflite_int8(model, images)

    print(f"\n✓ Models saved:")
    print(f"  - Keras model: {MODEL_SAVE_PATH}")
    print(f"  - TFLite INT8: {TFLITE_SAVE_PATH}")
    print(f"\n✓ 라즈베리파이에서 테스트하세요!")

def convert_to_tflite_int8(model, sample_images):
    """
    INT8 Full Integer Quantization
    색상 분류는 매우 가벼워야 함
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # INT8 양자화 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset
    def representative_data_gen():
        num_samples = min(100, len(sample_images))
        for i in range(num_samples):
            img = sample_images[i:i+1]
            yield [img.astype(np.float32)]

    converter.representative_dataset = representative_data_gen

    # Full INT8 강제
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # 변환
    print("[INFO] Converting... (may take a minute)")
    tflite_model = converter.convert()

    # 저장
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"✓ TFLite INT8 model saved: {TFLITE_SAVE_PATH}")
    print(f"✓ Model size: {size_kb:.1f} KB (목표: < 500KB)")

    # 입력/출력 정보
    interpreter = tf.lite.Interpreter(model_path=TFLITE_SAVE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\n모델 입출력 정보:")
    print(f"  Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']} {output_details[0]['dtype']}")

# ---------------- 실행 ----------------
if __name__ == '__main__':
    train_model()
