#!/usr/bin/env python3
# train_lightweight.py
# 라즈베리파이 실시간 추론을 위한 초경량 Segmentation 모델
# 목표: 10+ FPS (< 100ms inference time)

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split

# tensorflow-addons for rotation (optional)
try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    HAS_TFA = False
    print("Warning: tensorflow-addons not found. Rotation augmentation disabled.")

# ---------------- 설정 ----------------
IMG_SIZE = 160  # 240 → 160 (연산량 약 2.25배 감소)
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0001

# 데이터 경로
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

# 모델 저장 경로
MODEL_SAVE_PATH = "./line_segmentation_light.h5"
TFLITE_SAVE_PATH = "./line_segmentation_light.tflite"

# ---------------- 데이터 로드 ----------------
def load_data(images_dir, masks_dir, img_size=IMG_SIZE):
    """이미지와 마스크 로드"""
    images = []
    masks = []

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    for img_name in image_files:
        # 원본 이미지
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # 마스크
        mask_name = img_name.replace('.jpg', '.png').replace('.png', '.png')
        mask_path = os.path.join(masks_dir, mask_name)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (img_size, img_size))
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    print(f"Loaded {len(images)} image-mask pairs")
    return images, masks

# ---------------- Data Augmentation (강화) ----------------
def augment(image, mask):
    """데이터 증강"""
    # 랜덤 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # 랜덤 밝기 조정
    image = tf.image.random_brightness(image, 0.3)

    # 랜덤 대비 조정
    image = tf.image.random_contrast(image, 0.7, 1.4)

    # 랜덤 색조 조정
    image = tf.image.random_hue(image, 0.1)

    # 랜덤 채도 조정
    image = tf.image.random_saturation(image, 0.7, 1.3)

    # 랜덤 회전 (±10도)
    if HAS_TFA and tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform((), -10, 10) * (3.14159 / 180.0)
        image = tfa.image.rotate(image, angle, interpolation='bilinear')
        mask = tfa.image.rotate(mask, angle, interpolation='nearest')

    # 랜덤 Gaussian 노이즈
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
        image = image + noise

    # 값 범위 클리핑
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask

def create_dataset(images, masks, batch_size, augment_data=True):
    """TensorFlow Dataset 생성"""
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ---------------- 초경량 모델 정의 ----------------
def build_tiny_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), dropout_rate=0.2):

    inputs = layers.Input(shape=input_shape)

    # ---------------- Encoder (다운샘플링) ----------------
    # Block 1: 160 → 80
    x = layers.SeparableConv2D(16, 3, padding='same', activation='relu')(inputs)
    skip1 = layers.SeparableConv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(skip1)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2: 80 → 40
    x = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(x)
    skip2 = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(skip2)
    x = layers.Dropout(dropout_rate)(x)

    # Block 3: 40 → 20
    x = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
    skip3 = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(skip3)
    x = layers.Dropout(dropout_rate)(x)

    # ---------------- Bottleneck (20x20) ----------------
    x = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # ---------------- Decoder (업샘플링) ----------------
    # Block 4: 20 → 40
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 5: 40 → 80
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 6: 80 → 160
    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.SeparableConv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # ---------------- 최종 출력 ----------------
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name='TinyUNet')

    return model

# ---------------- Loss & Metrics ----------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient (F1-score for segmentation)"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def combined_loss(y_true, y_pred):
    """Binary Crossentropy + Dice Loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - dice_coefficient(y_true, y_pred)
    return bce + dice

# ---------------- 학습 함수 ----------------
def train_model():
    print("="*50)
    print("초경량 Segmentation 모델 학습")
    print(f"IMG_SIZE: {IMG_SIZE}x{IMG_SIZE}")
    print(f"목표: 라즈베리파이 10+ FPS")
    print("="*50)

    # 데이터 로드
    print("\nLoading data...")
    images, masks = load_data(IMAGES_DIR, MASKS_DIR)

    if len(images) == 0:
        print("Error: No data found!")
        return

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Dataset 생성
    train_dataset = create_dataset(X_train, y_train, BATCH_SIZE, augment_data=True)
    val_dataset = create_dataset(X_val, y_val, BATCH_SIZE, augment_data=False)

    # 모델 생성
    print("\nBuilding lightweight model...")
    model = build_tiny_unet()
    model.summary()

    # 파라미터 수 출력
    total_params = model.count_params()
    print(f"\n총 파라미터 수: {total_params:,} (목표: < 500K)")

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[
            dice_coefficient,
            'binary_accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_dice_coefficient',
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
            monitor='val_dice_coefficient',
            mode='max',
            patience=15,
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
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best validation Dice coefficient: {max(history.history['val_dice_coefficient']):.4f}")
    print(f"Best validation accuracy: {max(history.history['val_binary_accuracy']):.4f}")

    # TFLite 변환 (INT8 양자화)
    print("\n" + "="*50)
    print("Converting to TFLite (INT8)...")
    print("="*50)
    convert_to_tflite_int8(model, images)

    print(f"\n✓ Models saved:")
    print(f"  - Keras model: {MODEL_SAVE_PATH}")
    print(f"  - TFLite INT8: {TFLITE_SAVE_PATH}")
    print(f"\n✓ 라즈베리파이에서 테스트하세요!")

def convert_to_tflite_int8(model, sample_images):
    """
    INT8 Full Integer Quantization
    - 입력: uint8 (0~255)
    - 출력: uint8 (0~255)
    - 속도: 5~10배 향상
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # INT8 양자화 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset (캘리브레이션용)
    def representative_data_gen():
        num_samples = min(100, len(sample_images))
        for i in range(num_samples):
            # Float32 입력 (0~1)
            img = sample_images[i:i+1]
            yield [img.astype(np.float32)]

    converter.representative_dataset = representative_data_gen

    # Full INT8 강제
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # 입력: 0~255
    converter.inference_output_type = tf.uint8  # 출력: 0~255

    # 변환
    print("[INFO] Converting... (may take a few minutes)")
    tflite_model = converter.convert()

    # 저장
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024 / 1024
    print(f"✓ TFLite INT8 model saved: {TFLITE_SAVE_PATH}")
    print(f"✓ Model size: {size_mb:.2f} MB (목표: < 2MB)")

    # 입력/출력 정보
    interpreter = tf.lite.Interpreter(model_path=TFLITE_SAVE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\n모델 입출력 정보:")
    print(f"  Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
    print(f"\n사용법:")
    print(f"  1. 입력: 0~255 uint8 이미지")
    print(f"  2. 출력: 0~255 uint8 마스크")
    print(f"  3. 이진화: output > 127")

# ---------------- 실행 ----------------
if __name__ == '__main__':
    train_model()
