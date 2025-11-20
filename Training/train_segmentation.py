#!/usr/bin/env python3
# train_segmentation.py
# Semantic Segmentation 모델 학습 스크립트

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split

# ---------------- 설정 ----------------
IMG_SIZE = 240
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

# 데이터 경로
IMAGES_DIR = "./data/images"  # 원본 이미지
MASKS_DIR = "./data/masks"    # 라벨링된 마스크 (라인 부분은 흰색, 나머지는 검정)

MODEL_SAVE_PATH = "./line_segmentation.h5"
TFLITE_SAVE_PATH = "./line_segmentation.tflite"

# ---------------- 데이터 로딩 ----------------
def load_dataset(images_dir, masks_dir):
    """이미지와 마스크 로딩"""
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    images = []
    masks = []

    print(f"Loading {len(image_files)} images...")

    for img_file in image_files:
        # 이미지 로드
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # 마스크 로드 (같은 파일명으로 저장되어 있어야 함)
        mask_path = os.path.join(masks_dir, img_file)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_file}, skipping...")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # 마스크를 0/1로 정규화
        mask = (mask > 127).astype(np.float32)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32)[..., np.newaxis]  # (N, 240, 240, 1)

    print(f"Loaded {len(images)} image-mask pairs")
    return images, masks

# ---------------- Data Augmentation ----------------
def augment(image, mask):
    """데이터 증강"""
    # 랜덤 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # 랜덤 밝기 조정
    image = tf.image.random_brightness(image, 0.2)

    # 랜덤 대비 조정
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # 값 범위 클리핑
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask

def create_dataset(images, masks, batch_size, augment_data=True):
    """TensorFlow Dataset 생성"""
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ---------------- 모델 정의 (MobileNetV2 기반 U-Net) ----------------
def build_unet_mobilenet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    MobileNetV2를 인코더로 사용하는 경량 U-Net
    라즈베리파이에서 실시간 추론 가능
    """
    inputs = layers.Input(shape=input_shape)

    # 인코더: MobileNetV2 (pre-trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet'
    )

    # Skip connections를 위한 레이어 선택
    skip_names = [
        'block_1_expand_relu',   # 120x120
        'block_3_expand_relu',   # 60x60
        'block_6_expand_relu',   # 30x30
        'block_13_expand_relu',  # 15x15
    ]

    skip_outputs = [base_model.get_layer(name).output for name in skip_names]

    # 인코더 출력
    encoder_output = base_model.output  # 8x8

    # 디코더
    x = encoder_output

    # Upsampling + Skip connections
    for i, skip in reversed(list(enumerate(skip_outputs))):
        # Upsampling
        x = layers.Conv2DTranspose(
            filters=128 // (2 ** i),
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu'
        )(x)

        # Skip connection
        x = layers.Concatenate()([x, skip])

        # Convolution
        x = layers.Conv2D(128 // (2 ** i), 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(128 // (2 ** i), 3, padding='same', activation='relu')(x)

    # 최종 upsampling (240x240로 복원)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)

    # 출력 레이어
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='unet_mobilenet')

    return model

# ---------------- Dice Loss (Segmentation에 효과적) ----------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient (F1 score의 일종)"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred):
    """Dice loss"""
    return 1.0 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Dice Loss + Binary Crossentropy"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# ---------------- 학습 ----------------
def train_model():
    """모델 학습 메인 함수"""
    print("=" * 50)
    print("Semantic Segmentation 모델 학습 시작")
    print("=" * 50)

    # 데이터 로드
    images, masks = load_dataset(IMAGES_DIR, MASKS_DIR)

    if len(images) == 0:
        print("Error: No data found!")
        print(f"Please prepare images in {IMAGES_DIR} and masks in {MASKS_DIR}")
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
    print("\nBuilding model...")
    model = build_unet_mobilenet()
    model.summary()

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
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
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

    # 최종 성능 출력
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    print(f"Best validation Dice coefficient: {max(history.history['val_dice_coefficient']):.4f}")
    print(f"Best validation accuracy: {max(history.history['val_binary_accuracy']):.4f}")

    # TFLite 변환
    print("\nConverting to TFLite...")
    convert_to_tflite(model)

    print(f"\nModels saved:")
    print(f"  - Keras model: {MODEL_SAVE_PATH}")
    print(f"  - TFLite model: {TFLITE_SAVE_PATH}")

def convert_to_tflite(model):
    """Keras 모델을 TFLite로 변환 (최적화 포함)"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 최적화: Float16 quantization (속도 향상 + 모델 크기 감소)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # 저장
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

# ---------------- 실행 ----------------
if __name__ == '__main__':
    train_model()
