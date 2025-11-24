import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split

try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    HAS_TFA = False
    print("Warning: tensorflow-addons not found. Rotation augmentation disabled.")

# ---------------- 설정 ----------------
IMG_SIZE = 240
BATCH_SIZE = 8  # 작은 데이터셋에는 작은 배치가 더 좋음
EPOCHS = 50
LEARNING_RATE = 0.0001  # 학습률 낮춤 (과적합 방지)

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

        # 마스크 로드
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

# ---------------- Data Augmentation (강화) ----------------
def augment(image, mask):
    """데이터 증강 (과적합 방지를 위해 강화)"""
    # 랜덤 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # 랜덤 밝기 조정 (범위 확대)
    image = tf.image.random_brightness(image, 0.3)

    # 랜덤 대비 조정 (범위 확대)
    image = tf.image.random_contrast(image, 0.7, 1.4)

    # 랜덤 색조 조정 (조명 변화 대응)
    image = tf.image.random_hue(image, 0.1)

    # 랜덤 채도 조정
    image = tf.image.random_saturation(image, 0.7, 1.3)

    # 랜덤 회전 (±10도) - tensorflow-addons가 있을 때만
    if HAS_TFA and tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform((), -10, 10) * (3.14159 / 180.0)
        image = tfa.image.rotate(image, angle, interpolation='bilinear')
        mask = tfa.image.rotate(mask, angle, interpolation='nearest')

    # 랜덤 Gaussian 노이즈 추가 (실제 카메라 노이즈 시뮬레이션)
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

    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ---------------- 모델 정의 (MobileNetV2 기반 U-Net with Regularization) ----------------
def build_unet_mobilenet(input_shape=(IMG_SIZE, IMG_SIZE, 3), dropout_rate=0.3):
    """
    MobileNetV2를 인코더로 사용하는 경량 U-Net
    과적합 방지를 위해 Dropout 추가
    """
    inputs = layers.Input(shape=input_shape)

    # 인코더: MobileNetV2 (pre-trained on ImageNet)
    # 작은 데이터셋에서는 freeze하는 게 더 나을 수 있음
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet'
    )

    # MobileNetV2의 일부 레이어만 학습 (Fine-tuning)
    # 초반 레이어는 freeze (일반적인 특징 추출기)
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # 처음 100개 레이어 동결
        layer.trainable = False

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

    # Upsampling + Skip connections with dynamic resizing
    for i, skip in reversed(list(enumerate(skip_outputs))):
        # Upsampling
        x = layers.Conv2DTranspose(
            filters=128 // (2 ** i),
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu'
        )(x)

        # Skip connection: resize x to match skip size
        # MobileNetV2 skip 크기가 예측 불가능하므로 x를 skip에 맞춤
        x_resized = layers.Resizing(
            height=skip.shape[1],
            width=skip.shape[2],
            interpolation='bilinear'
        )(x)

        x = layers.Concatenate()([x_resized, skip])

        # Convolution with Dropout (과적합 방지)
        x = layers.Conv2D(128 // (2 ** i), 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.Dropout(dropout_rate)(x)  # Dropout 추가
        x = layers.Conv2D(128 // (2 ** i), 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = layers.Dropout(dropout_rate)(x)  # Dropout 추가

    # 최종 upsampling (240x240로 복원)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)

    # 출력 레이어
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='unet_mobilenet')

    return model

# ---------------- Dice Loss ----------------
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

    # TFLite 변환 (두 버전 모두 생성)
    print("\nConverting to TFLite...")

    # 1. INT8 양자화 (빠름, 권장)
    convert_to_tflite(model, use_int8=True)

    # 2. Float16 양자화 (정확도 높음)
    convert_to_tflite(model, use_int8=False)

    print(f"\nModels saved:")
    print(f"  - Keras model: {MODEL_SAVE_PATH}")
    print(f"  - TFLite INT8: {TFLITE_SAVE_PATH.replace('.tflite', '_int8.tflite')} (권장)")
    print(f"  - TFLite Float16: {TFLITE_SAVE_PATH.replace('.tflite', '_float16.tflite')}")

def convert_to_tflite(model, use_int8=True):
    """
    Keras 모델을 TFLite로 변환 (최적화 포함)

    Args:
        model: Keras 모델
        use_int8: True면 INT8 양자화 (빠름, 약간 정확도 하락)
                  False면 Float16 (느림, 정확도 유지)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if use_int8:
        print("[INFO] Using INT8 quantization (faster inference)")

        # INT8 양자화 설정
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset 생성 (양자화 캘리브레이션용)
        def representative_data_gen():
            # 학습 데이터에서 샘플링
            images_dir = IMAGES_DIR
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

            # 최대 100개 샘플 사용
            num_samples = min(100, len(image_files))

            for i in range(num_samples):
                img_path = os.path.join(images_dir, image_files[i])
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                yield [img]

        converter.representative_dataset = representative_data_gen

        # INT8 입력/출력 강제
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # 입력을 uint8로
        converter.inference_output_type = tf.uint8  # 출력을 uint8로

    else:
        print("[INFO] Using Float16 quantization (higher accuracy)")
        # Float16 quantization (속도 향상 + 모델 크기 감소)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # 저장
    save_path = TFLITE_SAVE_PATH.replace('.tflite', '_int8.tflite' if use_int8 else '_float16.tflite')
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved: {save_path}")
    print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

# ---------------- 실행 ----------------
if __name__ == '__main__':
    train_model()
