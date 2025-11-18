#!/usr/bin/env python3
# train_source.py  (TF1 Freeze -> TFLite 안정 변환 버전)
import os, argparse, time, cv2, numpy as np, pandas as pd
import tensorflow.compat.v1 as tf

# --- TF1 모드 설정 ---
tf.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()  # TF2 리소스 변수 경로 비활성화(호환 ↑)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="annotations.csv: columns [filepath,label_id,label]")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--img_size", type=int, default=240)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tflite", default="./tflite_0915/model.tflite", help="output TFLite file path")
ap.add_argument("--save_frozen_pb", action="store_true", help="frozen_graph.pb를 남김")
args = ap.parse_args()

IMG = args.img_size

# ---------- 데이터 ----------
df = pd.read_csv(args.csv)
if not {"filepath","label_id"}.issubset(df.columns):
    raise SystemExit("CSV must have columns: filepath,label_id[,label]")

filepaths = df["filepath"].tolist()
labels_id = df["label_id"].astype(int).tolist()
num_classes = int(df["label_id"].nunique())

def one_hot(ids, C):
    y = np.zeros((len(ids), C), dtype=np.float32)
    y[np.arange(len(ids)), ids] = 1.0
    return y

def load_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)       # BGR
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img.shape[1], img.shape[0]) != (IMG, IMG):
        img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def batches(X_paths, y_ids, batch):
    idx = np.arange(len(X_paths))
    np.random.shuffle(idx)
    for i in range(0, len(idx), batch):
        j = idx[i:i+batch]
        X = np.stack([load_image_rgb(X_paths[k]) for k in j], axis=0)     # (B,H,W,3)
        Y = one_hot([y_ids[k] for k in j], num_classes)                   # (B,C)
        yield X, Y

def hms(sec):
    m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{s:06f}"

# ---------- TF1 그래프 ----------
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, IMG, IMG, 3], name="X")
Y = tf.placeholder(tf.float32, [None, num_classes], name="Y")

# 변수에 명시적 name 부여(호환성 ↑)
W1 = tf.Variable(tf.random.normal([3,3,3,32], stddev=0.01), name="W1")
L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME'), name="L1_relu")
L1 = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="L1_pool")   # /2

W2 = tf.Variable(tf.random.normal([3,3,32,64], stddev=0.01), name="W2")
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME'), name="L2_relu")
L2 = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="L2_pool")   # /4

# Flatten
flat_dim = int(np.prod(L2.get_shape().as_list()[1:]))
L2f = tf.reshape(L2, [-1, flat_dim], name="flatten")

W4 = tf.Variable(tf.random.normal([flat_dim, 512], stddev=0.01), name="W4")
L4 = tf.nn.relu(tf.matmul(L2f, W4), name="L4_relu")

W5 = tf.Variable(tf.random.normal([512, num_classes], stddev=0.01), name="W5")
logits = tf.matmul(L4, W5, name="logits")
probs  = tf.nn.softmax(logits, name="probs")  # 출력 노드

# Loss/Opt/Metric
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
opt  = tf.train.AdamOptimizer(args.lr).minimize(loss)
pred = tf.argmax(logits, axis=1)
true = tf.argmax(Y, axis=1)
acc  = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

# ---------- 학습 + TFLite 변환 ----------
tflite_abs = os.path.abspath(args.tflite)
os.makedirs(os.path.dirname(tflite_abs), exist_ok=True)
print(f"[INFO] TFLite output    : {tflite_abs}")

cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=cfg) as sess:
    sess.run(tf.global_variables_initializer())

    t0 = time.time()
    print(f"[TIME] training start @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, args.epochs + 1):
        e0 = time.time()
        loss_sum, acc_sum, cnt = 0.0, 0.0, 0
        for Xb, Yb in batches(filepaths, labels_id, args.batch):
            _, lv, av = sess.run([opt, loss, acc], feed_dict={X:Xb, Y:Yb})
            loss_sum += lv; acc_sum += av; cnt += 1
        print(f"[epoch {epoch:03d}] loss={loss_sum/max(1,cnt):.6f} acc={acc_sum/max(1,cnt):.6f}  ({hms(time.time()-e0)})")

    # ---- 프리즈(Frozen Graph) -> TFLite (레거시 변환기, 안정 경로) ----
    print("[INFO] Freezing graph (convert variables to constants)...")
    from tensorflow.compat.v1.graph_util import convert_variables_to_constants
    frozen = convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=["probs"],   # 반드시 출력 노드명과 일치
    )

    # frozen_graph.pb 경로
    if args.save_frozen_pb:
        frozen_pb = os.path.join(os.path.dirname(tflite_abs), "frozen_graph.pb")
    else:
        frozen_pb = os.path.join(".", "frozen_graph.pb")

    with tf.io.gfile.GFile(frozen_pb, "wb") as f:
        f.write(frozen.SerializeToString())
    if args.save_frozen_pb:
        print(f"[INFO] Frozen graph saved: {frozen_pb}")
    else:
        # 남기지 않으려면 완료 후 삭제할 수도 있음(지금은 남겨둠)
        pass

    print("[INFO] Converting frozen graph to TFLite (legacy TOCO path)...")
    t0_tfl = time.time()
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        frozen_pb,
        input_arrays=["X"],
        output_arrays=["probs"],
        input_shapes={"X": [1, IMG, IMG, 3]},  # 배치 1 가정(TFLite 추론용)
    )
    converter.experimental_new_converter = False
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]


    tflite_model = converter.convert()
    with open(tflite_abs, "wb") as f:
        f.write(tflite_model)
    print(f"[TFLITE] saved to {tflite_abs}  (convert {hms(time.time()-t0_tfl)})")

    print(f"[TIME] training end   @ {time.strftime('%Y-%m-%d %H:%M:%S')}  (total {hms(time.time()-t0)})")
