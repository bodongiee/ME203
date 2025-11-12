# label_mover.py
import cv2
import os
import re
import csv
import time
import shutil
from pathlib import Path


BASE_DIR = Path("data")  
TARGETS = ["left", "right", "forward", "unknown"]  
WINDOW_NAME = "Label & Move  [L]=left  [R]=right  [F]=forward [U]=unknown [B]=undo  [S]=skip  [Q]=quit"
LOG_FILE = BASE_DIR / "label_log.csv"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def ensure_dirs():
    BASE_DIR.mkdir(exist_ok=True)
    for t in TARGETS:
        (BASE_DIR / t).mkdir(exist_ok=True)

def load_images():
    """
    data 폴더 바로 아래에 있는 이미지들만 대상으로 함.
    (이미 left/right/forward로 이동된 파일은 자동 제외)
    frame_1004.png ~ frame_3037.png와 같은 이름을 숫자 기준으로 정렬.
    """
    files = []
    for p in BASE_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    def key_fn(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    files.sort(key=key_fn)
    return files

def write_log(row):
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts", "action", "file", "src", "dst"])
        w.writerow(row)

def unique_destination(dst_path: Path) -> Path:
    if not dst_path.exists():
        return dst_path
    stem, ext = dst_path.stem, dst_path.suffix
    i = 1
    while True:
        cand = dst_path.with_name(f"{stem}_{i}{ext}")
        if not cand.exists():
            return cand
        i += 1

def move_file(src: Path, dst_folder: Path) -> Path:
    dst_path = unique_destination(dst_folder / src.name)
    shutil.move(str(src), str(dst_path))
    return dst_path

def show_image(img_path: Path, max_width=1400, max_height=900):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def main():
    ensure_dirs()
    files = load_images()
    if not files:
        print("분류할 이미지가 없습니다. data 폴더를 확인하세요.")
        return

    print("\n[조작 안내]")
    print("  L = left  |  R = right  |  F = forward")
    print("  B = undo  |  S = skip   |  Q = quit")
    print("라벨링 진행 중...\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)


    undo_stack = []

    idx = 0
    total = len(files)

    while idx < total:
        img_path = files[idx]
        if not img_path.exists():
            idx += 1
            continue

        img_disp = show_image(img_path)
        if img_disp is None:
            print(f"이미지 로드 실패: {img_path.name}  => 자동 건너뜀")
            write_log([int(time.time()), "skip_load_fail", img_path.name, str(img_path.parent), ""])
            idx += 1
            continue

        # 화면 출력
        overlay = img_disp.copy()
        cv2.putText(
            overlay,
            f"[{idx+1}/{total}] {img_path.name}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(0) & 0xFF

        key_char = chr(key).lower() if key != 255 else ""
        if key_char == "q":
            print("중단합니다.")
            break
        elif key_char == "s":
            # 스킵
            write_log([int(time.time()), "skip", img_path.name, str(img_path.parent), ""])
            idx += 1
            continue
        elif key_char == "b":
            if not undo_stack:
                print("되돌릴 이동 기록이 없습니다.")
                continue
            moved_from, moved_to = undo_stack.pop()
            if moved_to.exists():
                back_to = unique_destination(moved_from / moved_to.name)
                shutil.move(str(moved_to), str(back_to))
                write_log([int(time.time()), "undo", moved_to.name, str(moved_to.parent), str(back_to.parent)])
                print(f"UNDO: {moved_to.name}  =>  {back_to.parent.name}")
            else:
                print("되돌릴 대상 파일이 없습니다(이미 이동/삭제됨).")
            continue
        elif key_char in ("l", "r", "f", "u"):
            if key_char == "l":
                target = BASE_DIR / "left"
            elif key_char == "r":
                target = BASE_DIR / "right"
            elif key_char == "u":
                target = BASE_DIR / "unknown"
            else:
                target = BASE_DIR / "forward"

            before_parent = img_path.parent
            new_path = move_file(img_path, target)
            undo_stack.append((before_parent, new_path))
            write_log([int(time.time()), "move", new_path.name, str(before_parent), str(target)])
            print(f"→ {new_path.name}  이동: {target.name}")
            idx += 1
            continue
        else:
            print("유효한 키가 아닙니다. [L/R/F]로 이동, [S]는 건너뛰기, [B]는 되돌리기, [Q]는 종료.")
            continue

    cv2.destroyAllWindows()
    print("\n작업 종료 ✅")

if __name__ == "__main__":
    main()
