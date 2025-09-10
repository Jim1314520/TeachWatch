# cam_recog.py —— 实时摄像头识别（与 hy.jpg 比对）
# 说明：
#  - 使用 cv2.cvtColor 保证传给 dlib 的数组是连续内存，避免 compute_face_descriptor 的 TypeError
#  - 识别到目标时会把带框画面的快照保存到 output/img.jpg（供你后续发邮件脚本直接读取）
#  - 按键：Q/ESC 退出，S 手动保存一张截图到 output/snapshot_*.jpg

import time
from pathlib import Path

import cv2
import numpy as np
import face_recognition as fr

# ======== 可调参数 ========
KNOWN_IMG = "hy.jpg"   # 你的目标人物参考照（正脸、清晰）
TOLERANCE = 0.45       # 阈值；越小越严格(0.35~0.60可调)
SCALE = 0.5            # 下采样系数；0.5 表示缩小到一半做人脸检测
DRAW_NAME = "Target"   # 命中后显示的名字
AUTO_SAVE_INTERVAL = 5 # 命中后自动保存间隔（秒），避免每帧都写盘
# ==========================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)
AUTO_SAVE_PATH = OUT_DIR / "img.jpg"

# 1) 载入并编码目标人脸
known = fr.load_image_file(str(BASE_DIR / KNOWN_IMG))
known_encs = fr.face_encodings(known)
assert known_encs, f"{KNOWN_IMG} 没检测到人脸，换一张更清晰、正脸的参考照"
known_enc = known_encs[0]

# 2) 打开摄像头（0/1/2 依次尝试）
def open_cam():
    for i in (0, 1, 2):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows 上 DSHOW 更稳定
        if cap.isOpened():
            print("使用摄像头索引:", i)
            return cap
    raise RuntimeError("找不到可用摄像头（试试把索引改成 1 或 2）")

cap = open_cam()
cv2.namedWindow("Face Watch", cv2.WINDOW_NORMAL)

last_hit_ts = 0.0
last_auto_save_ts = 0.0
process_toggle = True  # 隔帧处理以降低 CPU 负载

while True:
    ok, frame = cap.read()
    if not ok:
        print("读取摄像头失败")
        break

    show = frame.copy()  # 用于画框显示
    hits = []
    locs = []

    if process_toggle:
        # 3) 预处理：缩放 + BGR->RGB（必须用 cv2.cvtColor，保证连续内存）
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # ✅ 连续内存

        # 4) 检测与编码
        locs = fr.face_locations(rgb, model="hog")     # CPU 友好的 HOG
        encs = fr.face_encodings(rgb, locs)            # 用检测到的位置提特征

        # 5) 逐脸匹配
        hits = [fr.compare_faces([known_enc], e, tolerance=TOLERANCE)[0] for e in encs]

        # 6) 把小图坐标映射回原图并画框/名字
        scale_inv = int(round(1.0 / SCALE))
        for (top, right, bottom, left), hit in zip(locs, hits):
            top, right, bottom, left = [v * scale_inv for v in (top, right, bottom, left)]
            color = (0, 255, 0) if hit else (0, 0, 255)
            cv2.rectangle(show, (left, top), (right, bottom), color, 2)
            cv2.putText(show, DRAW_NAME if hit else "Unknown",
                        (left, max(0, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if hit:
                last_hit_ts = time.time()

        # 7) 命中后自动保存一张到 output/img.jpg（供发邮件脚本使用）
        if any(hits) and (time.time() - last_auto_save_ts) > AUTO_SAVE_INTERVAL:
            cv2.imwrite(str(AUTO_SAVE_PATH), show)
            print("已自动保存：", AUTO_SAVE_PATH)
            last_auto_save_ts = time.time()

    process_toggle = not process_toggle

    # 在左上角显示状态
    status = f"Hit: {time.strftime('%H:%M:%S', time.localtime(last_hit_ts)) if last_hit_ts else '-'}  Tol={TOLERANCE}"
    cv2.putText(show, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Face Watch", show)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):   # Q 或 ESC 退出
        break
    if key == ord('s'):  # 手动保存截图
        snap = OUT_DIR / f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(str(snap), show)
        print("已保存：", snap)

cap.release()
cv2.destroyAllWindows()
