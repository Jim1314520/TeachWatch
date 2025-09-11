# cam_preview.py —— 实时预览 + 检测框 + 识别结果 + FPS + 使用的检测器提示
# 操作：Q/ESC 退出；S 保存截图；A 切换检测器模式（auto/scrfd/hog）

import time, json
from pathlib import Path
import cv2
import numpy as np
import face_recognition as fr

BASE = Path(__file__).resolve().parent
CFG_PATH = BASE / "config.json"
KNOWN_DIR = BASE / "known"
OUT_DIR  = BASE / "output"
OUT_DIR.mkdir(exist_ok=True)

# -------- 读取配置 --------
cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
tol   = cfg.get("recognition", {}).get("tolerance", 0.45)
scale = cfg.get("recognition", {}).get("scale", 0.5)
model = cfg.get("recognition", {}).get("model", "hog")

# -------- InsightFace / SCRFD 检测器 --------
INS_DET_SIZE = (640, 640)   # 大=更准，小=更快：(512,512) 也可以
MAX_FACES    = 12
_ins_app = None

def init_scrfd():
    global _ins_app
    if _ins_app is not None:
        return _ins_app
    from insightface.app import FaceAnalysis
    _ins_app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection'],
        providers=['CPUExecutionProvider']
    )
    _ins_app.prepare(ctx_id=0, det_size=INS_DET_SIZE)
    print(f"[SCRFD] ready det_size={INS_DET_SIZE}")
    return _ins_app

def detect_faces_scrfd(frame_bgr, max_faces=MAX_FACES):
    app = init_scrfd()
    faces = app.get(frame_bgr)
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)[:max_faces]
    h, w = frame_bgr.shape[:2]
    locs = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox.astype(int))
        l = max(0, x1); t = max(0, y1); r = min(w-1, x2); b = min(h-1, y2)
        if r > l and b > t:
            locs.append((t, r, b, l))
    return locs

# -------- 人脸库 --------
def load_facebank():
    names, encs = [], []
    for p in sorted(KNOWN_DIR.glob("*.*")):
        if p.suffix.lower() not in {".jpg",".jpeg",".png"}:
            continue
        img = fr.load_image_file(str(p))
        fe  = fr.face_encodings(img)
        if not fe:
            print(f"[facebank] {p.name} 无人脸，跳过")
            continue
        names.append(p.stem); encs.append(fe[0])
        print(f"[facebank] 加载：{p.name}")
    return names, encs

names, bank = load_facebank()
if not names:
    print("known/ 里没有有效参考照，退出。")
    raise SystemExit(1)

# -------- 打开摄像头 --------
cap = None
for i in (0,1,2):
    c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if c.isOpened():
        cap = c
        print("使用摄像头索引:", i)
        break
if cap is None:
    print("找不到可用摄像头")
    raise SystemExit(2)

cv2.namedWindow("Face Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Preview", 960, 540)

last = time.time()
fps = 0.0
det_mode = "auto"  # auto/scrfd/hog

while True:
    ok, frame = cap.read()
    if not ok:
        print("读取摄像头失败")
        break

    used = "none"
    locs_big = []

    # ---- 选择检测器 ----
    if det_mode in ("auto", "scrfd"):
        try:
            locs_big = detect_faces_scrfd(frame, max_faces=MAX_FACES)
            if locs_big:
                used = "scrfd"
        except Exception as e:
            print("[SCRFD] error:", e)

    if det_mode in ("auto", "hog") and not locs_big:
        small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs_small = fr.face_locations(rgb_small, model=model)
        s = int(round(1.0 / scale))
        locs_big = [(t*s, r*s, b*s, l*s) for (t, r, b, l) in locs_small]
        if locs_big:
            used = "hog"

    # ---- 编码 + 匹配 ----
    vis = frame.copy()
    if locs_big:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = fr.face_encodings(frame_rgb, locs_big)
        for (t, r, b, l), e in zip(locs_big, encs):
            name = "Unknown"
            if len(bank):
                d = fr.face_distance(bank, e)
                idx = int(np.argmin(d))
                if d[idx] < tol:
                    name = names[idx] + f" ({d[idx]:.2f})"
                else:
                    name = f"Unknown ({d[idx]:.2f})"
            color = (0,255,0) if name.startswith(tuple(names)) else (0,0,255)
            cv2.rectangle(vis, (l, t), (r, b), color, 2)
            cv2.putText(vis, name, (l, max(0, t-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---- HUD：FPS / 检测器 / 阈值 ----
    now = time.time()
    fps = 0.9*fps + 0.1*(1.0 / max(1e-6, now - last))
    last = now
    hud = f"FPS:{fps:.1f}  DET:{used.upper()}  Tol:{tol:.2f}  Mode:{det_mode}"
    cv2.putText(vis, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Face Preview", vis)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q'), ord('Q')):
        break
    if k in (ord('s'), ord('S')):
        p = OUT_DIR / f"preview_{int(time.time())}.jpg"
        cv2.imwrite(str(p), vis)
        print("[保存]", p)
    if k in (ord('a'), ord('A')):
        det_mode = "scrfd" if det_mode == "auto" else ("hog" if det_mode == "scrfd" else "auto")
        print("检测器模式 =>", det_mode)

cap.release()
cv2.destroyAllWindows()
