# watcher_service.py —— 无界面后台监控：多人识别 + 识别即三连发 + 事件冷却 + 可暂停(释放摄像头)
import json, time, ssl, logging, logging.handlers, sys
from email.message import EmailMessage
from pathlib import Path



import cv2
import numpy as np
import face_recognition as fr
from flask import Flask, Response
from threading import Thread

BASE = Path(__file__).resolve().parent
CFG_PATH = BASE / "config.json"
OUT_DIR = BASE / "output"
KNOWN_DIR = BASE / "known"
PAUSE_FLAG = BASE / "PAUSE"      # ✅ 有这个文件就“暂停”：释放摄像头并等待恢复
OUT_DIR.mkdir(exist_ok=True)
KNOWN_DIR.mkdir(exist_ok=True)



# --- 使用 InsightFace / SCRFD 做人脸检测 ---
INS_DET_SIZE = (640, 640)   # 检测输入尺寸：大=更准，小=更快。可调成 (512,512) / (320,320)
MAX_FACES    = 8           # 每帧最多处理的人脸数，防止多人时过载
_ins_app = None


# 预览开关文件（存在就编码推流）、端口
PREVIEW_FLAG = BASE / "PREVIEW"
PREVIEW_PORT = 5000
_latest_jpeg = None

def _start_preview_server():
    app = Flask(__name__)
    @app.get("/")
    def index():
        return "<h3>FaceWatch Preview</h3><img src='/video' style='max-width:100%'>"
    @app.get("/video")
    def video():
        def gen():
            import time as _t
            boundary = b"--frame"
            while True:
                frame = globals().get("_latest_jpeg", None)
                if frame:
                    yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                _t.sleep(0.03)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    Thread(target=lambda: app.run("127.0.0.1", PREVIEW_PORT, threaded=True, use_reloader=False), daemon=True).start()



def _init_insightface():
    """懒加载 SCRFD 检测器（CPU 推理）"""
    global _ins_app
    if _ins_app is not None:
        return _ins_app
    from insightface.app import FaceAnalysis
    _ins_app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection'],
        providers=['CPUExecutionProvider']  # 只用 CPU，更稳定
    )
    _ins_app.prepare(ctx_id=0, det_size=INS_DET_SIZE)
    logging.info("[SCRFD] detector ready, det_size=%s", INS_DET_SIZE)
    return _ins_app

def detect_faces_scrfd(frame_bgr, max_faces=MAX_FACES):
    """
    用 SCRFD 在 BGR 原图上检测，返回 (top, right, bottom, left) 列表，
    以便直接喂给 face_recognition.face_encodings(..., known_face_locations=...)
    """
    app = _init_insightface()
    faces = app.get(frame_bgr)  # 每个元素含 .bbox 和 .det_score
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)[:max_faces]
    h, w = frame_bgr.shape[:2]
    locs = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox.astype(int))
        l = max(0, x1); t = max(0, y1); r = min(w - 1, x2); b = min(h - 1, y2)
        if r > l and b > t:
            locs.append((t, r, b, l))
    return locs



def load_cfg():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg.setdefault("recognition", {}).setdefault("tolerance", 0.45)
    cfg["recognition"].setdefault("scale", 0.5)
    cfg["recognition"].setdefault("model", "hog")
    cfg.setdefault("limits", {}).setdefault("auto_save_interval_sec", 5)
    cfg["limits"].setdefault("event_cooldown_sec", 60)  # 同一人再次三连发前的最短间隔
    cfg.setdefault("recipients_map", {}).setdefault("_default", [])
    cfg.setdefault("recipients_all", [])
    cfg.setdefault("logging", {}).setdefault("level", "INFO")

    cfg["logging"].setdefault("file", "logs/facewatch.log")
    cfg["logging"].setdefault("max_mb", 5)
    cfg["logging"].setdefault("backups", 5)
    return cfg

def setup_logger(log_cfg):
    log_file = BASE / log_cfg["file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=log_cfg["max_mb"]*1024*1024, backupCount=log_cfg["backups"], encoding="utf-8"
    )
    logging.basicConfig(
        level=getattr(logging, log_cfg["level"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[handler, logging.StreamHandler(sys.stdout)],
    )

cfg = load_cfg()
setup_logger(cfg["logging"])

def send_mail(to_list, subject, body, attach_path: Path | None, smtp_cfg: dict):
    import smtplib
    if not to_list:
        logging.info("[邮件] 收件人为空，跳过"); return False
    msg = EmailMessage()
    msg["From"] = smtp_cfg["sender"]; msg["To"] = ", ".join(to_list); msg["Subject"] = subject
    msg.set_content(body)
    if attach_path and attach_path.exists():
        data = attach_path.read_bytes()
        msg.add_attachment(data, maintype="image", subtype="jpeg", filename=attach_path.name)
    s = None
    try:
        if smtp_cfg["port"] == 465:
            s = smtplib.SMTP_SSL(smtp_cfg["host"], smtp_cfg["port"], timeout=30, context=ssl.create_default_context())
        else:
            s = smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"], timeout=30)
            s.ehlo(); s.starttls(context=ssl.create_default_context()); s.ehlo()
        s.login(smtp_cfg["sender"], smtp_cfg["auth"])
        refused = s.send_message(msg)
        if refused: logging.error("[邮件] 部分拒收：%s", refused); return False
        return True
    except smtplib.SMTPAuthenticationError as e:
        logging.error("[邮件] 认证失败：%s", e); return False
    except smtplib.SMTPResponseException as e:
        logging.error("[邮件] 服务器错误：%s %s", e.smtp_code, e.smtp_error); return False
    except Exception as e:
        logging.error("[邮件] 发送异常：%s", e); return False
    finally:
        if s is not None:
            try: s.quit()
            except Exception: pass

def load_facebank():
    names, encs = [], []
    for p in sorted(KNOWN_DIR.glob("*.*")):
        if p.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
        img = fr.load_image_file(str(p))
        fe = fr.face_encodings(img)
        if not fe: logging.warning("[facebank] %s 未检测到人脸，跳过", p.name); continue
        names.append(p.stem); encs.append(fe[0]); logging.info("[facebank] 加载：%s", p.name)
    if not names: logging.error("[facebank] known/ 为空或无有效人脸。");
    return names, encs

def recipients_all():
    # 优先用 recipients_all；若没配置，则回退到 recipients_map._default
    lst = cfg.get("recipients_all") or []
    if lst:
        return lst
    return cfg.get("recipients_map", {}).get("_default", [])




# def run_loop():
#     tol   = cfg["recognition"]["tolerance"]
#     scale = cfg["recognition"]["scale"]
#     model = cfg["recognition"]["model"]
#     auto_save_int  = cfg["limits"]["auto_save_interval_sec"]
#     event_cooldown = cfg["limits"]["event_cooldown_sec"]
#     smtp_cfg       = cfg["smtp"]
#
#     # ---- SCRFD 检测配置（多人更稳）----
#     INS_DET_SIZE = (640, 640)   # 卡的话可改 (512,512)/(320,320)
#     MAX_FACES    = 12
#     ins_app = None              # 懒加载 SCRFD
#
#     # 人脸库
#     names, bank = load_facebank()
#     if not names:
#         time.sleep(5); return
#
#     # 全局事件级时间戳（不再按人名）
#     last_any_event_ts = 0.0   # 上次三连发时间（全局）
#     last_any_save_ts  = 0.0   # 最近一次公共快照时间
#
#     cap = None
#     process_toggle = True
#     idle_sleep = 0.001
#
#     # 初始化 SCRFD（失败则退回 HOG）
#     try:
#         from insightface.app import FaceAnalysis
#         ins_app = FaceAnalysis(
#             name='buffalo_l',
#             allowed_modules=['detection'],
#             providers=['CPUExecutionProvider']
#         )
#         ins_app.prepare(ctx_id=0, det_size=INS_DET_SIZE)
#         logging.info("[SCRFD] detector ready, det_size=%s", INS_DET_SIZE)
#     except Exception as e:
#         ins_app = None
#         logging.exception("[SCRFD] init failed, fallback to HOG: %s", e)
#
#     try:
#         while True:
#             # —— 暂停：有 PAUSE 文件就释放摄像头并等待 ——
#             if PAUSE_FLAG.exists():
#                 if cap is not None:
#                     try: cap.release()
#                     except Exception: pass
#                     cap = None
#                     logging.info("[暂停] 已释放摄像头，直到删除 PAUSE 文件再继续")
#                 time.sleep(1.0)
#                 continue
#
#             # 打开摄像头（自恢复）
#             if cap is None:
#                 for idx in (0,1,2):
#                     c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
#                     if c.isOpened():
#                         cap = c; logging.info("摄像头索引: %s", idx); break
#                 if cap is None:
#                     logging.error("找不到可用摄像头，3秒后重试…"); time.sleep(3); continue
#
#             ok, frame = cap.read()
#             if not ok:
#                 logging.error("读取摄像头失败，重开中…")
#                 try: cap.release()
#                 except Exception: pass
#                 cap = None; time.sleep(1.0); continue
#
#             # 降负载（隔帧处理）
#             if not process_toggle:
#                 process_toggle = True; time.sleep(idle_sleep); continue
#             process_toggle = False
#
#             # ========= 检测 + 编码 =========
#             locs_big = []
#
#             if ins_app is not None:
#                 try:
#                     faces = ins_app.get(frame)  # f.bbox, f.det_score
#                     if faces:
#                         faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)[:MAX_FACES]
#                         h, w = frame.shape[:2]
#                         for f in faces:
#                             x1, y1, x2, y2 = map(int, f.bbox.astype(int))
#                             l = max(0, x1); t = max(0, y1); r = min(w-1, x2); b = min(h-1, y2)
#                             if r > l and b > t:
#                                 locs_big.append((t, r, b, l))
#                 except Exception as e:
#                     logging.warning("[SCRFD] detect error, fallback this frame: %s", e)
#
#             if not locs_big:
#                 # 兜底：HOG（小图检测，映射回原图）
#                 small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
#                 rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
#                 locs_small = fr.face_locations(rgb_small, model=model)
#                 s = int(round(1.0 / scale))
#                 locs_big = [(t*s, r*s, b*s, l*s) for (t, r, b, l) in locs_small]
#
#             if not locs_big:
#                 time.sleep(idle_sleep); continue
#
#             # 编码（用原图RGB + 大坐标）
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             encs = fr.face_encodings(frame_rgb, locs_big)
#             if not encs:
#                 time.sleep(idle_sleep); continue
#             # ===============================
#
#             # —— 识别 + 为预览准备标签/颜色 ——
#             rec_names = []
#             labels, colors = [], []
#             for e in encs:
#                 d = fr.face_distance(bank, e) if len(bank) else np.array([1.0])
#                 idx = int(np.argmin(d))
#                 if len(bank) and d[idx] < tol:
#                     rec_names.append(names[idx])
#                     labels.append(f"{names[idx]} {d[idx]:.2f}")
#                     colors.append((0, 255, 0))   # 命中
#                 else:
#                     lbl = f"Unknown {float(d[idx]):.2f}" if len(d) else "Unknown"
#                     labels.append(lbl)
#                     colors.append((0, 0, 255))   # 未知
#
#             now = time.time()
#
#             # —— 浏览器预览：当 PREVIEW 文件存在时推送一帧 JPEG ——
#             if PREVIEW_FLAG.exists():
#                 try:
#                     show = frame.copy()
#                     if locs_big:
#                         for (t, r, b, l), label, color in zip(locs_big, labels, colors):
#                             cv2.rectangle(show, (l, t), (r, b), color, 2)
#                             cv2.putText(show, label, (l, max(0, t - 8)),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#                     ok_jpg, buf = cv2.imencode(".jpg", show, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
#                     if ok_jpg:
#                         globals()["_latest_jpeg"] = buf.tobytes()
#                 except Exception:
#                     pass
#             else:
#                 globals()["_latest_jpeg"] = None
#
#             # —— 任意命中即“全局群发三封”（按事件冷却） ——
#             if not rec_names:
#                 time.sleep(idle_sleep); continue
#
#             # 1) 保存一份公共快照（限速）
#             any_path = OUT_DIR / "known_last.jpg"
#             if (now - last_any_save_ts) > auto_save_int or (not any_path.exists()):
#                 try:
#                     # 复用上面的 show；如果没创建就用 frame
#                     if 'show' not in locals():
#                         show = frame
#                     cv2.imwrite(str(any_path), show)
#                     logging.info("[保存] %s", any_path)
#                 except Exception:
#                     pass
#                 last_any_save_ts = now
#
#             # 2) 冷却控制（全局，不分人名）
#             if (now - last_any_event_ts) < event_cooldown:
#                 time.sleep(idle_sleep); continue
#
#             # 3) 群发三封
#             to_list = (cfg.get("recipients_all") or
#                        cfg.get("recipients_map", {}).get("_default", []))
#             names_str = ", ".join(sorted(set(rec_names)))
#             subject = f"已识别到图库人员：{names_str}"[:120]
#             body    = f"系统在 {time.strftime('%Y-%m-%d %H:%M:%S')} 识别到图库人员：{names_str}"
#
#             attach = any_path if any_path.exists() else (OUT_DIR / "img.jpg")
#             if not attach.exists():
#                 cv2.imwrite(str(attach), frame)
#
#             for i in range(3):
#                 ok_i = send_mail(to_list, subject, body, attach, smtp_cfg)
#                 logging.info("[三连发][ANY] 第 %d/3：%s", i+1, "成功" if ok_i else "失败")
#                 time.sleep(0.6)
#
#             last_any_event_ts = time.time()
#             time.sleep(idle_sleep)
#
#     finally:
#         try:
#             if cap is not None: cap.release()
#         except Exception:
#             pass


def run_loop():
    tol   = cfg["recognition"]["tolerance"]
    scale = cfg["recognition"]["scale"]
    model = cfg["recognition"]["model"]
    auto_save_int  = cfg["limits"]["auto_save_interval_sec"]
    event_cooldown = cfg["limits"]["event_cooldown_sec"]
    smtp_cfg       = cfg["smtp"]

    # ---- SCRFD 检测配置（多人更稳）----
    INS_DET_SIZE = (640, 640)   # 卡的话可改 (512,512)/(320,320)
    MAX_FACES    = 12
    ins_app = None              # 懒加载 SCRFD

    # 人脸库
    names, bank = load_facebank()
    if not names:
        time.sleep(5); return

    # 全局事件级时间戳（不再按人名）
    last_any_event_ts = 0.0   # 上次三连发时间（全局）
    last_any_save_ts  = 0.0   # 最近一次公共快照时间

    # 初始化 SCRFD（失败则退回 HOG）
    try:
        from insightface.app import FaceAnalysis
        ins_app = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=['detection'],
            providers=['CPUExecutionProvider']
        )
        ins_app.prepare(ctx_id=0, det_size=INS_DET_SIZE)
        logging.info("[SCRFD] detector ready, det_size=%s", INS_DET_SIZE)
    except Exception as e:
        ins_app = None
        logging.exception("[SCRFD] init failed, fallback to HOG: %s", e)

    # ========= 流水线：取流线程 + 检测线程 =========
    from threading import Thread, Event
    import queue

    frame_q  = queue.Queue(maxsize=1)   # 始终只保留最新一帧
    result_q = queue.Queue(maxsize=2)   # 结果队列（满了丢旧）
    stop_evt = Event()

    cap = None
    t_grab = t_det = None
    running = False  # 是否已启动采集/检测线程

    def start_pipeline():
        nonlocal cap, t_grab, t_det, running
        # 打开摄像头
        for idx in (0, 1, 2):
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if c.isOpened():
                cap = c
                logging.info("摄像头索引: %s", idx)
                break
        if cap is None:
            logging.error("找不到可用摄像头，3秒后重试…")
            return False

        stop_evt.clear()

        # 取流线程：只负责 cap.read()，覆盖最新帧
        def grab_loop():
            while not stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01); continue
                if frame_q.full():
                    try: frame_q.get_nowait()
                    except queue.Empty: pass
                try: frame_q.put_nowait(frame)
                except queue.Full: pass

        # 检测/识别线程：重计算都在这里
        def detect_loop():
            nonlocal ins_app
            while not stop_evt.is_set():
                try:
                    frame = frame_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                # —— SCRFD 检测；失败时本帧用 HOG 兜底 ——
                locs_big = []
                if ins_app is not None:
                    try:
                        faces = ins_app.get(frame)  # f.bbox, f.det_score
                        if faces:
                            faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)[:MAX_FACES]
                            h, w = frame.shape[:2]
                            for f in faces:
                                x1, y1, x2, y2 = map(int, f.bbox.astype(int))
                                l = max(0, x1); t = max(0, y1); r = min(w-1, x2); b = min(h-1, y2)
                                if r > l and b > t:
                                    locs_big.append((t, r, b, l))
                    except Exception as e:
                        logging.warning("[SCRFD] detect error: %s", e)

                if not locs_big:
                    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    locs_small = fr.face_locations(rgb_small, model=model)
                    s = int(round(1.0 / scale))
                    locs_big = [(t*s, r*s, b*s, l*s) for (t, r, b, l) in locs_small]

                # 编码 + 比对
                rec_names, labels, colors = [], [], []
                if locs_big:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encs = fr.face_encodings(frame_rgb, locs_big)
                    for e in encs:
                        d = fr.face_distance(bank, e) if len(bank) else np.array([1.0])
                        idx = int(np.argmin(d))
                        if len(bank) and d[idx] < tol:
                            rec_names.append(names[idx])
                            labels.append(f"{names[idx]} {d[idx]:.2f}")
                            colors.append((0, 255, 0))
                        else:
                            labels.append(f"Unknown {float(d[idx]):.2f}" if len(d) else "Unknown")
                            colors.append((0, 0, 255))

                # 推结果给主线程（满了丢旧，低延迟）
                if result_q.full():
                    try: result_q.get_nowait()
                    except queue.Empty: pass
                try:
                    result_q.put_nowait((frame, locs_big, rec_names, labels, colors, time.time()))
                except queue.Full:
                    pass

        t_grab = Thread(target=grab_loop, daemon=True); t_grab.start()
        t_det  = Thread(target=detect_loop, daemon=True); t_det.start()
        running = True
        return True

    def stop_pipeline():
        nonlocal cap, running
        if not running:
            return
        stop_evt.set()
        time.sleep(0.05)
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cap = None
        # 清空队列
        try:
            while not frame_q.empty():  frame_q.get_nowait()
            while not result_q.empty(): result_q.get_nowait()
        except Exception:
            pass
        running = False

    # ========== 主循环：预览 + 事件/通知 + 暂停恢复 ==========
    while True:
        # 暂停：释放摄像头并停线程
        if PAUSE_FLAG.exists():
            if running:
                logging.info("[暂停] 已释放摄像头，直到删除 PAUSE 文件再继续")
                stop_pipeline()
            time.sleep(0.5)
            continue

        # 未运行则尝试启动
        if not running:
            if not start_pipeline():
                time.sleep(3)
                continue

        # 取一条检测结果（无结果就继续）
        try:
            frame, locs_big, rec_names, labels, colors, ts = result_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # 预览：只有 PREVIEW 文件存在时才编码 JPEG（省 CPU）
        if PREVIEW_FLAG.exists():
            try:
                show = frame.copy()
                for (t, r, b, l), label, color in zip(locs_big, labels, colors):
                    cv2.rectangle(show, (l, t), (r, b), color, 2)
                    cv2.putText(show, label, (l, max(0, t - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                ok_jpg, buf = cv2.imencode(".jpg", show, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok_jpg:
                    globals()["_latest_jpeg"] = buf.tobytes()
            except Exception:
                pass
        else:
            globals()["_latest_jpeg"] = None

        # 任意命中即“全局三连发”（按事件冷却）
        if rec_names:
            now = time.time()
            any_path = OUT_DIR / "known_last.jpg"
            if (now - last_any_save_ts) > auto_save_int or (not any_path.exists()):
                try:
                    cv2.imwrite(str(any_path), show if 'show' in locals() else frame)
                    logging.info("[保存] %s", any_path)
                except Exception:
                    pass
                last_any_save_ts = now

            if (now - last_any_event_ts) >= event_cooldown:
                to_list = (cfg.get("recipients_all") or
                           cfg.get("recipients_map", {}).get("_default", []))
                names_str = "、".join(sorted(set(rec_names)))
                subject = f"已识别到图库人员：{names_str}"[:120]
                body    = f"系统在 {time.strftime('%Y-%m-%d %H:%M:%S')} 识别到：{names_str}"
                attach  = any_path if any_path.exists() else (OUT_DIR / "img.jpg")
                if not attach.exists():
                    cv2.imwrite(str(attach), frame)
                for i in range(3):
                    ok_i = send_mail(to_list, subject, body, attach, smtp_cfg)
                    logging.info("[三连发][ANY] 第 %d/3：%s", i+1, "成功" if ok_i else "失败")
                    time.sleep(0.6)
                last_any_event_ts = time.time()


if __name__ == "__main__":
    logging.info("FaceWatch 后台进程启动（识别即三连发）")
    _start_preview_server()  # ← 启动浏览器预览服务
    while True:
        try:
            run_loop()
        except Exception as e:
            logging.exception("主循环异常：%s", e)
        time.sleep(3)
