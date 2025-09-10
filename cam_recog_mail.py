# cam_recog_mail.py —— 实时摄像头识别 + 命中自动群发邮件（附当前画面）
# 使用说明：
# 1) 准备一张目标人物参考照 hy.jpg（正脸、清晰、单人）
# 2) 配置 SMTP_* 四项（发件邮箱、授权码等），准备 recipients.csv（列名 email）
# 3) 运行后：识别到目标 -> 立即连发3封（一次性），之后不再发送；按 R 可复位；E 手动发送也受上限限制
#    按 Q/ESC 退出，按 S 手动保存截图到 output/snapshot_*.jpg

import time
import csv
import ssl
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np
import face_recognition as fr

# ======== 识别参数 ========
KNOWN_IMG = "hy.jpg"   # 参考照文件名（放在脚本同目录）
TOLERANCE = 0.45       # 比对阈值（越小越严格）
SCALE = 0.5            # 检测时的下采样系数，0.5=缩小一半
DRAW_NAME = "Target"   # 命中显示的名字
AUTO_SAVE_INTERVAL = 5 # 自动写 output/img.jpg 的最短间隔（秒）
MAX_SENDS = 3          # ✅ 本次运行最多发送3封（自动连发+手动都计入）
# ==========================

# ======== 邮件配置（按你邮箱来）========
# 选择其一填写（QQ/163/Outlook等），授权码需在邮箱里开通“客户端授权码/应用专用密码”
SMTP_HOST = "smtp.qq.com"      # QQ邮箱示例：smtp.qq.com
SMTP_PORT = 587                # 465=SSL，587=STARTTLS（Outlook 365常用587）
SENDER    = "2942566932@qq.com"
AUTH      = "jqsjtpjsevkydehe"    # 注意不是登录密码
SUBJECT   = "已识别到指定人员"
BODY      = "系统刚刚识别到：{}".format(DRAW_NAME)
RECIPIENTS_CSV = "recipients.csv"  # 收件人列表（第一行列名 email）
# =====================================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)
AUTO_SAVE_PATH = OUT_DIR / "img.jpg"     # 提供给你的其它流程使用的固定文件

def load_known(path: Path):
    img = fr.load_image_file(str(path))
    encs = fr.face_encodings(img)
    assert encs, f"{path.name} 没检测到人脸，请换更清晰的正脸参考照"
    return encs[0]

def open_cam():
    for i in (0, 1, 2):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows上DSHOW稳定
        if cap.isOpened():
            print("使用摄像头索引:", i)
            return cap
    raise RuntimeError("找不到可用摄像头（改用索引 1/2 试试）")

def load_recipients(csv_path: Path):
    recvs = []
    if not csv_path.exists():
        print(f"[警告] 未找到 {csv_path}，将不会发送邮件。")
        return recvs
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mail = (row.get("email") or "").strip()
            if mail:
                recvs.append(mail)
    if not recvs:
        print(f"[警告] {csv_path} 中没有有效邮箱；将不会发送邮件。")
    return recvs

def send_mail(to_list, subject, body, attach_path: Path | None):
    import smtplib, ssl

    if not to_list:
        print("[邮件] 收件人为空，跳过发送")
        return False

    msg = EmailMessage()
    msg["From"] = SENDER
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject
    msg.set_content(body)

    if attach_path and attach_path.exists():
        data = attach_path.read_bytes()
        msg.add_attachment(data, maintype="image", subtype="jpeg", filename=attach_path.name)

    s = None
    try:
        if SMTP_PORT == 465:
            s = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30, context=ssl.create_default_context())
        else:
            s = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            s.ehlo()
            s.starttls(context=ssl.create_default_context())
            s.ehlo()

        s.login(SENDER, AUTH)

        refused = s.send_message(msg)  # 空 dict == 全部成功
        if refused:
            print("[邮件] 部分拒收：", refused)
            return False

        print("[邮件] 已发送：", to_list)
        return True

    except smtplib.SMTPAuthenticationError as e:
        print("[邮件] 认证失败（检查授权码/是否开启SMTP）：", e)
        return False
    except smtplib.SMTPResponseException as e:
        print(f"[邮件] 服务器返回错误：{e.smtp_code} {e.smtp_error}")
        return False
    except Exception as e:
        print("[邮件] 发送异常：", e)
        return False
    finally:
        if s is not None:
            try:
                s.quit()
            except Exception:
                pass

def main():
    known_enc = load_known(BASE_DIR / KNOWN_IMG)
    # 如需从 CSV 读取，改回下面这一行
    # recipients = load_recipients(BASE_DIR / RECIPIENTS_CSV)
    recipients = ["3033146640@qq.com","jokertom285@gmail.com","2942566932@qq.com"]
    print("[收件人]", recipients)

    cap = open_cam()
    cv2.namedWindow("Face Watch", cv2.WINDOW_NORMAL)

    last_hit_ts = 0.0
    last_auto_save_ts = 0.0
    send_success_count = 0           # ✅ 已成功发送计数
    sent_done = False                # ✅ 达到上限后不再自动发送
    process_toggle = True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头失败")
            break

        show = frame  # 先用原图显示
        hits, locs_big = [], []

        if process_toggle:
            # 下采样+颜色空间转换 —— 用 cv2.cvtColor 保证连续内存
            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # 检测 & 编码
            locs = fr.face_locations(rgb, model="hog")
            encs = fr.face_encodings(rgb, locs)
            hits = [fr.compare_faces([known_enc], e, tolerance=TOLERANCE)[0] for e in encs]

            # 坐标映射回原图
            s = int(round(1.0 / SCALE))
            locs_big = [(t*s, r*s, b*s, l*s) for (t, r, b, l) in locs]

            # 画框到预览图
            show = frame.copy()
            for (t, r, b, l), hit in zip(locs_big, hits):
                color = (0, 255, 0) if hit else (0, 0, 255)
                cv2.rectangle(show, (l, t), (r, b), color, 2)
                cv2.putText(show, DRAW_NAME if hit else "Unknown",
                            (l, max(0, t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if any(hits):
                last_hit_ts = time.time()
                # 自动保存最新画面
                if (time.time() - last_auto_save_ts) > AUTO_SAVE_INTERVAL:
                    cv2.imwrite(str(AUTO_SAVE_PATH), show)
                    print("[自动保存]", AUTO_SAVE_PATH)
                    last_auto_save_ts = time.time()

                # ✅ 命中后一次性连发至多3封（只执行一次）
                if not sent_done:
                    # 确保有可用截图
                    if not AUTO_SAVE_PATH.exists():
                        cv2.imwrite(str(AUTO_SAVE_PATH), show)

                    remaining = MAX_SENDS - send_success_count
                    for _ in range(remaining):
                        ok = send_mail(recipients, SUBJECT, BODY, AUTO_SAVE_PATH)
                        if ok:
                            send_success_count += 1
                        time.sleep(0.6)  # 轻微间隔，避免被限流

                    sent_done = True
                    print(f"[提示] 已达上限：{send_success_count}/{MAX_SENDS}，不再自动发送（按 R 可复位）")

        process_toggle = not process_toggle

        # 左上角状态提示
        hit_text = time.strftime('%H:%M:%S', time.localtime(last_hit_ts)) if last_hit_ts else '-'
        done_tag = " DONE" if sent_done else ""
        status = f"Hit:{hit_text} Tol={TOLERANCE} Sent {send_success_count}/{MAX_SENDS}{done_tag}"
        cv2.putText(show, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Watch", show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q / ESC
            break
        if key == ord('s'):
            snap = OUT_DIR / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(snap), show)
            print("[手动保存]", snap)
        if key == ord('e'):
            # 手动发送也受上限约束
            if sent_done:
                print("[提示] 已达上限，按 R 重置后再试")
            else:
                if AUTO_SAVE_PATH.exists() or cv2.imwrite(str(AUTO_SAVE_PATH), show):
                    ok = send_mail(recipients, SUBJECT + "（手动）", BODY, AUTO_SAVE_PATH)
                    if ok:
                        send_success_count += 1
                        print(f"[计数] 已发送 {send_success_count}/{MAX_SENDS}")
                        if send_success_count >= MAX_SENDS:
                            sent_done = True
                            print("[提示] 已达上限，不再发送（按 R 可复位）")
        if key in (ord('r'), ord('R')):
            # ✅ 复位计数与状态
            sent_done = False
            send_success_count = 0
            print("[复位] 发送计数已清零，可再次最多发送3封")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
