import cv2
import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime, date
from PIL import Image, ImageTk
import threading
import numpy as np

# ── Optional DeepFace import with clear error ─────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
KNOWN_DIR = os.path.join(BASE_DIR, "known_faces")
LOGS_DIR  = os.path.join(BASE_DIR, "attendance_logs")

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ── Helpers ────────────────────────────────────────────────────────────────
def get_today_log():
    return os.path.join(LOGS_DIR, date.today().strftime("%Y-%m-%d") + "_attendance.csv")


def mark_attendance(name):
    log_file = get_today_log()
    already  = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for row in csv.DictReader(f):
                already.add(row.get("Name", ""))
    if name not in already:
        write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        with open(log_file, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["Name", "Date", "Time", "Status"])
            w.writerow([name, date.today().strftime("%Y-%m-%d"),
                        datetime.now().strftime("%H:%M:%S"), "Present"])
        return True
    return False


def recognize_face(face_img):
    """Compare face_img (BGR numpy array) against known_faces folder."""
    if not DEEPFACE_OK:
        return "Unknown"
    best_name, best_dist = "Unknown", 999
    for fname in os.listdir(KNOWN_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        ref_path = os.path.join(KNOWN_DIR, fname)
        try:
            result = DeepFace.verify(face_img, ref_path,
                                     model_name="VGG-Face",
                                     enforce_detection=False,
                                     silent=True)
            dist = result.get("distance", 999)
            if result.get("verified") and dist < best_dist:
                best_dist = dist
                best_name = os.path.splitext(fname)[0]
        except Exception:
            pass
    return best_name


# ── Main App ───────────────────────────────────────────────────────────────
class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        self.camera_active = False
        self.cap           = None
        self.today_present = set()
        self.recognizing   = False   # lock so only 1 DeepFace call at a time

        self._build_ui()
        self._load_today_attendance()
        self._update_clock()

        if not DEEPFACE_OK:
            messagebox.showwarning(
                "DeepFace not installed",
                "Run:  pip install deepface tf-keras\n\n"
                "Face detection will still work, but recognition is disabled until installed."
            )

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg="#16213e", height=60)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="🎓 Face Recognition Attendance System",
                 font=("Helvetica", 18, "bold"), bg="#16213e", fg="white", pady=15
                 ).pack(side=tk.LEFT, padx=20)
        self.clock_lbl = tk.Label(hdr, font=("Helvetica", 12), bg="#16213e", fg="#a8dadc")
        self.clock_lbl.pack(side=tk.RIGHT, padx=20)

        # Body
        body = tk.Frame(self.root, bg="#1a1a2e")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(body, bg="#1a1a2e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(body, bg="#1a1a2e", width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        # Camera area
        cam_f = tk.Frame(left, bg="#16213e", relief=tk.RAISED, bd=2)
        cam_f.pack(fill=tk.BOTH, expand=True)
        self.video_lbl = tk.Label(cam_f, bg="#0f3460",
                                  text="📷  Camera Feed\n\nPress  ▶ Start Camera  to begin",
                                  font=("Helvetica", 14), fg="#a8dadc")
        self.video_lbl.pack(fill=tk.BOTH, expand=True)

        self.status_lbl = tk.Label(left, text="System Ready",
                                   font=("Helvetica", 10), bg="#0f3460",
                                   fg="#a8dadc", pady=5)
        self.status_lbl.pack(fill=tk.X, pady=(5, 0))

        # Buttons
        ctrl = tk.Frame(left, bg="#1a1a2e")
        ctrl.pack(fill=tk.X, pady=8)
        self.start_btn = self._btn(ctrl, "▶ Start Camera",  "#2ecc71", self.start_camera)
        self.stop_btn  = self._btn(ctrl, "⏹ Stop Camera",   "#e74c3c", self.stop_camera, tk.DISABLED)
        reg_btn        = self._btn(ctrl, "➕ Register Face", "#3498db", self.register_face)
        log_btn        = self._btn(ctrl, "📋 View Log",      "#9b59b6", self.view_log)
        for b in (self.start_btn, self.stop_btn, reg_btn, log_btn):
            b.pack(side=tk.LEFT, padx=4)

        # ── Right panel ────────────────────────────────────────────────────
        self._panel_label(right, "📊 Today's Attendance")
        stats = tk.Frame(right, bg="#16213e")
        stats.pack(fill=tk.X, padx=8, pady=4)
        self.present_lbl  = self._stat(stats, "Present",  "0",
                                       str(len(self.today_present)), "#2ecc71")
        enrolled = len([f for f in os.listdir(KNOWN_DIR)
                        if f.lower().endswith((".jpg",".jpeg",".png"))])
        self.enrolled_lbl = self._stat(stats, "Enrolled", str(enrolled), str(enrolled), "#3498db")

        list_f = tk.Frame(right, bg="#0f3460", relief=tk.SUNKEN, bd=1)
        list_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#0f3460", foreground="white",
                        fieldbackground="#0f3460", rowheight=28)
        style.configure("Treeview.Heading", background="#16213e",
                        foreground="#a8dadc", font=("Helvetica", 10, "bold"))
        style.map("Treeview", background=[("selected", "#2ecc71")],
                  foreground=[("selected", "#000000")])

        self.tree = ttk.Treeview(list_f, columns=("Name", "Time"), show="headings", height=15)
        self.tree.heading("Name", text="👤 Name")
        self.tree.heading("Time", text="🕐 Time")
        self.tree.column("Name", width=150)
        self.tree.column("Time", width=80)
        sb = ttk.Scrollbar(list_f, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._panel_label(right, "👥 Registered People")
        self.enrolled_list = tk.Listbox(right, bg="#0f3460", fg="white",
                                        font=("Helvetica", 10), height=6,
                                        selectbackground="#3498db", relief=tk.FLAT)
        self.enrolled_list.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._refresh_enrolled()

    def _btn(self, p, text, color, cmd, state=tk.NORMAL):
        return tk.Button(p, text=text, font=("Helvetica", 10, "bold"),
                         bg=color, fg="white", relief=tk.FLAT, padx=12, pady=6,
                         cursor="hand2", command=cmd, state=state,
                         activebackground=color, activeforeground="white")

    def _panel_label(self, p, text):
        tk.Label(p, text=text, font=("Helvetica", 11, "bold"),
                 bg="#16213e", fg="#a8dadc", pady=6).pack(fill=tk.X, padx=8, pady=(8, 0))

    def _stat(self, p, label, value, _unused, color):
        f = tk.Frame(p, bg="#0f3460", relief=tk.RAISED, bd=1, padx=10, pady=6)
        f.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        lbl = tk.Label(f, text=value, font=("Helvetica", 22, "bold"), bg="#0f3460", fg=color)
        lbl.pack()
        tk.Label(f, text=label, font=("Helvetica", 9), bg="#0f3460", fg="#a8dadc").pack()
        return lbl

    def _update_clock(self):
        self.clock_lbl.config(text=datetime.now().strftime("%A, %d %b %Y   %H:%M:%S"))
        self.root.after(1000, self._update_clock)

    # ── Camera ─────────────────────────────────────────────────────────────
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.\nCheck your camera is connected.")
            return
        self.camera_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text="📡 Camera Active – Scanning for faces…", fg="#2ecc71")
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_lbl.config(image="",
                              text="📷  Camera Feed\n\nPress  ▶ Start Camera  to begin")
        self.status_lbl.config(text="Camera stopped.", fg="#a8dadc")

    def _camera_loop(self):
        frame_n = 0
        last_results = {}   # face_id -> name  (persists a few frames)

        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_n += 1
            display = frame.copy()
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces   = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            for i, (x, y, w, h) in enumerate(faces):
                face_crop = frame[y:y+h, x:x+w]
                name  = last_results.get(i, "Detecting…")
                color = (0, 200, 255)

                # Run recognition in background every 10 frames
                if frame_n % 10 == 0 and not self.recognizing and DEEPFACE_OK:
                    self.recognizing = True
                    crop_copy = face_crop.copy()
                    idx       = i
                    def _recognize(crop=crop_copy, face_idx=idx):
                        result = recognize_face(crop)
                        last_results[face_idx] = result
                        self.recognizing = False
                        if result != "Unknown":
                            if mark_attendance(result):
                                self.root.after(0, self._on_new_attendance, result)
                    threading.Thread(target=_recognize, daemon=True).start()

                if name not in ("Detecting…", "Unknown"):
                    color = (0, 255, 0)
                elif name == "Unknown":
                    color = (0, 0, 255)

                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(display, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(display, name, (x+6, y+h-8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1)

            # Render
            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((760, 500))
            photo = ImageTk.PhotoImage(image=img)
            self.root.after(0, lambda p=photo: self._show_frame(p))

        self.root.after(0, self.stop_camera)

    def _show_frame(self, photo):
        self.video_lbl.config(image=photo, text="")
        self.video_lbl.image = photo

    def _on_new_attendance(self, name):
        t = datetime.now().strftime("%H:%M:%S")
        self.today_present.add(name)
        self.tree.insert("", 0, values=(name, t), tags=("new",))
        self.tree.tag_configure("new", background="#1a472a", foreground="#2ecc71")
        self.present_lbl.config(text=str(len(self.today_present)))
        self.status_lbl.config(text=f"✅ {name} marked present at {t}", fg="#2ecc71")

    # ── Register ────────────────────────────────────────────────────────────
    def register_face(self):
        name = simpledialog.askstring("Register", "Enter full name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip().title()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam.")
            return

        win = tk.Toplevel(self.root)
        win.title(f"Register: {name}")
        win.geometry("520x500")
        win.configure(bg="#1a1a2e")
        win.grab_set()

        tk.Label(win, text=f"Registering: {name}",
                 font=("Helvetica", 13, "bold"), bg="#1a1a2e", fg="white").pack(pady=8)
        tk.Label(win, text="Position your face clearly, then click Capture.",
                 font=("Helvetica", 10), bg="#1a1a2e", fg="#a8dadc").pack()

        vid = tk.Label(win, bg="#0f3460")
        vid.pack(pady=6, padx=10, fill=tk.BOTH, expand=True)

        last_frame = [None]
        running    = [True]

        def _feed():
            if not running[0]:
                return
            ret, f = cap.read()
            if ret:
                last_frame[0] = f.copy()
                gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
                for (x, y, w, h) in faces:
                    cv2.rectangle(f, (x, y), (x+w, y+h), (0, 255, 0), 2)
                img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                img.thumbnail((480, 360))
                ph = ImageTk.PhotoImage(img)
                vid.config(image=ph)
                vid.image = ph
            win.after(33, _feed)

        _feed()

        def capture():
            f = last_frame[0]
            if f is None:
                messagebox.showwarning("Wait", "No frame yet.", parent=win)
                return
            gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) == 0:
                messagebox.showwarning("No Face", "No face detected. Try again.", parent=win)
                return

            path = os.path.join(KNOWN_DIR, f"{name}.jpg")
            cv2.imwrite(path, f)
            enrolled = len([x for x in os.listdir(KNOWN_DIR)
                            if x.lower().endswith((".jpg",".jpeg",".png"))])
            self.enrolled_lbl.config(text=str(enrolled))
            self._refresh_enrolled()
            running[0] = False
            cap.release()
            win.destroy()
            messagebox.showinfo("Done", f"✅ {name} registered successfully!")

        def on_close():
            running[0] = False
            cap.release()
            win.destroy()

        tk.Button(win, text="📸 Capture & Register",
                  font=("Helvetica", 11, "bold"), bg="#2ecc71", fg="white",
                  relief=tk.FLAT, padx=14, pady=8, cursor="hand2",
                  command=capture).pack(pady=6)
        win.protocol("WM_DELETE_WINDOW", on_close)

    # ── View Log ────────────────────────────────────────────────────────────
    def view_log(self):
        log = get_today_log()
        win = tk.Toplevel(self.root)
        win.title("Today's Attendance Log")
        win.geometry("520x420")
        win.configure(bg="#1a1a2e")
        tk.Label(win, text=f"📋  {date.today().strftime('%A, %d %b %Y')}",
                 font=("Helvetica", 13, "bold"), bg="#1a1a2e", fg="white", pady=10).pack()

        cols = ("Name", "Date", "Time", "Status")
        t = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            t.heading(c, text=c)
            t.column(c, width=110)
        t.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        if os.path.exists(log):
            with open(log, "r") as f:
                for row in csv.DictReader(f):
                    t.insert("", tk.END, values=(row["Name"], row["Date"], row["Time"], row["Status"]))
        else:
            tk.Label(win, text="No attendance recorded today.",
                     bg="#1a1a2e", fg="#a8dadc", font=("Helvetica", 11)).pack(pady=20)

        tk.Button(win, text=f"📁 Log saved at:\n{log}",
                  font=("Helvetica", 9), bg="#9b59b6", fg="white",
                  relief=tk.FLAT, padx=10, pady=5, wraplength=460,
                  command=lambda: messagebox.showinfo("Path", log, parent=win)
                  ).pack(pady=(0, 10))

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _load_today_attendance(self):
        log = get_today_log()
        if not os.path.exists(log):
            return
        with open(log, "r") as f:
            for row in csv.DictReader(f):
                self.today_present.add(row["Name"])
                self.tree.insert("", tk.END, values=(row["Name"], row["Time"]))
        self.present_lbl.config(text=str(len(self.today_present)))

    def _refresh_enrolled(self):
        self.enrolled_list.delete(0, tk.END)
        for f in sorted(os.listdir(KNOWN_DIR)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.enrolled_list.insert(tk.END, f"  👤 {os.path.splitext(f)[0]}")

    def on_closing(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ── Entry ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
