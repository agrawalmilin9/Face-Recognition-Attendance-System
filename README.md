# 🎓 Face Recognition Attendance System

Real-time webcam attendance using **DeepFace + OpenCV**.

---

## ✅ Install (one command)

```bash
pip install opencv-python deepface tf-keras numpy Pillow
```

---

## 🚀 Run

```bash
python attendance_system.py
```

---

## 🖥️ How to Use

### 1 – Register a person
1. Click **➕ Register Face**
2. Type their name → OK
3. Face appears in frame with a green box
4. Click **📸 Capture & Register**

### 2 – Take attendance
1. Click **▶ Start Camera**
2. Stand in front of webcam
3. Green box + name = ✅ attendance auto-marked
4. Yellow = detecting, Red = unknown

### 3 – View records
- Live list on the right panel
- Click **📋 View Log** for full table
- CSV saved in `attendance_logs/YYYY-MM-DD_attendance.csv`

---

## 📁 Folder Structure

```
attendance_system/
├── attendance_system.py
├── requirements.txt
├── known_faces/          ← registered face photos (.jpg)
└── attendance_logs/      ← daily CSVs
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| First recognition is slow | Normal — DeepFace loads model on first run (~10 sec) |
| Face not recognized | Re-register in the same lighting you'll use daily |
| Camera index error | Change `cv2.VideoCapture(0)` → `(1)` |
| `tf-keras` error | Run `pip install tf-keras --upgrade` |
