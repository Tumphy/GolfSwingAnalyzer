import os
import cv2
import json
import time
import numpy as np
import mediapipe as mp
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from gtts import gTTS
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots
import pygame
from scipy.spatial.transform import Rotation
import threading

# Optional 3D reconstruction
import torch
import torch.nn as nn

# ---------------------------
# CONFIGURATION
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

for d in [OUTPUT_DIR, MODELS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

CONFIG = {
    "use_3d_reconstruction": True,     # Whether to do any 3D logic (Mediapipe or custom)
    "use_custom_3d_model": False,      # If True, load a PyTorch 2D->3D model
    "use_audio_feedback": True,
    "save_fault_report": True,
    "visualize_club_path": True,
    "display_skeleton": True,
    "display_angles": True,
    "club_type": "Driver",
    "debug_mode": False,
    "tempo_threshold": [2.5, 3.5],     # Acceptable ratio of backswing:downswing
}

# --------------------------------
# 1) YOLO & MediaPipe Setup
# --------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def load_yolo_models():
    """Load YOLO model(s). We'll just focus on the club for this example."""
    from ultralytics import YOLO
    
    club_path = os.path.join(MODELS_DIR, "yolo_club.pt")
    if os.path.exists(club_path):
        club_model = YOLO(club_path)
        print(f"Club detection model loaded from: {club_path}")
    else:
        print(f"No custom club model at {club_path}, falling back to yolov8n.pt")
        club_model = YOLO("yolov8n.pt")
    
    return club_model

# Optional custom 2D->3D model
class Pose2Dto3DModel(nn.Module):
    def __init__(self, input_size=66, hidden_size=1024, output_size=99):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

def load_3d_model():
    model_path = os.path.join(MODELS_DIR, "2d_to_3d_model.pth")
    if os.path.exists(model_path):
        print(f"Loading custom 2D->3D model from {model_path}...")
        model = Pose2Dto3DModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        print(f"No custom 3D model found at {model_path}, fallback to MediaPipe partial 3D.")
        return None

# --------------------------------
# 2) Baseline & Phase Data
# --------------------------------
def load_baseline_data():
    path = os.path.join(DATA_DIR, "swing_baseline_data.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            print(f"Loaded baseline data from {path}")
            return data
    else:
        print(f"No baseline data at {path}, using empty defaults.")
        return {
            "Driver": {"Face-On": {}, "Down-the-Line": {}},
            "Iron":   {"Face-On": {}, "Down-the-Line": {}}
        }

def load_swing_phases():
    path = os.path.join(DATA_DIR, "swing_phases.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            sp = json.load(f)
            return sp.get("SwingPhases", {})
    else:
        print("No swing_phases.json found, using defaults.")
        return {
            "0": "Takeaway",
            "1": "Early Backswing",
            "2": "Top of Backswing",
            "3": "Transition",
            "4": "Downswing",
            "5": "Impact",
            "6": "Follow-Through"
        }

# --------------------------------
# 3) Geometry & Phase Detection
# --------------------------------
def calculate_angle(a, b, c, use_3d=False):
    """
    Returns angle ABC in degrees, with B as the vertex.
    a,b,c = [x,y,z], or [x,y]
    If use_3d=True, uses x,y,z. Otherwise just x,y.
    """
    if use_3d and len(a) >= 3 and len(b) >= 3 and len(c) >= 3:
        A = np.array(a[:3])
        B = np.array(b[:3])
        C = np.array(c[:3])
    else:
        A = np.array(a[:2])
        B = np.array(b[:2])
        C = np.array(c[:2])
    BA = A - B
    BC = C - B
    mag_ba = np.linalg.norm(BA)
    mag_bc = np.linalg.norm(BC)
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0
    dot_val = np.clip(np.dot(BA, BC)/(mag_ba*mag_bc), -1.0, 1.0)
    return np.degrees(np.arccos(dot_val))

def detect_view_type(landmarks, confidence_threshold=0.7):
    """
    Determine if the view is Face-On or Down-the-Line
    Return (view_type, confidence)
    
    'landmarks' should be a list of [x, y, z].
    We'll look at the difference in x vs. y for the shoulders.
    """
    import mediapipe as mp
    mp_pose_local = mp.solutions.pose
    
    if (not landmarks) or (len(landmarks) <= mp_pose_local.PoseLandmark.RIGHT_SHOULDER.value):
        return "Unknown", 0.0
    
    lsh = landmarks[mp_pose_local.PoseLandmark.LEFT_SHOULDER.value]
    rsh = landmarks[mp_pose_local.PoseLandmark.RIGHT_SHOULDER.value]
    
    dx = abs(lsh[0] - rsh[0])
    dy = abs(lsh[1] - rsh[1])
    
    if dx > 3*dy:
        vt = "Face-On"
        conf = min(1.0, dx/(dy+0.01))
    elif dx < dy:
        vt = "Down-the-Line"
        conf = min(1.0, dy/(dx+0.01))
    else:
        # tie case
        vt = "Face-On" if dx>dy else "Down-the-Line"
        conf = 0.5 + 0.3*(abs(dx-dy)/(dx+dy))
    
    if conf < confidence_threshold:
        return "Unknown", conf
    return vt, conf

def detect_stable_position(trajectory, start_idx, end_idx, threshold=0.02):
    if end_idx <= start_idx + 5:
        return start_idx
    seg = trajectory[start_idx:end_idx]
    moves = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    stable = moves < threshold
    if not any(stable):
        return start_idx
    max_len = 0
    max_start = 0
    cur_len = 0
    cur_start = 0
    for i, st in enumerate(stable):
        if st:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > max_len:
                max_len = cur_len
                max_start = cur_start
            cur_len = 0
    if cur_len > max_len:
        max_len = cur_len
        max_start = cur_start
    return start_idx + max_start

def detect_impact(trajectory, start_idx):
    if start_idx >= len(trajectory) - 5:
        return len(trajectory) - 1
    v = np.diff(trajectory[start_idx:], axis=0)
    a = np.diff(v, axis=0)
    a_mag = np.linalg.norm(a, axis=1)
    peak = np.argmax(a_mag) + start_idx + 1
    return min(peak, len(trajectory) - 1)

def detect_key_swing_positions(landmarks_history, club_positions=None, mp_pose_obj=mp_pose):
    """
    Return a dict with confidence and 'phases': { "0": frame_idx, "1": frame_idx, ... }
    so we can do advanced phase assignment.
    """
    if len(landmarks_history) < 10:
        return {"confidence": 0.0, "phases": {}}
    
    track_points = []
    if club_positions and len(club_positions) == len(landmarks_history):
        track_points = [cp if cp is not None else [0,0] for cp in club_positions]
    else:
        # fallback to right wrist
        for lmk in landmarks_history:
            if lmk:
                rw = lmk[mp_pose_obj.PoseLandmark.RIGHT_WRIST.value][:2]
                track_points.append(rw)
            else:
                track_points.append([0,0])
    
    traj = np.array(track_points)
    
    # 1) stable start => address
    start_idx = detect_stable_position(traj, 0, min(20, len(traj)//5))
    
    # 2) top of backswing => max x displacement from start
    x_disp = traj[:,0] - traj[start_idx,0]
    candidates = np.where(x_disp > 0)[0]
    if len(candidates)>0:
        top_idx = candidates[np.argmax(x_disp[candidates])]
    else:
        top_idx = len(traj)//3
    
    # 3) impact => peak acceleration after top
    search_start = top_idx + (len(traj) - top_idx)//4
    impact_idx = detect_impact(traj, search_start)
    
    # 4) follow through
    follow_idx = min(len(traj)-1, impact_idx + (impact_idx - top_idx))
    
    # middle phases
    early_backswing_idx = start_idx + (top_idx - start_idx)//3
    transition_idx = top_idx + (impact_idx - top_idx)//4
    downswing_idx = transition_idx + (impact_idx - transition_idx)//2
    
    confidence = min(1.0, max(0.1, (impact_idx - start_idx)/(len(traj)*0.5)))
    phases = {
        "0": start_idx,
        "1": early_backswing_idx,
        "2": top_idx,
        "3": transition_idx,
        "4": downswing_idx,
        "5": impact_idx,
        "6": follow_idx
    }
    return {"confidence": confidence, "phases": phases}

def assign_swing_phase(frame_idx, key_positions, total_frames):
    """
    Using advanced detection results from key_positions to get the correct phase.
    If not found, fallback to heuristic.
    """
    if not key_positions or "phases" not in key_positions:
        return heuristic_swing_phase(frame_idx, total_frames)
    
    ph_dict = key_positions["phases"]
    sorted_phases = sorted([(int(k), v) for k,v in ph_dict.items()], key=lambda x: x[1])
    current_phase_idx = 0
    for pnum, pf in sorted_phases:
        if frame_idx < pf:
            break
        current_phase_idx = pnum
    
    # load possible textual mappings
    swing_phases = load_swing_phases()
    return swing_phases.get(str(current_phase_idx), f"Phase {current_phase_idx}")

def heuristic_swing_phase(frame_idx, total_frames):
    """
    If we cannot do advanced detection, fallback to naive fraction.
    """
    frac = frame_idx/total_frames
    if frac<0.15: return "Takeaway"
    elif frac<0.30: return "Early Backswing"
    elif frac<0.45: return "Top of Backswing"
    elif frac<0.60: return "Transition"
    elif frac<0.75: return "Downswing"
    elif frac<0.85: return "Impact"
    else: return "Follow-Through"

# --------------------------------
# 4) Tempo & Fault Detection
# --------------------------------
def analyze_tempo(key_positions):
    """
    Evaluate backswing vs. downswing frames => ratio => interpret
    """
    phases = key_positions.get("phases", {})
    takeaway = phases.get("0", 0)
    top      = phases.get("2", 0)
    impact   = phases.get("5", 0)
    
    backswing = top - takeaway
    downswing = impact - top
    if downswing <= 0:
        return None, "Error", "No valid downswing or incomplete detection."
    
    ratio = backswing/downswing
    mn, mx = CONFIG["tempo_threshold"]
    if mn <= ratio <= mx:
        return ratio, "Good", f"Tempo ratio {ratio:.2f}:1 is within {mn}-{mx}."
    elif ratio<mn:
        return ratio, "Fast", f"Backswing is too quick vs. downswing ({ratio:.2f}:1)."
    else:
        return ratio, "Slow", f"Backswing is too slow vs. downswing ({ratio:.2f}:1)."

def evaluate_angle(angle_name, angle_value, baseline_phase, buffer=5.0):
    """
    Compare angle to baseline => ("Good"/"Acceptable"/"Fault", color, message).
    baseline_phase e.g. { "ShoulderTurn": "20-40°", ... }
    """
    if angle_name not in baseline_phase:
        return "Unknown", (200,200,200), f"{angle_name}: {angle_value:.1f}°"
    
    rng_str = baseline_phase[angle_name].replace("°","")
    try:
        mn, mx = map(float, rng_str.split("-"))
    except:
        mn, mx = (0,180)
    
    if mn<=angle_value<=mx:
        return "Good", (0,255,0), f"{angle_name}: {angle_value:.1f}° ✓"
    elif (mn - buffer)<=angle_value<=(mx + buffer):
        return "Acceptable", (0,255,255), f"{angle_name}: {angle_value:.1f}° (~{mn}-{mx})"
    else:
        # fault
        color = (0,0,255)
        if angle_value<mn:
            direction = f"too low, target={mn}"
        else:
            direction = f"too high, target={mx}"
        return "Fault", color, f"{angle_name}: {angle_value:.1f}° ({direction})"

def detect_faults(angles, viewpoint, club_type, phase, baseline_data):
    """
    angles: { "ShoulderTurn": val, ... }
    Return a list of fault dicts.
    """
    fs = []
    base_phase = baseline_data.get(club_type, {}).get(viewpoint, {}).get(phase, {})
    if not base_phase:
        return fs
    for aname, aval in angles.items():
        status, _, _ = evaluate_angle(aname, aval, base_phase)
        if status=="Fault":
            fs.append({
                "type": "AngleFault",
                "name": aname,
                "value": aval,
                "phase": phase
            })
    return fs

# --------------------------------
# 5) Visualization
# --------------------------------
def draw_skeleton(frame, landmarks, connections=mp_pose.POSE_CONNECTIONS):
    """Simple function to draw skeleton lines on the frame."""
    if not landmarks:
        return frame
    h,w,_ = frame.shape
    
    # Convert normalized coords to px
    pts = []
    for lm in landmarks:
        if len(lm)>=2:
            x = int(lm[0]*w)
            y = int(lm[1]*h)
            pts.append((x,y))
        else:
            pts.append((0,0))
    
    # Draw lines
    for c in connections:
        start, end = c
        if start<len(pts) and end<len(pts):
            pt1 = pts[start]
            pt2 = pts[end]
            cv2.line(frame, pt1, pt2, (0,255,0), 2)
            cv2.circle(frame, pt1, 3, (255,0,0), -1)
            cv2.circle(frame, pt2, 3, (255,0,0), -1)
    return frame

# --------------------------------
# 6) Main Two-Pass Flow
# --------------------------------
def main():
    # Initialize audio if needed
    if CONFIG["use_audio_feedback"]:
        pygame.mixer.init()

    # Load baseline data & YOLO
    baseline_data = load_baseline_data()
    swing_phases_data = load_swing_phases()
    club_model = load_yolo_models()
    
    # Optional custom 2D->3D
    model_3d = None
    if CONFIG["use_3d_reconstruction"] and CONFIG["use_custom_3d_model"]:
        model_3d = load_3d_model()
    
    # 1) Prompt user for a video (mov, mp4, etc.)
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a Golf Swing Video",
        filetypes=[("Video Files","*.mp4 *.mov *.avi *.mkv *.3gp *.webm"), ("All Files","*.*")]
    )
    if not video_path:
        print("No video selected, exiting.")
        return
    
    # Attempt to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ---------------------------------------------------------
    # PASS 1: Collect Landmarks for advanced phase detection
    # ---------------------------------------------------------
    print("=== PASS 1: Gathering landmarks ===")
    landmarks_history = []
    club_positions    = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Pose detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        land_3d = None
        
        if results.pose_landmarks:
            mp_lm = results.pose_landmarks.landmark
            land_3d = [[lm.x, lm.y, lm.z] for lm in mp_lm]
            
            # If custom 2D->3D
            if model_3d:
                land_2d = np.array([[lm.x, lm.y] for lm in mp_lm]).flatten()[None,...]  # shape [1,66]
                t_in = torch.tensor(land_2d, dtype=torch.float32)
                with torch.no_grad():
                    pred = model_3d(t_in).numpy()  # shape [1,99]
                land_3d = pred.reshape(-1,3).tolist()
        
        landmarks_history.append(land_3d)
        
        # YOLO for club detection
        club_res = club_model(frame_rgb)
        best_center = None
        for det in club_res:
            for box in det.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                best_center = (cx/width, cy/height)
                break
        club_positions.append(best_center)
    
    cap.release()
    
    # Detect phases & do tempo analysis
    key_positions = detect_key_swing_positions(landmarks_history, club_positions)
    tempo_ratio, tempo_status, tempo_msg = analyze_tempo(key_positions)
    
    print(f"Key swing positions detected with confidence={key_positions['confidence']:.2f}")
    print(f"Tempo => ratio={tempo_ratio}, status={tempo_status}, desc={tempo_msg}")
    
    # ---------------------------------------------------------
    # PASS 2: Re-read video & annotate using advanced phases
    # ---------------------------------------------------------
    print("=== PASS 2: Annotating output ===")
    cap2 = cv2.VideoCapture(video_path)
    out_name = os.path.splitext(os.path.basename(video_path))[0] + "_analyzed.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width,height))
    
    all_faults = []
    frame_idx = 0
    viewpoint = None
    viewpoint_conf = 0.0
    
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame_idx += 1
        
        # Landmarks from pass1
        lmarks_3d = landmarks_history[frame_idx-1]  # 0-based index
        # Advanced phase
        phase_name = assign_swing_phase(frame_idx-1, key_positions, total_frames)
        
        # Attempt viewpoint detection on first frames that have good landmarks
        if viewpoint is None and lmarks_3d:
            vt, conf = detect_view_type(lmarks_3d)
            if vt!="Unknown" and conf>0.5:
                viewpoint = vt
                viewpoint_conf = conf
        
        # Draw skeleton if we have landmarks
        if lmarks_3d and CONFIG["display_skeleton"]:
            frame = draw_skeleton(frame, lmarks_3d)
        
        # Calculate angles
        angles = {}
        if lmarks_3d and len(lmarks_3d)>mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
            LHIP = lmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value]
            RHIP = lmarks_3d[mp_pose.PoseLandmark.RIGHT_HIP.value]
            LSH  = lmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            RSH  = lmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            angles["ShoulderTurn"] = calculate_angle(LHIP, RHIP, LSH, use_3d=CONFIG["use_3d_reconstruction"])
            angles["SpineAngle"]   = calculate_angle(LHIP, RHIP, RSH, use_3d=CONFIG["use_3d_reconstruction"])
        
        # fallback
        if viewpoint is None:
            viewpoint = "Face-On"
        
        # Fault detection
        frame_faults = detect_faults(angles, viewpoint, CONFIG["club_type"], phase_name, baseline_data)
        for fobj in frame_faults:
            fobj["frame"] = frame_idx
        all_faults.extend(frame_faults)
        
        # Display angles
        if CONFIG["display_angles"]:
            y_offset = 30
            for aname, aval in angles.items():
                base_phase = baseline_data.get(CONFIG["club_type"], {}).get(viewpoint, {}).get(phase_name, {})
                st, color, msg = evaluate_angle(aname, aval, base_phase)
                cv2.putText(frame, f"{aname}: {aval:.1f}", (30,y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
            # show phase
            cv2.putText(frame, f"Phase: {phase_name}", (30,y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        
        out_writer.write(frame)
    
    cap2.release()
    out_writer.release()
    
    # Summarize
    if all_faults and CONFIG["save_fault_report"]:
        report_data = {
            "video": os.path.basename(video_path),
            "phases_confidence": key_positions["confidence"],
            "tempo_ratio": tempo_ratio,
            "tempo_status": tempo_status,
            "tempo_message": tempo_msg,
            "faults": all_faults
        }
        fault_report_path = os.path.join(OUTPUT_DIR, "fault_report.json")
        with open(fault_report_path, "w") as ff:
            json.dump(report_data, ff, indent=2)
        
        print(f"Fault report saved to: {fault_report_path}")
        # Generate text feedback
        feedback_str = generate_fault_feedback(all_faults, phase_specific=True)
        
        # Add tempo info
        if tempo_ratio is not None:
            feedback_str += f"\nTempo => {tempo_status}: {tempo_msg}"
        
        print("*** Fault Feedback ***\n", feedback_str)
        
        # Audio feedback
        if CONFIG["use_audio_feedback"]:
            audio_data = gTTS(text=feedback_str, lang='en')
            mp3_path = os.path.join(OUTPUT_DIR, "feedback.mp3")
            audio_data.save(mp3_path)
            def play_audio():
                pygame.mixer.music.load(mp3_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.stop()
            threading.Thread(target=play_audio).start()
    else:
        if not all_faults:
            print("No major faults detected! Great job!")
        print(f"Your tempo ratio => {tempo_ratio} ({tempo_status}): {tempo_msg}")
    
    final_vid_path = os.path.join(OUTPUT_DIR, out_name)
    print(f"Annotated video saved to: {final_vid_path}")
    print("Analysis complete!")

def generate_fault_feedback(all_faults, phase_specific=True):
    """Simple function to turn a list of faults into readable text."""
    if not all_faults:
        return "No significant faults detected in your swing."
    
    from collections import defaultdict
    ph_map = defaultdict(list)
    for f in all_faults:
        ph_map[f["phase"]].append(f)
    
    lines = []
    for phase_name in sorted(ph_map.keys()):
        lines.append(f"\n*** {phase_name} ***")
        for fault in ph_map[phase_name]:
            aname = fault["name"]
            aval  = fault["value"]
            lines.append(f"  • {aname} => {aval:.1f}° (Frame {fault['frame']}) is out of range.")
    
    return "\n".join(lines)

if __name__ == "__main__":
    main()

