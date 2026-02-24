import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from collections import deque
import urllib.request
import glob

# MediaPipe Task API setup
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model():
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        print("Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path

def calculate_angle(a, b, c):
    """Calculates angle at point b using pixel coordinates."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_video(video_path, output_dir, options_base, options_dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        cap.release()
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analyzed_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Trajectory storage
    keys = ['pelvis', 'l_hand', 'r_hand', 'l_foot', 'r_foot', 'l_elbow', 'r_elbow', 'l_knee', 'r_knee', 'com']
    trajectories = {k: deque(maxlen=options_dict['traj_len']) for k in keys}
    prev_com, velocity = None, 0.0

    colors = {
        'pelvis': (255, 0, 0), 'l_hand': (0, 255, 0), 'r_hand': (128, 255, 0),
        'l_foot': (0, 0, 255), 'r_foot': (128, 0, 255), 'l_elbow': (255, 255, 0),
        'r_elbow': (0, 255, 255), 'l_knee': (0, 165, 255), 'r_knee': (255, 0, 255),
        'com': (255, 255, 255)
    }

    print(f"Processing {os.path.basename(video_path)} (Persistent Angles Mode)...")

    with vision.PoseLandmarker.create_from_options(options_base) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            timestamp_ms = int(frame_idx * 1000 / fps)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            image = frame.copy()
            if pose_landmarker_result.pose_landmarks:
                landmarks = pose_landmarker_result.pose_landmarks[0]
                idx = {'l_hip': 23, 'r_hip': 24, 'l_wrist': 15, 'r_wrist': 16, 'l_ankle': 27, 'r_ankle': 28, 
                       'l_shoulder': 11, 'r_shoulder': 12, 'l_elbow': 13, 'r_elbow': 14, 'l_knee': 25, 'r_knee': 26}
                
                def is_vis(i, thresh=0.3): 
                    return landmarks[i].visibility > thresh

                def get_p(i): 
                    return [landmarks[i].x, landmarks[i].y]
                
                def get_p_px(i): 
                    return [landmarks[i].x * width, landmarks[i].y * height]

                # Trajectories (Keep visibility check for trajectory to avoid jitter)
                if is_vis(idx['l_wrist']): trajectories['l_hand'].append(get_p(idx['l_wrist']))
                if is_vis(idx['r_wrist']): trajectories['r_hand'].append(get_p(idx['r_wrist']))
                if is_vis(idx['l_ankle']): trajectories['l_foot'].append(get_p(idx['l_ankle']))
                if is_vis(idx['r_ankle']): trajectories['r_foot'].append(get_p(idx['r_ankle']))
                if is_vis(idx['l_elbow']): trajectories['l_elbow'].append(get_p(idx['l_elbow']))
                if is_vis(idx['r_elbow']): trajectories['r_elbow'].append(get_p(idx['r_elbow']))
                if is_vis(idx['l_knee']): trajectories['l_knee'].append(get_p(idx['l_knee']))
                if is_vis(idx['r_knee']): trajectories['r_knee'].append(get_p(idx['r_knee']))

                hip_vis = is_vis(idx['l_hip']) and is_vis(idx['r_hip'])
                shoulder_vis = is_vis(idx['l_shoulder']) and is_vis(idx['r_shoulder'])

                if hip_vis:
                    l_h, r_h = get_p(idx['l_hip']), get_p(idx['r_hip'])
                    trajectories['pelvis'].append([(l_h[0]+r_h[0])/2, (l_h[1]+r_h[1])/2])

                if hip_vis and shoulder_vis:
                    l_h, r_h, l_s, r_s = get_p(idx['l_hip']), get_p(idx['r_hip']), get_p(idx['l_shoulder']), get_p(idx['r_shoulder'])
                    com = np.mean([l_h, r_h, l_s, r_s], axis=0)
                    trajectories['com'].append(com)
                    if prev_com is not None:
                        velocity = np.sqrt(np.sum((com - prev_com)**2)) * fps
                    prev_com = com

                # Persistent Angle Calculation (No visibility check for display)
                angles = {}
                if options_dict['show_l_hand']:
                    angles['L Elbow'] = calculate_angle(get_p_px(idx['l_shoulder']), get_p_px(idx['l_elbow']), get_p_px(idx['l_wrist']))
                if options_dict['show_r_hand']:
                    angles['R Elbow'] = calculate_angle(get_p_px(idx['r_shoulder']), get_p_px(idx['r_elbow']), get_p_px(idx['r_wrist']))
                if options_dict['show_l_foot']:
                    angles['L Knee'] = calculate_angle(get_p_px(idx['l_hip']), get_p_px(idx['l_knee']), get_p_px(idx['l_ankle']))
                if options_dict['show_r_foot']:
                    angles['R Knee'] = calculate_angle(get_p_px(idx['r_hip']), get_p_px(idx['r_knee']), get_p_px(idx['r_ankle']))

                def draw_traj(points, color, label, enabled):
                    if not enabled or len(points) == 0: return
                    if options_dict['show_start']:
                        start_pt = (int(points[0][0]*width), int(points[0][1]*height))
                        cv2.circle(image, start_pt, 8, (0, 0, 255), -1) 
                        cv2.circle(image, start_pt, 9, (0, 0, 0), 1)
                    curr_pt = (int(points[-1][0]*width), int(points[-1][1]*height))
                    cv2.putText(image, label, (curr_pt[0]+15, curr_pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    for i in range(1, len(points)):
                        cv2.line(image, (int(points[i-1][0]*width), int(points[i-1][1]*height)), 
                                 (int(points[i][0]*width), int(points[i][1]*height)), color, 2)

                draw_traj(trajectories['pelvis'], colors['pelvis'], "Pelvis", options_dict['show_pelvis'])
                draw_traj(trajectories['l_hand'], colors['l_hand'], "L-Hand", options_dict['show_l_hand'])
                draw_traj(trajectories['r_hand'], colors['r_hand'], "R-Hand", options_dict['show_r_hand'])
                draw_traj(trajectories['l_foot'], colors['l_foot'], "L-Foot", options_dict['show_l_foot'])
                draw_traj(trajectories['r_foot'], colors['r_foot'], "R-Foot", options_dict['show_r_foot'])
                draw_traj(trajectories['l_elbow'], colors['l_elbow'], "L-Elbow", options_dict['show_l_elbow'])
                draw_traj(trajectories['r_elbow'], colors['r_elbow'], "R-Elbow", options_dict['show_r_elbow'])
                draw_traj(trajectories['l_knee'], colors['l_knee'], "L-Knee", options_dict['show_l_knee'])
                draw_traj(trajectories['r_knee'], colors['r_knee'], "R-Knee", options_dict['show_r_knee'])
                draw_traj(trajectories['com'], colors['com'], "CoM", options_dict['show_com'])

                if len(trajectories['com']) > 0:
                    com_px = (int(trajectories['com'][-1][0]*width), int(trajectories['com'][-1][1]*height))
                    cv2.circle(image, com_px, 12, (255, 255, 255), -1)
                    cv2.circle(image, com_px, 14, (0, 0, 0), 2)
                
                info_y = 50
                cv2.putText(image, f"Velocity: {velocity:.2f} n/s", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                for i, (k, v) in enumerate(angles.items()):
                    cv2.putText(image, f"{k}: {v:.1f} deg", (20, info_y + 45*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height))
            out.write(image)
            frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-pelvis", action="store_false", dest="show_pelvis")
    parser.add_argument("--no-l-hand", action="store_false", dest="show_l_hand")
    parser.add_argument("--no-r-hand", action="store_false", dest="show_r_hand")
    parser.add_argument("--no-l-foot", action="store_false", dest="show_l_foot")
    parser.add_argument("--no-r-foot", action="store_false", dest="show_r_foot")
    parser.add_argument("--no-l-elbow", action="store_false", dest="show_l_elbow")
    parser.add_argument("--no-r-elbow", action="store_false", dest="show_r_elbow")
    parser.add_argument("--no-l-knee", action="store_false", dest="show_l_knee")
    parser.add_argument("--no-r-knee", action="store_false", dest="show_r_knee")
    parser.add_argument("--no-com", action="store_false", dest="show_com")
    parser.add_argument("--no-trajectories", action="store_false", dest="show_all_trajs")
    parser.add_argument("--traj-len", type=int, default=1000)
    parser.add_argument("--no-start", action="store_false", dest="show_start")
    parser.add_argument("--mono-color", action="store_true")
    
    args = parser.parse_args()
    options_dict = vars(args)
    if not args.show_all_trajs:
        for k in ['show_pelvis', 'show_l_hand', 'show_r_hand', 'show_l_foot', 'show_r_foot', 
                  'show_l_elbow', 'show_r_elbow', 'show_l_knee', 'show_r_knee', 'show_com']:
            options_dict[k] = False

    model_path = download_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options_base = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    for video_path in video_files:
        process_video(video_path, args.output_dir, options_base, options_dict)
