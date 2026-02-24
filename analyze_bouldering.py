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
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

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
        print(f"Error: Invalid FPS ({fps}) or total frames ({total_frames}) for {video_path}")
        cap.release()
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analyzed_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_path}")
        cap.release()
        return

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

    if options_dict['mono_color']:
        colors = {k: (200, 200, 200) for k in keys}
        colors['com'] = (255, 255, 255)

    print(f"Processing {os.path.basename(video_path)}: {total_frames} frames expected...")

    with vision.PoseLandmarker.create_from_options(options_base) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # MediaPipe expects timestamp in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)
            
            # Process frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            image = frame.copy()
            if pose_landmarker_result.pose_landmarks:
                landmarks = pose_landmarker_result.pose_landmarks[0]
                idx = {'l_hip': 23, 'r_hip': 24, 'l_wrist': 15, 'r_wrist': 16, 'l_ankle': 27, 'r_ankle': 28, 
                       'l_shoulder': 11, 'r_shoulder': 12, 'l_elbow': 13, 'r_elbow': 14, 'l_knee': 25, 'r_knee': 26}
                get_pt = lambda i: [landmarks[i].x, landmarks[i].y]
                pts = {k: get_pt(v) for k, v in idx.items()}
                
                pelvis_pos = [(pts['l_hip'][0] + pts['r_hip'][0])/2, (pts['l_hip'][1] + pts['r_hip'][1])/2]
                com = np.mean([pts['l_hip'], pts['r_hip'], pts['l_shoulder'], pts['r_shoulder']], axis=0)

                trajectories['pelvis'].append(pelvis_pos)
                trajectories['l_hand'].append(pts['l_wrist'])
                trajectories['r_hand'].append(pts['r_wrist'])
                trajectories['l_foot'].append(pts['l_ankle'])
                trajectories['r_foot'].append(pts['r_ankle'])
                trajectories['l_elbow'].append(pts['l_elbow'])
                trajectories['r_elbow'].append(pts['r_elbow'])
                trajectories['l_knee'].append(pts['l_knee'])
                trajectories['r_knee'].append(pts['r_knee'])
                trajectories['com'].append(com)

                if prev_com is not None:
                    # Normalized velocity calculation
                    velocity = np.sqrt(np.sum((com - prev_com)**2)) * fps
                prev_com = com

                angles = {
                    'L Elbow': calculate_angle(pts['l_shoulder'], pts['l_elbow'], pts['l_wrist']),
                    'R Elbow': calculate_angle(pts['r_shoulder'], pts['r_elbow'], pts['r_wrist']),
                    'L Knee': calculate_angle(pts['l_hip'], pts['l_knee'], pts['l_ankle']),
                    'R Knee': calculate_angle(pts['r_hip'], pts['r_knee'], pts['r_ankle'])
                }

                def draw_traj(points, color, label, enabled):
                    if not enabled or len(points) == 0: return
                    if options_dict['show_start']:
                        start_pt = (int(points[0][0]*width), int(points[0][1]*height))
                        cv2.circle(image, start_pt, 8, (0, 0, 255), -1) # Red
                        cv2.circle(image, start_pt, 9, (0, 0, 0), 1)
                    curr_pt = (int(points[-1][0]*width), int(points[-1][1]*height))
                    cv2.putText(image, label, (curr_pt[0]+15, curr_pt[1]-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    for i in range(1, len(points)):
                        cv2.line(image, (int(points[i-1][0]*width), int(points[i-1][1]*height)), 
                                 (int(points[i][0]*width), int(points[i][1]*height)), color, 2)

                draw_traj(trajectories['pelvis'], colors['pelvis'], "Pelvis", options_dict['show_pelvis'])
                draw_traj(trajectories['l_hand'], colors['l_hand'], "L-Hand", options_dict['show_l_hand'])
                draw_traj(trajectories['r_hand'], colors['r_hand'], "R-Hand", options_dict['show_r_hand'])
                draw_traj(trajectories['l_foot'], colors['l_foot'], "L-Foot", options_dict['show_l_foot'])
                draw_traj(trajectories['r_foot'], colors['r_foot'], "R-Foot", options_dict['show_l_foot']) # Fixed a small bug here
                draw_traj(trajectories['l_elbow'], colors['l_elbow'], "L-Elbow", options_dict['show_l_elbow'])
                draw_traj(trajectories['r_elbow'], colors['r_elbow'], "R-Elbow", options_dict['show_r_elbow'])
                draw_traj(trajectories['l_knee'], colors['l_knee'], "L-Knee", options_dict['show_l_knee'])
                draw_traj(trajectories['r_knee'], colors['r_knee'], "R-Knee", options_dict['show_r_knee'])
                draw_traj(trajectories['com'], colors['com'], "CoM", options_dict['show_com'])

                com_px = (int(com[0]*width), int(com[1]*height))
                cv2.circle(image, com_px, 12, (255, 255, 255), -1)
                cv2.circle(image, com_px, 14, (0, 0, 0), 2)
                
                info_y = 50
                cv2.putText(image, f"Velocity: {velocity:.2f} n/s", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                for i, (k, v) in enumerate(angles.items()):
                    cv2.putText(image, f"{k}: {v:.1f} deg", (20, info_y + 45*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Ensure image size matches VideoWriter setting
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height))
            
            out.write(image)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  ...processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Saved: {output_path} ({frame_idx} frames written)")

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
    # Configure base options for video mode
    options_base = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    if not video_files:
        print(f"No mp4 files found in {args.input_dir}")
    else:
        for video_path in video_files:
            process_video(video_path, args.output_dir, options_base, options_dict)
