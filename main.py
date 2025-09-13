from utils import read_video, save_video
from trackers import Tracker
import cv2
import webcolors
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

# Global variables for tracking selections
selected_bet = None
selected_amount = None
bet_confirmed = False

# Coordinates for clickable areas
option_coords = []
amount_coords = []
confirm_coords = ()
new_bet_coords = ()
dashboard_visible = True
toggle_button_coords = ()  # x1, y1, x2, y2

# --- Mouse callback ---
def mouse_callback(event, x, y, flags, param):
    global selected_bet, selected_amount, bet_confirmed, dashboard_visible
    betting_options, bet_amount_options = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # Toggle dashboard button
        x1, y1, x2, y2 = toggle_button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            dashboard_visible = not dashboard_visible
            return  # skip other clicks
        if not dashboard_visible:
            return  # disable clicks when dashboard hidden
        # --- Existing option click logic ---
        for idx, (ox1, oy1, ox2, oy2) in enumerate(option_coords):
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                selected_bet = f"{list(betting_options.items())[idx][0]} {list(betting_options.items())[idx][1]}"
        for idx, (ax1, ay1, ax2, ay2) in enumerate(amount_coords):
            if ax1 <= x <= ax2 and ay1 <= y <= ay2:
                selected_amount = bet_amount_options[idx]
        cx1, cy1, cx2, cy2 = confirm_coords
        if cx1 <= x <= cx2 and cy1 <= y <= cy2 and selected_bet and selected_amount:
            bet_confirmed = True
            print(f"New Bet Created: {selected_bet} Amount: {selected_amount}")
        nx1, ny1, nx2, ny2 = new_bet_coords
        if nx1 <= x <= nx2 and ny1 <= y <= ny2:
            selected_bet = None
            selected_amount = None
            bet_confirmed = False
            print("Ready for a new bet!")

# --- Draw dashboard ---
def draw_betting_dashboard(frame, team_ball_control, frame_idx, betting_options, bet_amount_options):
    global toggle_button_coords, dashboard_visible
    h, w, _ = frame.shape
    sidebar_w = 600
    overlay = frame.copy()

    # --- Sidebar ---
    if dashboard_visible:
        cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # --- Toggle Button ---
    button_h = 60
    toggle_button_coords = (w - sidebar_w, 0, w, button_h)
    cv2.rectangle(frame, (toggle_button_coords[0], toggle_button_coords[1]),
                  (toggle_button_coords[2], toggle_button_coords[3]), (0, 140, 0), -1)
    symbol = "▼" if dashboard_visible else "▲"
    cv2.putText(frame, "BETTING DASHBOARD", (toggle_button_coords[0]+20, toggle_button_coords[1]+40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, symbol, (toggle_button_coords[2]-50, toggle_button_coords[1]+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if not dashboard_visible:
        return frame

    # --- Dashboard contents ---
    y_cursor = button_h + 20
    section_spacing = 50  # Space between main sections
    box_height = 50
    box_radius = 10
    inter_item_spacing = 15 

    def draw_rounded_box(img, top_left, bottom_right, color, radius=10, thickness=-1):
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
        cv2.circle(img, (x1+radius, y1+radius), radius, color, thickness)
        cv2.circle(img, (x2-radius, y1+radius), radius, color, thickness)
        cv2.circle(img, (x1+radius, y2-radius), radius, color, thickness)
        cv2.circle(img, (x2-radius, y2-radius), radius, color, thickness)

    # --- Selected Bet ---
    cv2.putText(frame, "Selected Bet:", (w - sidebar_w + 20, y_cursor + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    y_cursor += 30
    draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), (50, 50, 50), box_radius)
    cv2.putText(frame, selected_bet if selected_bet else "None", (w - sidebar_w + 25, y_cursor + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # --- Amount ---
    cv2.putText(frame, "Amount:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    y_cursor += 30
    draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), (50, 50, 50), box_radius)
    cv2.putText(frame, selected_amount if selected_amount else "None", (w - sidebar_w + 25, y_cursor + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    y_cursor += box_height + section_spacing

    # --- Options ---
    cv2.putText(frame, "Options:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    y_cursor += 30
    option_coords = []
    for i, (k, v) in enumerate(betting_options.items(), start=1):
        color = (80, 80, 80)
        if selected_bet and f"{k} {v}" == selected_bet:
            color = (0, 120, 255)  # Highlight selected
        draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), color, box_radius)
        cv2.putText(frame, f"{i}. {k} {v}", (w - sidebar_w + 25, y_cursor + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        option_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing  # smaller spacing between options

    y_cursor += section_spacing // 2  # extra space before amount options

    # --- Amount Options ---
    cv2.putText(frame, "Select Amount:", (w - sidebar_w + 20, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    y_cursor += 30
    amount_coords = []
    for i, amt in enumerate(bet_amount_options):
        color = (80, 80, 80)
        if selected_amount == amt:
            color = (0, 120, 255)
        draw_rounded_box(frame, (w - sidebar_w + 15, y_cursor), (w - 20, y_cursor + box_height), color, box_radius)
        cv2.putText(frame, f"{i+1}. {amt}", (w - sidebar_w + 25, y_cursor + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        amount_coords.append((w - sidebar_w + 15, y_cursor, w - 20, y_cursor + box_height))
        y_cursor += box_height + inter_item_spacing

    # --- Confirm & New Bet ---
    confirm_y = y_cursor + 20
    global confirm_coords, new_bet_coords
    confirm_coords = (w - sidebar_w + 50, confirm_y, w - sidebar_w + 250, confirm_y + box_height)
    new_bet_coords = (w - sidebar_w + 270, confirm_y, w - sidebar_w + 470, confirm_y + box_height)

    draw_rounded_box(frame, (confirm_coords[0], confirm_coords[1]), (confirm_coords[2], confirm_coords[3]),
                     (0, 255, 0) if not bet_confirmed else (0, 150, 0), box_radius)
    cv2.putText(frame, "CONFIRM", (confirm_coords[0]+25, confirm_coords[1]+35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

    draw_rounded_box(frame, (new_bet_coords[0], new_bet_coords[1]), (new_bet_coords[2], new_bet_coords[3]), (255, 140, 0), box_radius)
    cv2.putText(frame, "NEW BET", (new_bet_coords[0]+25, new_bet_coords[1]+35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

    return frame

# --- Helper function to get closest color name ---
CSS3_NAMES_TO_HEX = {
    'Black': '#000000', 'White': '#FFFFFF', 'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0000FF',
    'Yellow': '#FFFF00', 'Cyan': '#00FFFF', 'Magenta': '#FF00FF', 'Gray': '#808080', 'Orange': '#FFA500',
    'Pink': '#FFC0CB', 'Brown': '#A52A2A', 'Purple': '#800080', 'Lime': '#00FF00', 'Navy': '#000080'
}

def get_closest_color_name(rgb_tuple):
    min_dist = float("inf")
    closest_name = None
    for name, hex_val in CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
        dist = (r_c - rgb_tuple[0])**2 + (g_c - rgb_tuple[1])**2 + (b_c - rgb_tuple[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def main():
    global selected_bet, selected_amount, bet_confirmed
    video_frames = read_video('input_videos/game.mp4')
    tracker = Tracker('yolov8n.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    # --- Assign team colors ---
    frame = video_frames[0]
    player_detections = tracks['players'][0]
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame, player_detections)
    team_colors = team_assigner.get_team_colors()
    team1_color = [int(c) for c in team_colors[1]]
    team2_color = [int(c) for c in team_colors[2]]
    # Convert to friendly names
    team1_label = f"Team {get_closest_color_name(tuple(team1_color))}"
    team2_label = f"Team {get_closest_color_name(tuple(team2_color))}"

    # Update betting options
    betting_options = {
        "Player 7": "to assist",
        "Player 10": "to assist",
        "Player 11": "to cover most distance",
        team1_label: "to win possession",
        team2_label: "to make most passes"
    }
    bet_amount_options = ["$10", "$20", "$50", "$100"]

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed & distance
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign player teams
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else "Unknown")
    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # --- Interactive GUI playback ---
    cv2.namedWindow("Football Analysis", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Football Analysis", mouse_callback, param=(betting_options, bet_amount_options))
    frame_idx = 0
    while frame_idx < len(output_video_frames):
        frame = output_video_frames[frame_idx]
        frame_with_dashboard = draw_betting_dashboard(frame, team_ball_control, frame_idx, betting_options, bet_amount_options)
        cv2.imshow("Football Analysis", frame_with_dashboard)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        frame_idx += 1
    cv2.destroyAllWindows()

    # Save video with dashboard overlay
    output_video_frames = [
        draw_betting_dashboard(f, team_ball_control, idx, betting_options, bet_amount_options)
        for idx, f in enumerate(output_video_frames)
    ]
    save_video(output_video_frames, 'output_videos/output_video_with_dashboard.mp4')

    # Print final bet if confirmed
    if bet_confirmed:
        print(f"Final Bet: {selected_bet} Amount: {selected_amount}")
    else:
        print("No bet was confirmed.")

if __name__ == "__main__":
    main()