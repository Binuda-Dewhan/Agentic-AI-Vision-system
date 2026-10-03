import sys
import cv2

points = []

def click_event(event, x, y, flags, params):
    # On left mouse click, record the point and draw it
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point recorded: ({x}, {y})")
        
        # Draw a circle at the clicked point
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # Draw a line connecting to the previous point
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)
            
        # If 4 points are clicked, connect the last point to the first to close the polygon
        if len(points) == 4:
            cv2.line(img, points[-1], points[0], (255, 0, 0), 2)
            
        cv2.imshow(window_name, img)

if len(sys.argv) < 2:
    print("Usage: python get_bed_polygon.py <path_to_video>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    sys.exit(1)

ret, img = cap.read()
cap.release()

if not ret:
    print("Error: Could not read the first frame of the video.")
    sys.exit(1)

print("=" * 60)
print("INSTRUCTIONS:")
print("1. A window will open showing the first frame of your video.")
print("2. Click the 4 corners of your bed in order (e.g., top-left, top-right, bottom-right, bottom-left).")
print("3. Press the 'q' key to close the window and generate your config.")
print("=" * 60)

window_name = 'Click 4 corners of the bed (Press Q when done)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# Set a reasonable initial size that fits on most screens
cv2.resizeWindow(window_name, 1280, 720)

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)

# Wait until 'q' is pressed
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

if len(points) > 0:
    print("\n" + "=" * 60)
    print("✅ Done! Add the following to your config.yaml file:")
    print("=" * 60)
    print("bed_region:")
    print("  bed_region_polygon:")
    for pt in points:
        print(f"    - [{pt[0]}, {pt[1]}]")
    print("=" * 60)
else:
    print("\nNo points were selected.")
