import cv2
from ultralytics import YOLO

# --- LOAD YOUR TRAINED MODEL ---
# Make sure the path to your best.pt file is correct
try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- OPEN YOUR PC'S WEBCAM ---
# 0 is usually the default webcam. If you have multiple cameras, you might need to change it to 1, 2, etc.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

# --- PROCESS THE WEBCAM FEED ---
while True:
    # Read one frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    # --- RUN THE MODEL ON THE FRAME ---
    # The model will detect objects and return the results
    results = model(frame)

    # Get the frame with the detection boxes and labels drawn on it
    annotated_frame = results[0].plot()

    # --- SHOW THE VIDEO FEED ---
    # A new window will pop up to display the live, annotated video
    cv2.imshow("YOLOv11 Live Detection", annotated_frame)

    # --- PRESS 'q' TO QUIT ---
    # Wait for 1 millisecond, and if the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEAN UP ---
# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
