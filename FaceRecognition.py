import cv2
import threading

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the match variable
match = False

# Matching function
# Threshold = .55 works
def is_image_match(imgVideo, reference_image, threshold=0.55):
    global match

    # Convert the live frame and reference image to grayscale for template matching
    live_frame_gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(live_frame_gray, reference_image_gray, cv2.TM_CCOEFF_NORMED)

    # Find the best match location
    _, max_val, _, _ = cv2.minMaxLoc(result)

    # Check if the maximum correlation value is above the threshold
    if max_val >= threshold:
        match = True
    else:
        match = False

# Initialize the video capture
capture1 = cv2.VideoCapture(0)

count = 0

while True:
    reference_image = cv2.imread('PictureOfMe.jpeg')
    ret, imgVideo = capture1.read()

    if not ret:
        print("Error: Could not capture a frame.")
        break

    if count % 24 == 0:
        # Use threading to run the matching function concurrently
        threading.Thread(target=is_image_match, args=(imgVideo, reference_image)).start()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY)

        # Detect faces in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    count += 1

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(imgVideo, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = imgVideo[y:y + h, x:x + w]
        if match:
            cv2.putText(imgVideo, "Face is ME", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(imgVideo, "Face is NOT ME", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display the image in a window
    cv2.imshow('img', imgVideo)

    # Wait for the 'Esc' key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the video capture and close the window
capture1.release()
cv2.destroyAllWindows()
