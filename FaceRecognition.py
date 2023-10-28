import cv2
import threading

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

match = False

# Matching function
#threshold = .55 works best when using only one image
def is_image_match(imgVideo, reference_image, threshold=0.55):
    global match

    #convert to grayscale
    live_frame_gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    #template matching
    result = cv2.matchTemplate(live_frame_gray, reference_image_gray, cv2.TM_CCOEFF_NORMED)

    #best match location
    _, max_val, _, _ = cv2.minMaxLoc(result)

    #check if the maximum correlation value is above the threshold
    if max_val >= threshold:
        match = True
    else:
        match = False

capture1 = cv2.VideoCapture(0)

count = 0

while True:
    reference_image = cv2.imread('PictureOfMe.jpeg')
    ret, imgVideo = capture1.read()

    if not ret:
        print("Error: Could not capture a frame.")
        break

    if count % 24 == 0:
        threading.Thread(target=is_image_match, args=(imgVideo, reference_image)).start()
        gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY)
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

    cv2.imshow('img', imgVideo)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture1.release()
cv2.destroyAllWindows()
