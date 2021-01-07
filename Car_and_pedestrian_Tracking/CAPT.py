import cv2

# Collect video
# video = cv2.VideoCapture('Tesla.mp4')
# video = cv2.VideoCapture('pedestrian.mp4')
video = cv2.VideoCapture('Pedestrians.mp4')


# Our pretrained classifeir 
classifier_file = 'cars.xml'
classifier_file_pd = 'haarcascade_fullbody.xml'

# Create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(classifier_file_pd)

# Run forver until closed
while True:
    
    # Read frame
    (read_successful, frame) = video.read()

    # SAfe code
    if read_successful:
        # Convert to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break 

    # Detect Cars
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscale_frame)

    # Draw the box around the car 
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+2), (x+w , y+w), (255, 0, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w , y+w), (0, 0, 255), 2)

    for (x,y,w,h) in pedestrian:
        cv2.rectangle(frame, (x,y), (x+w , y+w), (0, 255, 255), 2)
    
    # Display the image
    cv2.imshow('Car Tracker', frame)

    # Wait key to make box pause
    key = cv2.waitKey(1)

    # Quit the app
    if key==81 or key==113:
        break
# Release the video
video.release()

print("Code Ended")












