import cv2
import numpy as np

#open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ------------------ red ------------------
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # ------------------ green------------------
    lower_green = np.array([36, 50, 70])
    upper_green = np.array([89, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # ------------------ blue ------------------
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ------------------ yellow ------------------
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


   
    def detect_color(mask, color_name, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, color_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

    detect_color(red_mask, "Red", (0, 0, 255))
    detect_color(green_mask, "Green", (0, 255, 0))
    detect_color(blue_mask, "Blue", (255, 0, 0))
    detect_color(yellow_mask, "Yellow", (0, 255, 255))

    cv2.imshow("Color Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
