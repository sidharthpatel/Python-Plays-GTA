import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D

def process_img(original_image):
    # Convert an image to GrayScale
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=200)

    return processed_img

# Establish an initial Countdown of 4 3 2 1
for i in list(range(4)) [::-1]:
    print(i+1)
    time.sleep(1)

# Record the current time
last_time = time.time()

while(True):
    # Capture the screen ratio (0, 40) to (800, 600)
    # Captures GTA
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 600)))
    new_screen = process_img(screen)

    print('down')
    PressKey(W)
    time.sleep(3)
    print('Up')
    ReleaseKey(W)

    # Return the time it took to loop through per image
    print('Loop took {} seconds'.format(time.time() - last_time))

    # Display the processed image with edge detection
    cv2.imshow('window', new_screen)

    # Show the captured frames using OpenCV
    # cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

    # Kill OpenCV if `q` is pressed
    # WaitKey(x) -> wait for x milliseconds and check if the q key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break