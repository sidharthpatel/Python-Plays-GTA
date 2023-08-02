import numpy as np
from PIL import ImageGrab
import cv2
import time

def process_img(original_image):
    # Convert an image to GrayScale
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=200)
    return processed_img



# Record the current time
last_time = time.time()

while(True):
    # Capture the screen ratio (0, 40) to (800, 600)
    # Captures GTA
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 600)))
    new_screen = process_img(screen)

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