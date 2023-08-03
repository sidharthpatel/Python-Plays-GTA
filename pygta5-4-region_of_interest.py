import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
'''
    Region of Interest
    Given that the window size is (0,0) to (800,600)
    Consider vertices in process_img as your ROI

'''
def roi(img, vertices):
    # Creates an array of img size -> (800,600) and fills it with 0s
    mask = np.zeros_like(img)
    # Fills the area occupied by the vertices as 255
    cv2.fillPoly(mask, vertices, 255)
    # Retrieves only the ROI
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    # Convert an image to GrayScale
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    processed_img = cv2.Canny(processed_img, threshold1=150, threshold2=250)

    # Vertices
    vertices = np.array([[0,600],[0,150],[800,150],[800,600],[600,400],[200,400]])

    processed_img = roi(processed_img, [vertices])

    return processed_img

def main():
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

main()