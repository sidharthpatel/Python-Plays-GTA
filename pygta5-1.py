import numpy as np
from PIL import ImageGrab
import cv2
import time

# Record the current time
last_time = time.time()

while(True):
    # Capture the screen ratio (0, 40) to (800, 600)
    # Captures GTA
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 600)))

    # Return the time it took to loop through per image
    print('Loop took {} seconds'.format(time.time() - last_time))

    # Show the captured frames using OpenCV
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    # Kill OpenCV if `q` is pressed
    # WaitKey(x) -> wait for x milliseconds and check if the q key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break