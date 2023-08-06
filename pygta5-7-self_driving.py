import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass

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

def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        print(str(e))





def process_img(original_image):
    # Convert an image to GrayScale
    # processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    processed_img = cv2.Canny(original_image, threshold1=150, threshold2=250)

    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0.5)

    # Vertices
    vertices = np.array([[0,600],[0,150],[800,150],[800,600],[600,400],[200,400]])

    processed_img = roi(processed_img, [vertices])

    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 5)
    # draw_lines(processed_img, lines)
    try:
        l1, l2 = draw_lanes(original_image, lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img, original_image

def main():
    # Record the current time
    last_time = time.time()

    while(True):
        # Capture the screen ratio (0, 40) to (800, 600)
        # Captures GTA
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 600)))
        new_screen, original_image = process_img(screen)

        # Return the time it took to loop through per image
        print('Loop took {} seconds'.format(time.time() - last_time))

        # Display the processed image with edge detection
        cv2.imshow('window', new_screen)

        # Show the captured frames using OpenCV
        cv2.imshow('window2', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        # Kill OpenCV if `q` is pressed
        # WaitKey(x) -> wait for x milliseconds and check if the q key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()