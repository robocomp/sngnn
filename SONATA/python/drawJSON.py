import sys
import numpy as np
import cv2 as cv
import json

image_size = 1024
data = []
ids_dict = {}


def print_usage():
    print("Please introduce the name of the json file")
    print("Example: python3 ", sys.argv[0], " jsons_file")


# Load json file
def load_json_frame(i):
    global data
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print_usage()
        sys.exit(0)

    if len(data) == 0:
        with open(sys.argv[1]) as json_file:
            data = json.load(json_file)
    elif i == len(data):
        print("All frames showed.")
        print("Exiting")
        cv.destroyAllWindows()
        sys.exit(0)

    return data[i]


def print_scene(frame):
    # Scale and rotation matrix
    coor_sifht = int(image_size/2)
    scale = 100

    # Create white background
    img = np.ones((image_size, image_size, 3), np.uint8) * 255

    print("LABEL", frame["command"])

    # Print walls
    for wall in frame["walls"]:
        x1 = int(coor_sifht + (wall["x1"] * scale))
        y1 = int(coor_sifht - (wall["y1"] * scale))
        x2 = int(coor_sifht + (wall["x2"] * scale))
        y2 = int(coor_sifht - (wall["y2"] * scale))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)

    # Print people
    for human in frame["people"]:
        idx = human["id"]
        x = int(coor_sifht + (human["x"] * scale))
        y = int(coor_sifht - (human["y"] * scale))
        angle = (180/np.pi) * (human["a"]+(np.pi/2))  # Negative because right hand rule is used
        # Body
        cv.ellipse(img, (x, y), (30, 15), angle, 0, 360, (255, 0, 0), -1)
        # Right eye
        r_eye_angle = (0.6 + human["a"] - np.pi/2)
        r_eye_coor = [int(x + (np.cos(r_eye_angle)*22)), int(y + (np.sin(r_eye_angle)*22))]
        cv.circle(img, tuple(r_eye_coor), 5, (255, 255, 255), -1)
        cv.circle(img, tuple(r_eye_coor), 6, (0, 0, 0), 2)
        # Left eye
        l_eye_angle = (np.pi-0.6 + human["a"] - np.pi/2)
        l_eye_coor = [int(x + (np.cos(l_eye_angle )*22)), int(y + (np.sin(l_eye_angle )*22))]
        cv.circle(img, tuple(l_eye_coor), 5, (255, 255, 255), -1)
        cv.circle(img, tuple(l_eye_coor), 6, (0, 0, 0), 2)
        # Velocity vector
        vx = int(x - human["vx"]*scale)
        vy = int(y + human["vy"]*scale)
        cv.line(img, (x, y), (vx, vy), (0, 0, 255), 2)

        ids_dict[str(idx)] = ("h", tuple(r_eye_coor), tuple(l_eye_coor))

    # Print objects
    for obj in frame["objects"]:
        idx = obj["id"]
        x = int(coor_sifht + (obj["x"] * scale))
        y = int(coor_sifht - (obj["y"] * scale))
        size_x = int(obj["size_x"] * scale)
        size_y = int(obj["size_y"] * scale)
        angle_o = (180/np.pi) * (obj["a"])
        rect = ((x, y), (size_x, size_y), angle_o)
        box = cv.boxPoints(rect)
        box = np.array(box).astype(np.int32)
        cv.drawContours(img, [box], 0, (0, 255, 0), 5, cv.FILLED)
        # end_line = (int((box[2][0]+box[3][0])/2), int((box[2][1]+box[3][1])/2))
        # Velocity vector
        vx = int(x - obj["vx"] * scale)
        vy = int(y + obj["vy"] * scale)
        cv.line(img, (x, y), (vx, vy), (0, 0, 255), 2)

        ids_dict[str(idx)] = ("o", (x, y))

    # Print robot
    pt1 = (coor_sifht - 20, coor_sifht + 20)
    pt2 = (coor_sifht + 20, coor_sifht + 20)
    pt3 = (coor_sifht, coor_sifht - 20)

    cv.circle(img, pt1, 2, (0, 0, 255), -1)
    cv.circle(img, pt2, 2, (0, 0, 255), -1)
    cv.circle(img, pt3, 2, (0, 0, 255), -1)

    triangle_cnt = np.array([pt1, pt2, pt3])
    cv.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

    #Print relations
    for i in frame["interaction"]:
        dist = str(i["dst"])
        src = str(i["src"])
        if ids_dict[dist][0] is "h" and ids_dict[src][0] is "o":
            cv.line(img, ids_dict[dist][1], ids_dict[src][1], (0, 0, 0), 2)
            cv.line(img, ids_dict[dist][2], ids_dict[src][1], (0, 0, 0), 2)
        elif ids_dict[dist][0] is "h" and ids_dict[src][0] is "h":
            cv.line(img, ids_dict[dist][1], ids_dict[src][2], (0, 0, 0), 2)
            cv.line(img, ids_dict[dist][2], ids_dict[src][1], (0, 0, 0), 2)
        elif ids_dict[dist][0] is "o" and ids_dict[src][0] is "h":
            cv.line(img, ids_dict[dist][1], ids_dict[src][1], (0, 0, 0), 2)
            cv.line(img, ids_dict[dist][1], ids_dict[src][2], (0, 0, 0), 2)

    # Print time stamp
    font = cv.FONT_HERSHEY_SIMPLEX
    text = "Room at time: " + str(frame["timestamp"])
    cv.putText(img, text, (10, image_size-45), font, 1.5, (30, 78, 50), 2, cv.LINE_AA)

    # Print goal
    x_g = int(coor_sifht + (frame["goal"][0]["x"] * scale))
    y_g = int(coor_sifht - (frame["goal"][0]["y"] * scale))

    cv.line(img, (x_g - 10, y_g - 10), (x_g + 10, y_g + 10), (0, 0, 255), 8)
    cv.line(img, (x_g - 10, y_g + 10), (x_g + 10, y_g - 10), (0, 0, 255), 8)

    # Display final image
    cv.imshow("Display of the room", img)

frame_number = 0
frame = load_json_frame(frame_number)
print_scene(frame)
while True: #cv.getWindowProperty('Display of the room', cv.WND_PROP_VISIBLE) >= 1:
    k = cv.waitKey(20)
    if k == 27:  # Esc
        print()
        cv.destroyAllWindows()
        break
    elif k == 83 or k == 13:  # RightKey or enter
        frame_number += 1
        frame = load_json_frame(frame_number)
        print_scene(frame)
    elif k == 81 or k == 8:  # LeftKey or return
        frame_number = max(0, frame_number-1)
        frame = load_json_frame(frame_number)
        print_scene(frame)

    elif k == -1:
        continue
    else:
        print("Escape for closing the window and enter or right key for loading next frame")
