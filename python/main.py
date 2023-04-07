import cv2
import numpy as np
import util

# TODO: get number of frames from video before iterate
# temporary can just set a limit
limit_frame = 500
num_frame = 0

rgb_frames = np.zeros((limit_frame, util.plot_img_height, util.plot_img_width, 3), dtype=np.uint8)
yuv_frames = np.zeros((limit_frame + 1, util.FULL_FRAME_SIZE[1]*3//2, util.FULL_FRAME_SIZE[0]), dtype=np.uint8)
stacked_frames = np.zeros((limit_frame, 12, 128, 256), dtype=np.uint8)

cap = cv2.VideoCapture('../sample/video/road.hevc')

while(cap.isOpened()):
    ret, frame = cap.read()

    if num_frame >  limit_frame:
        break

    if ret == True:
       #print(num_frame)
        
        frame = cv2.resize(frame, util.FULL_FRAME_SIZE, interpolation = cv2.INTER_AREA)
        yuv_frame = util.bgr_to_yuv(frame)
        rgb_frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yuv_frames[num_frame] = yuv_frame
        rgb_frames[num_frame] = util.create_image_canvas(rgb_frame, util.CALIB_BB_TO_FULL, util.plot_img_height, util.plot_img_width)

        prepared_frames = util.transform_frames(yuv_frames)

        stacked_frames[num_frame] = np.vstack(prepared_frames[num_frame:num_frame+2])[None].reshape(12, 128, 256)

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        num_frame += 1

    else:
        break

