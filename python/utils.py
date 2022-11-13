from common.transformations.camera import transform_img
import numpy as np

# similar to camera.py
FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0


# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_zoom = 1.
medmodel_intrinsics = np.array(
  [[ eon_focal_length / medmodel_zoom,    0. ,  0.5 * MEDMODEL_INPUT_SIZE[0]],
   [   0. ,  eon_focal_length / medmodel_zoom,  MEDMODEL_CY],
   [   0. ,                            0. ,   1.]])


# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array([
  [FOCAL,   0.,   W/2.],
  [  0.,  FOCAL,  H/2.],
  [  0.,    0.,     1.]])


def reshape_yuv(frames):
    H = (frames.shape[1]*2)//3
    W = frames.shape[2]
    in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

    in_img1[:, 0] = frames[:, 0:H:2, 0::2]
    in_img1[:, 1] = frames[:, 1:H:2, 0::2]
    in_img1[:, 2] = frames[:, 0:H:2, 1::2]
    in_img1[:, 3] = frames[:, 1:H:2, 1::2]
    in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return in_img1

def transform_frames(frames):
    imgs_med_model = np.zeros((len(frames), 384, 512), dtype=np.uint8)
    for i, img in enumerate(frames):
        imgs_med_model[i] = transform_img(img, 
                                          from_intr=eon_intrinsics,
                                          to_intr=medmodel_intrinsics, 
                                          yuv=True,
                                          output_size=(512, 256))

    reshaped = reshape_yuv(imgs_med_model)

    return reshaped
