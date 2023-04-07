import numpy as np
import cv2
import math
import laika.orientation as orientation

plot_img_width=640
plot_img_height=480

FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
zoom = FULL_FRAME_SIZE[0] / plot_img_width
eon_focal_length = FOCAL = 910.0

zoom = FULL_FRAME_SIZE[0] / plot_img_width
CALIB_BB_TO_FULL = np.asarray([
    [zoom, 0., 0.],
    [0., zoom, 0.],
    [0., 0., 1.]])
    
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
    [0.,  FOCAL,  H/2.],
    [0.,    0.,     1.]])


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

#https://github.com/commaai/openpilot/blob/master/common/transformations/camera.py#L104
# latest version doesn't have this fuction, but it's used in v0.7
def normalize(img_pts, intrinsics=eon_intrinsics):
  # normalizes image coordinates
  # accepts single pt or array of pts
  intrinsics_inv = np.linalg.inv(intrinsics)
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_normalized = img_pts.dot(intrinsics_inv.T)
  img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
  return img_pts_normalized[:,:2].reshape(input_shape)


# https://github.com/commaai/openpilot/blob/v0.7/common/transformations/camera.py
# latest version doesn't have this fuction, but it's used in v0.7
def transform_img(base_img,
                 augment_trans=np.array([0,0,0]),
                 augment_eulers=np.array([0,0,0]),
                 from_intr=eon_intrinsics,
                 to_intr=eon_intrinsics,
                 output_size=None,
                 pretransform=None,
                 top_hacks=False,
                 yuv=False,
                 alpha=1.0,
                 beta=0,
                 blur=0):
  import cv2  # pylint: disable=import-error
  cv2.setNumThreads(1)

  if yuv:
    base_img = cv2.cvtColor(base_img, cv2.COLOR_YUV2RGB_I420)

  size = base_img.shape[:2]
  if not output_size:
    output_size = size[::-1]

  cy = from_intr[1,2]
  def get_M(h=1.22):
    quadrangle = np.array([[0, cy + 20],
                           [size[1]-1, cy + 20],
                           [0, size[0]-1],
                           [size[1]-1, size[0]-1]], dtype=np.float32)
    quadrangle_norm = np.hstack((normalize(quadrangle, intrinsics=from_intr), np.ones((4,1))))
    quadrangle_world = np.column_stack((h*quadrangle_norm[:,0]/quadrangle_norm[:,1],
                                        h*np.ones(4),
                                        h/quadrangle_norm[:,1]))
    rot = orientation.rot_from_euler(augment_eulers)
    to_extrinsics = np.hstack((rot.T, -augment_trans[:,None]))
    to_KE = to_intr.dot(to_extrinsics)
    warped_quadrangle_full = np.einsum('jk,ik->ij', to_KE, np.hstack((quadrangle_world, np.ones((4,1)))))
    warped_quadrangle = np.column_stack((warped_quadrangle_full[:,0]/warped_quadrangle_full[:,2],
                                         warped_quadrangle_full[:,1]/warped_quadrangle_full[:,2])).astype(np.float32)
    M = cv2.getPerspectiveTransform(quadrangle, warped_quadrangle.astype(np.float32))
    return M

  M = get_M()
  if pretransform is not None:
    M = M.dot(pretransform)
  augmented_rgb = cv2.warpPerspective(base_img, M, output_size, borderMode=cv2.BORDER_REPLICATE)

  if top_hacks:
    cyy = int(math.ceil(to_intr[1,2]))
    M = get_M(1000)
    if pretransform is not None:
      M = M.dot(pretransform)
    augmented_rgb[:cyy] = cv2.warpPerspective(base_img, M, (output_size[0], cyy), borderMode=cv2.BORDER_REPLICATE)

  # brightness and contrast augment
  augmented_rgb = np.clip((float(alpha)*augmented_rgb + beta), 0, 255).astype(np.uint8)

  # gaussian blur
  if blur > 0:
    augmented_rgb = cv2.GaussianBlur(augmented_rgb,(blur*2+1,blur*2+1),cv2.BORDER_DEFAULT)

  if yuv:
    augmented_img = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2YUV_I420)
  else:
    augmented_img = augmented_rgb
  return augmented_img



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

def create_image_canvas(img_rgb, zoom_matrix, plot_img_height, plot_img_width):
    '''Transform with a correct warp/zoom transformation.'''
    img_plot = np.zeros((plot_img_height, plot_img_width, 3), dtype='uint8')
    cv2.warpAffine(img_rgb, zoom_matrix[:2], (img_plot.shape[1], img_plot.shape[0]), dst=img_plot, flags=cv2.WARP_INVERSE_MAP)
    return img_plot

def bgr_to_yuv(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    assert yuv.shape == ((874*3//2, 1164))
    return yuv