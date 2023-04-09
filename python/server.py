from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import util

app = Flask(__name__)
app.url = '../sample/video/road.hevc'

def gen_frames():  # generate frame by frame from camera

    # TODO: get number of frames from video before iterate
    # temporary can just set a limit
    limit_frame = 500
    num_frame = 0

    rgb_frames = np.zeros((limit_frame, util.plot_img_height, util.plot_img_width, 3), dtype=np.uint8)
    yuv_frames = np.zeros((limit_frame + 1, util.FULL_FRAME_SIZE[1]*3//2, util.FULL_FRAME_SIZE[0]), dtype=np.uint8)
    stacked_frames = np.zeros((limit_frame, 12, 128, 256), dtype=np.uint8)
    recurrent_state = np.zeros((1, 512), dtype=np.float32)
    lanelines_preds = []
    road_edges_preds = []
    path_preds = []
    fill_color_gt = [0,  255, 0]
    line_color_gt = [255,255, 0]
    fill_color_preds = [0,  0,255]
    line_color_preds = [200,0,255]
    laneline_colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]


    rpy_calib_pred = np.array([0.00018335809, 0.034165092, -0.014245722]) / 2 
    calibration = util.Calibration(rpy_calib_pred, plot_img_width=util.plot_img_width, plot_img_height=util.plot_img_height)

    cap = cv2.VideoCapture(app.url)
    success, frame = cap.read()
    if not success:
        cap = cv2.VideoCapture('../sample/video/road.hevc')
        
    model, run_model = util.load_inference_model('../model/supercombo.onnx')
  

    while(cap.isOpened()):
        ret, frame = cap.read()

        if num_frame >  limit_frame:
            break

        if ret == True:
            
            frame = cv2.resize(frame, util.FULL_FRAME_SIZE, interpolation = cv2.INTER_AREA)
            yuv_frame = util.bgr_to_yuv(frame)
            rgb_frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yuv_frames[num_frame] = yuv_frame
            rgb_frames[num_frame] = util.create_image_canvas(rgb_frame, util.CALIB_BB_TO_FULL, util.plot_img_height, util.plot_img_width)

            prepared_frames = util.transform_frames(yuv_frames)

            stacked_frames[num_frame] = np.vstack(prepared_frames[num_frame:num_frame+2])[None].reshape(12, 128, 256)

            inputs = {
                    'input_imgs': stacked_frames[num_frame:num_frame+1].astype(np.float32),
                    'desire': np.zeros((1, 8), dtype=np.float32),
                    'traffic_convention': np.array([0, 1], dtype=np.float32).reshape(1, 2),
                    'initial_state': recurrent_state,
                }
            
            outs, recurrent_state = run_model(inputs)
            lanelines, road_edges, best_path = util.extract_preds(outs)[0]

            lanelines_preds.append(lanelines)
            road_edges_preds.append(road_edges)
            path_preds.append(best_path)

            img_plot = util.draw_path(lanelines_preds[num_frame], road_edges_preds[num_frame], path_preds[num_frame][0, :, :3], rgb_frames[num_frame], calibration, laneline_colors, fill_color=fill_color_preds, line_color=line_color_preds)    

            img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
            
            ret, buffer = cv2.imencode('.jpg', img_plot)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')  # concat frame one by one and show result
            
            num_frame += 1

        else:
            break

@app.route('/', methods=['GET'])
def index():
    """Video streaming home page."""
    if request.args.get("url") is not None:
        app.url = request.args.get("url")

    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)