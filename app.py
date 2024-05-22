
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import io
from PIL import Image
import torch
import sys
import os 
import numpy as np

# Add the path to the YOLOv5 directory to the module search path
yolov5_path = './yolov5'
sys.path.append(os.path.abspath(yolov5_path))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

server = app.server

# Define custom color palette and font styles
palette = {
    'background': '#f2f2f2',
    'text': '#333333',
    'primary': '#9932CC',
    'secondary': '#4B0082',
    'accent': '#BA55D3'
}

fonts = {
    'heading': 'Montserrat, sans-serif',
    'body': 'Open Sans, sans-serif'
}

app.layout = html.Div([
    html.Div([
        html.Img(
            src='/assets/logo.png',
            style={
                'height': '60px',
                'marginRight': '20px'
            }
        ),
        html.H1('Steel Defect Detection', style={'color': palette['primary'], 'margin': '0'})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '20px'}),
    
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '150px',
                'lineHeight': '150px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '10px',
                'textAlign': 'center',
                'margin': '20px 0',
                'backgroundColor': palette['background'],
                'color': palette['text'],
                'fontSize': '18px',
                'cursor': 'pointer',
                'transition': 'all 0.3s ease'
            },
            multiple=False
        ),
        html.Div(id='output-container', style={
            'textAlign': 'center',
            'marginTop': '20px',
        }, children=[
            html.Div(id='output-image-upload')
        ])
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}),
    
    html.Footer([
        html.P('© 2023 Steel Defect Detection. All rights reserved.', style={'margin': '0'}),
        html.P('Valcon', style={'margin': '5px 0'})
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': palette['secondary'], 'color': '#ffffff', 'fontSize': '14px'})
    
], style={'backgroundColor': palette['background'], 'fontFamily': fonts['body'], 'minHeight': '100vh', 'margin': '0'})

def detect_defect(image):
    # Load the YOLOv5 model
    weights_path = 'best.pt'
    model = DetectMultiBackend(weights_path, device='cpu', dnn=False, data='data/coco128.yaml', fp16=False)
    
    # Set model parameters
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45  # NMS IOU threshold
    
    # Convert PIL image to NumPy array
    img_np = np.array(image)
    
    # Check the number of channels in the image
    if len(img_np.shape) == 2:
        # Grayscale image, add a channel dimension and repeat it to create a 3-channel image
        img_np = np.repeat(np.expand_dims(img_np, axis=2), 3, axis=2)
    
    # Prepare the input image
    img_size = 640
    stride = 32
    img = letterbox(img_np, img_size, stride=stride)[0]
    
    # Convert the image to the expected format
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to('cpu')
    img = img.float()
    img /= 255.0
    
    # Perform inference
    pred = model(img, augment=False)
    pred = non_max_suppression(pred, model.conf, model.iou, classes=None, agnostic=False)
    
    # Process the detections
    for i, det in enumerate(pred):
        annotator = Annotator(img_np, line_width=2, example=str(model.names))
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_np.shape).round()
            
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f"{model.names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
        
    im0 = annotator.result()
    
    # Convert the image to base64 for display
    buffered = io.BytesIO()
    Image.fromarray(im0).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f'data:image/png;base64,{img_str}'

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        
        # Perform defect detection on the uploaded image
        result_image = detect_defect(image)
        
        return html.Div([
            html.H5(filename),
            html.Img(src=result_image, style={'width': '100%'}),
        ])

if __name__ == '__main__':
    app.run_server(debug=True)