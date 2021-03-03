import pandas as pd
import os
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import boonsdk
from boonsdk import app_from_env
from boonsdk.search import SimilarityQuery

from boonlab.proxies import download_proxy


def do_similarity(f):

    app = app_from_env()

    if len(f) == 2048:
        sim_hash = f
    else:
        sim_hash = app.client.upload_files("/ml/v1/sim-hash", [f], body=None)

    q = {
        "size": 16,
        "query": {
            "bool": {
                "must": [
                    SimilarityQuery(sim_hash, min_score=0.1)
                ]
            }
        }
    }

    sim_search = app.assets.search(q)

    st.text(len(sim_search.assets))
    paths = []
    images = []
    for a in sim_search.assets:
        name = 'tmp/' + str(a.id) + '.jpg'
        paths.append(name)
        im = download_proxy(a, 0)
        im = cv2.resize(im, None, fx=.5, fy=.5)
        images.append(im)

    st.image(images)

access_key = st.sidebar.text_input('accessKey')
secret_key = st.sidebar.text_input('secretKey')
app = boonsdk.BoonApp({"accessKey":access_key,"secretKey":secret_key})

# Specify canvas parameters in application
stroke_width = 3
stroke_color = '#e22'
bg_color = "#eee"
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])


pil_img = Image.open(bg_image)
img = np.array(pil_img)
xrc = img.shape[0]
xscale = 600 / xrc
yrc = img.shape[0]
yscale = 400 / yrc

scale = min(xscale, yscale)
resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)

img_draw = Image.fromarray(resized)


yr = img.shape[0]
xr = img.shape[1]
drawing_mode = 'rect'

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="" if bg_image else bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    width=600,
    height=400,
    drawing_mode=drawing_mode,
    key="canvas",
)

if len(canvas_result.json_data["objects"]) == 0:
    st.text('Draw a box to do similarity')
else:
    left = canvas_result.json_data["objects"][0]['left']
    top = canvas_result.json_data["objects"][0]['top']
    width = canvas_result.json_data["objects"][0]['width']
    height = canvas_result.json_data["objects"][0]['height']

    st.text(resized.shape)

    x1 = left
    y1 = top
    x2 = left+width
    y2 = top+height

    bbox = [x1, y1, x2, y2]
    crop = resized[y1:y2, x1:x2]
    st.sidebar.image(crop)
    crop_name = 'crop.jpg'
    cv2.imwrite(crop_name, crop)

    do_similarity(crop_name)
