import cv2
import streamlit as st
import numpy as np
from PIL import Image

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

st.title('顔にモザイクをかける')
uploaded_file = st.file_uploader("写真アップロード", type='jpg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(img_array,caption = '元画像',use_column_width = None)

    #OpenCVが用意した顔認識モデルを読み込み
    face_cascade_path = r'Streamlit/haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(face_cascade_path)

    src_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    #顔を検出して座標情報を返す
    facerect = cascade.detectMultiScale(src_gray)

    print('Face Coordinate:', facerect[0])

    if len(facerect) > 0:
        for [x,y,w,h] in facerect:
            dst_face = mosaic_area(img_array, x, y, w, h)

    img_array2 = np.array(dst_face)
    st.image(img_array2,caption = 'モザイク処理後',use_column_width = True)
