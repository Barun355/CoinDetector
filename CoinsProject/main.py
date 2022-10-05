# Importing required library
import numpy as np
import cv2 as cv
from utils import imutils
import streamlit as st


class CoinsApp:
    
    def __init__(self, title=None):
        self.title = title
        self.image = None
        self.coins = None
        self.h, self.w = 0, 0
        self.count = 0
        
    def set_title(self):
        if self.title:
            st.title(self.title)
        else:
            st.title('Welcome to the App')

    def set_image(self, image):
        npimg = np.frombuffer(image, np.uint8)
        image = cv.imdecode(npimg, cv.IMREAD_COLOR)
        return cv.cvtColor(image, cv.COLOR_RGB2BGR)

    def download(self, image):
        image = imutils.resize(image, self.w)
        cv.imwrite('Image.jpg', cv.cvtColor(image, cv.COLOR_BGR2RGB))
        with open('Image.jpg', 'rb') as file:
            st.download_button(
                label='Download the image',
                data = file,
                file_name='image.jpg',
                mime = 'image/jpg'
            )

    def detect_coins(self, image, mth, color=None):
        color = (0, 0, 0) if color is None else color
        coins = image.copy()
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if mth == 0: image = cv.medianBlur(image, 3)
        elif mth == 1: image = cv.GaussianBlur(image, (5, 5), 0)
        new_cnts = []
        edge = cv.Canny(image, 30, 150)

        (cnts, _) = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        for i, c in enumerate(cnts):
            ((center_X, center_Y), radius) = cv.minEnclosingCircle(c)
            if radius > 30 and radius < 50:
                new_cnts.append(c)
                cv.circle(coins, (int(center_X), int(center_Y)), int(radius), color, 2)
                
        # cv.drawContours(coins, new_cnts, -1, (0, 0, 0), 2)
        
        return coins, cnts, len(new_cnts)

    def display(self, mth = 0, color=None):
        byte_data = self.image.getvalue()
        st.image(byte_data, 'Uploaded Image', width=200)

        image = self.set_image(byte_data)
        self.h, self.w = image.shape[0], image.shape[1]
        image = imutils.resize(image, 255)

        (self.coins, cnts, self.count) = self.detect_coins(image, mth, color)

        st.image(self.coins, 'Coins Detected', width=200)
        st.info(f'The no of coins found {self.count}')

    def set_app(self):
        self.image = st.file_uploader('Upload the image')

        if self.image:
            st.success('Image Uploaded Successfully')        
            meth = 0
            method = st.radio('Choice the method', ('Method 01', 'Method 02'))
            if method == 'Method 01':
                meth = 0
            elif method == 'Method 02':
                meth = 1

            color = st.color_picker('Pick a color to draw circle')
            color = tuple(int(color[i+1:i+3], 16) for i in (0, 2, 4))
            
            if st.button('Find'):
                self.display(meth, color)
                self.download(self.coins)
            
        else:
            st.warning('Image not loaded')



    def run(self):
        self.set_title()
        self.set_app()




if __name__ == '__main__':
    coins_app = CoinsApp('Welcome to the Coin\'s counter')
    coins_app.run()
    
