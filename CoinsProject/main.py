# Importing required library
from altair import display
import numpy as np
import cv2 as cv
from utils import imutils
import streamlit as st


class CoinsApp:
    
    def __init__(self, title=None):
        self.title = title
        self.image = None
        self.coins = None
        self.count = 0
        
    def set_title(self):
        if self.title:
            st.title(self.title)
        else:
            st.title('Welcome to the App')

    def set_image(self, image):
        npimg = np.frombuffer(image, np.uint8)
        image = cv.imdecode(npimg, cv.IMREAD_COLOR)
        return image

    def detect_coins(self, image):
        coins = image.copy()
        image = cv.GaussianBlur(image, (9, 9), 0)

        edge = cv.Canny(image, 30, 150)

        (cnts, _) = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for i, c in enumerate(cnts):
            ((center_X, center_Y), radius) = cv.minEnclosingCircle(c)
            if radius > 30:
                print(radius, center_X, center_Y)

        cv.drawContours(coins, cnts, -1, (0, 0, 0), 2)
        
        return coins, len(cnts)

    def display(self):
        byte_data = self.image.getvalue()
        st.image(byte_data, 'Uploaded Image', width=200)

        image = self.set_image(byte_data)
        image = imutils.resize(image, 200)
        image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

        (self.coins, self.count) = self.detect_coins(image)

        st.image(self.coins, 'Coins Detected', width=200)
        st.write(f'The no of coins found {self.count}')

    def set_app(self):
        self.image = st.file_uploader('Upload the image')

        if self.image:
            st.write('Image Uploaded Successfully')        
            st.button('Find', on_click=self.display)
        else:
            st.write('Image not loaded')



    def run(self):
        self.set_title()
        self.set_app()




if __name__ == '__main__':
    coins_app = CoinsApp('Welcome to the Coin\'s counter')
    coins_app.run()
    