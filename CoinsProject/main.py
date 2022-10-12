# Importing required library
import numpy as np
import cv2 as cv
from utils import imutils
import streamlit as st
import os




class CoinsApp:
    
    def __init__(self, title=None):
        """
            The constructor set the title, image, coins image, height of the image, width of the image and the no. of coins found
            param1 : Title of the app
        """
        self.title = title
        self.image = None
        self.coins = None
        self.local = False
        self.h, self.w = 0, 0
        self.count = 0
        
    def set_title(self):
        """
            This method set_title() set the title of the app.
            If title is not define then by default it set title to 'Welcome to the App'
        """
        if self.title:
            st.title(self.title)
        else:
            st.title('Welcome to the App')

    def set_image(self, image):
        """
            This method read the byte data and then convert it into numpy array with unsigned integer 8 bit.
            After that it decode the numpy array to OpenCV colored image.

            param1 : image to convert from byte data to OpenCV image formate

            return -> BGR image
        """
        npimg = np.frombuffer(image, np.uint8)
        image = cv.imdecode(npimg, cv.IMREAD_COLOR)
        return cv.cvtColor(image, cv.COLOR_RGB2BGR)

    def download(self, image):
        """
            This method let the user to download the image to their local directory

            @parameter
            param1 : image to download
        """
        image = imutils.resize(image, self.w)
        cv.imwrite('TestImage/Image.jpg', cv.cvtColor(image, cv.COLOR_BGR2RGB))
        with open('TestImage/Image.jpg', 'rb') as file:
            st.download_button(
                label='Download the image',
                data = file,
                file_name='image.jpg',
                mime = 'image/jpg'
            )

    def detect_coins(self, image, mth, color=None):
        """
            This method detect the coins from the given image.
            This method uses Canny Edge Detection method to detect the edge of the image and then draw the circle around the image. 

            @paramters
            param1 : image in which the coins should be detected 
            param2 : method to Blur the image 
                        0 -> Median Blur
                        1 -> Gaussian Blur
            param3 : color by which the coins will be circled

            return -> The image with the coins surrounded with the circle


            @Steps
            Step 1 : It convert the image to GRAY Scale.
            Step 2 : Then it blur the image.
            Step 3 : It detect the edge using canny edge detection
            Step 4 : Find the contours of the image
            Step 5 : It check and update the contours if its radius is greater than 30 and less than 50
            Step 6 : And finally it draw the circle around the image.
        """
        # color = (0, 0, 0) if color is None else color
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
        
        return (coins, cnts, len(new_cnts))

    def display(self, mth = 0, color=None):
        """
            This method display the method Uploaded image and the image with the coins

            @parameters
            param : method to Blur the image 
                        0 -> Median Blur
                        1 -> Gaussian Blur
            param2 : color by which the coins will be circled
        """
        byte_data = self.image.getvalue()
        st.image(self.image, 'Uploaded Image', width=200)

        image = self.set_image(byte_data)
        self.h, self.w = image.shape[0], image.shape[1]
        image = imutils.resize(image, 255)

        (self.coins, cnts, self.count) = self.detect_coins(image, mth, color)

        st.image(self.coins, 'Coins Detected', width=200)
        st.info(f'The no of coins found {self.count}')

    def display_local(self, mth = 0, color=None):
        """
            This method load the local image.
            This method display the Uploaded image and the image with the coins.

            @parameters
            param : method to Blur the image 
                        0 -> Median Blur
                        1 -> Gaussian Blur
            param2 : color by which the coins will be circled
        """
        
        st.image(self.image, 'Uploaded Image', width=200)

        self.h, self.w = self.image.shape[0], self.image.shape[1]
        image = imutils.resize(self.image, 255)

        (self.coins, cnts, self.count) = self.detect_coins(image, mth, color)

        st.image(self.coins, 'Coins Detected', width=200)
        st.info(f'The no of coins found {self.count}')

    def __getlist__(self):
        """
            This method is used to get the list of the image in the current directory
        """
        ext = ['.jpg', 'jpeg', '.png']
        files = os.listdir('TestImage')
        lists = [os.path.splitext(file) for file in files]
        new_list = ['None']

        for idx, list in enumerate(lists):
            if list[1] in ext:
                new_list.append(files[idx])
        return new_list

    def set_app(self):
        """
            This method create the enviornment to which the user interact.
        """
        with st.sidebar:
            image = st.selectbox('Some Sample images', self.__getlist__())
            if image is not None:
                image = cv.imread(image)
                self.image = image
                self.local = True
            else:
                self.image = None
                self.local = False

        if self.image is None:
            self.image = st.file_uploader('Upload the image', type=['.jpg', '.jpeg'])

        if self.image is not None:
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
                st.balloons()
                self.display_local(meth, color) if self.local else self.display(meth, color)
                self.download(self.coins)
            
        else:
            st.warning('Image not loaded')



    def run(self):
        """
            This method finally set the title and app.
        """
        self.set_title()
        self.set_app()




if __name__ == '__main__':
    coins_app = CoinsApp('Welcome to the Coin\'s counter')
    with st.spinner('Loading...'):
        coins_app.run()
