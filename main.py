import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class shapeMatcher():
    def __init__(self,sample_path, img_2_path, img_c_path):
        self.threshold= 90
        self.img_sample = self.read_image(sample_path)
        self.img_c = self.read_image_c(img_c_path)
        self.img_2 = self.read_image(img_2_path)
        self.img_sample_raw = cv2.imread('./images/sample.jpg')
        

    
    def read_image(self, path):
        """
        inverse binary image: text in white color, background in black
        Gaussian blur to the noise
        Erosion to erodes away the boundaries of foregrounnd obejct and dimish the features of an image
        Dilation is used for increases the object area and accentuate features
        """
        img = cv2.imread(path, 0)
        
        _,img_bin = cv2.threshold(img,self.threshold,255,cv2.THRESH_BINARY)
        img_bin = cv2.GaussianBlur(img_bin, (9, 9), 0)
        img_bin_inverse = cv2.bitwise_not(img_bin)
        kernel = np.ones((5, 5), np.uint8)
        img_bin_inverse = cv2.dilate(img_bin_inverse, kernel, iterations=1)
        return img_bin_inverse
    
    def read_image_c(self, path):
        """
        inverse binary image: text in white color, background in black
        Gaussian blur to the noise
        Erosion to erodes away the boundaries of foregrounnd obejct and dimish the features of an image
        Dilation is used for increases the object area and accentuate features
        Erosion is used because it is more difficult to detect C since 6 looks similar with C
        """
        img = cv2.imread(path, 0)
        _,img_bin = cv2.threshold(img,self.threshold,255,cv2.THRESH_BINARY)
        img_bin = cv2.GaussianBlur(img_bin, (13, 13), 0)
        img_bin_inverse = cv2.bitwise_not(img_bin)
        kernel = np.ones((5, 5), np.uint8)

        img_bin_inverse = cv2.erode(img_bin_inverse, np.ones((3, 3), np.uint8), iterations=1)
        img_bin_inverse = cv2.dilate(img_bin_inverse, kernel, iterations=1)
        return img_bin_inverse
    
    def detect_2(self, threshold=0.16):
        """
        1. get the contour of the sample
        2. calculate each fft in the contours list and compare it with 2 using euclidean distance
        3. if the distance < threshold, means it looks like 2
        """
        contours = self.contour_sample
        img = self.img_sample_raw

        for i, contour in enumerate(contours):
            ### get the fourier descriptor for each contour using their complex vector
            complex_vector = self.get_complex_vector(contour)
            fft = self.get_dft(complex_vector)
            fd = finalFD(fft)

            if len(fd) == len(self.descriptor_c):
                distance_2 = np.linalg.norm(fd - self.descriptor_2, 2)
                if distance_2 <= threshold:
                    print('character 2 is detected')
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 5)
                    cv2.putText(img,'{:.5f}'.format(distance_2),(x, y-15), cv2.FONT_HERSHEY_SIMPLEX , 2,(255, 0, 0),5,cv2.LINE_AA)

        cv2.imwrite('images/result.jpg', img)
    
    def detect_c(self, threshold=0.17):
        """
        1. get the contour of the sample
        2. calculate each fft in the contours list and compare it with "C" using euclidean distance
        3. if the distance < threshold, means it looks like "C"
        """

        contours = self.contour_sample
        img = self.img_sample_raw

        for i, contour in enumerate(contours):
            ### get the fourier descriptor for each contour using their complex vector
            complex_vector = self.get_complex_vector(contour)
            fft = self.get_dft(complex_vector)
            fd = finalFD(fft)

            if len(fd) == len(self.descriptor_c):
                distance_c = np.linalg.norm(fd - self.descriptor_c, 2)
    
                if distance_c <= threshold:
                    print('character C is detected')
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 5)
                    cv2.putText(img,'{:.5f}'.format(distance_c),(x, y-15), cv2.FONT_HERSHEY_SIMPLEX , 2,(0, 0, 0),5,cv2.LINE_AA)

        cv2.imwrite('images/result.jpg', img)
    
    def ifft(self, descriptor, name):
        """
        recontrsuct image using fft 
        """
        mid = len(descriptor) // 2
        fft = np.fft.fftshift(descriptor)
        fft = fft[mid - 20: mid + 21]

        ifft = np.fft.ifftshift(fft)
        ifft[0] = 0
        ifft = np.fft.ifft(ifft)

        result = np.array([np.real(ifft), np.imag(ifft)]).T
        result = (result - result.min()) / (result.max() - result.min()) * 100
        result = result.astype('int32')
        result = np.reshape(result, (-1, 1, 2))

        bg = np.zeros(self.img_c .shape)
        cv2.drawContours(bg, [result], -1, (255, 255, 255), 1)

        cv2.imwrite(f'images/ifft_{name}.jpg', bg)
    @staticmethod
    def get_contour(img):
        """
        return the contours of image
        """
        contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours
    
    @staticmethod
    def draw_contour(img, contours, name):
        """
        draw contour on black background
        """
        bg = np.zeros(img.shape)
        cv2.drawContours(bg, contours, -1, (255,255,255), 3)
        cv2.imwrite(os.path.join('images', str(name)+'_contours.jpg'), bg)

    @staticmethod
    def get_complex_vector(contours):
        """
        convert the complex info into complex vector x + j*y
        """
        contours = contours.reshape(-1,2)
        return contours[:, 0] +1j*contours[:,1]
    
    @staticmethod
    def get_dft(complex_vector):
        """
        return dft 
        """
        temp = np.arange(len(complex_vector)).T.reshape((-1, 1))
        exponential = np.exp(-2*np.pi*1j*(np.matmul(temp,temp.T))/len(complex_vector)).T
        dft = np.matmul(complex_vector.T ,exponential)
        return dft
    

def getLowFreqFDs(descriptor):
    return descriptor[:15]

def scaleInvariant(descriptor):
    return descriptor / descriptor[1]

def rotationInvariant(descriptor):
    return np.absolute(descriptor)

def transInvariant(descriptor):
    return descriptor[1:]

def finalFD(descriptor):
    """
    final fourier descriptor which has
    1.low frequency fourier descriptor
    2.invariant to scale
    3.invariant to rotation
    4.invariant to transformation

    """
    descriptor = getLowFreqFDs(descriptor)
    descriptor = scaleInvariant(descriptor)
    descriptor = rotationInvariant(descriptor)
    descriptor = transInvariant(descriptor)

    return descriptor
        
    


if __name__=='__main__':
    matcher = shapeMatcher('./images/sample.jpg','./images/2.jpg','./images/c.jpg')
    matcher.contour_sample = matcher.get_contour(matcher.img_sample)
    matcher.contour_c = matcher.get_contour(matcher.img_c)
    matcher.contour_2= matcher.get_contour(matcher.img_2)
    
    matcher.draw_contour(matcher.img_c, matcher.contour_c,'c')
    matcher.draw_contour(matcher.img_2, matcher.contour_2,2)
    matcher.draw_contour(matcher.img_sample, matcher.contour_sample,'sample')

    matcher.complex_vector_c = matcher.get_complex_vector(matcher.contour_c[0])
    matcher.complex_vector_2 = matcher.get_complex_vector(matcher.contour_2[0])
    matcher.complex_vector_sample = matcher.get_complex_vector(matcher.contour_sample[0])

    fft_c = matcher.get_dft(matcher.complex_vector_c)
    fft_2 = matcher.get_dft(matcher.complex_vector_2)

    matcher.ifft(fft_c, "c")
    matcher.ifft(fft_2, "2")

    matcher.descriptor_c = finalFD(fft_c)
    matcher.descriptor_2 = finalFD(fft_2)


    matcher.detect_2(threshold=0.16)
    matcher.detect_c(threshold=0.17)



