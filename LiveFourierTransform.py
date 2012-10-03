# Live Fourier Transform 
# original Matlab code by Brian R. Pauw (http://www.lookingatnothing.com/)
# python code by Samuel K.Z. Tardif
# 2011-09-22

import pygame
import Image
from pygame.locals import *
import opencv
import sys
from opencv import highgui 
import time



def get_image(camera):
  """
  Grabs a frame from the camera 
  and returns it as a PIL image
  """
  im = highgui.cvQueryFrame(camera)
  im = opencv.cvGetMat(im)
  return opencv.adaptors.Ipl2PIL(im)  #convert Ipl image to PIL image



def logfft(im_data,kill_center_lines=True):
  """
  Computes the FFT of the image, shift the origin at the center
  and returns the log of the intensity (absolute value squared) 
  scaled on 8 bits
  """
  imft = numpy.fft.fftshift(numpy.fft.fft2(im_data))
  ft_I = abs(imft)**2
  ft_m = abs(imft)
  ft_a = angle(imft)
  
  if kill_center_lines:
    # kill the center lines for higher dynamic range 
    # by copying the next row/column
    h,w=ft_I.shape
    ft_I[h/2,:]=ft_I[h/2+1,:]
    ft_I[:,w/2]=ft_I[:,w/2+1]
    ft_m[h/2,:]=ft_m[h/2+1,:]
    ft_m[:,w/2]=ft_m[:,w/2+1]
  
  #logscale
  ft_I = log10(ft_I+ones(imft.shape))
  ft_m = log10(ft_m+ones(imft.shape))
  
  #over 8 bits
  log10Imax = ft_I.max()
  ft_I = (255*ft_I/log10Imax).astype(uint8)
  ft_m = (255*ft_m/log10(sqrt(10**log10Imax))).astype(uint8)
  ft_a = (255*(ft_a+pi)/(2*pi)).astype(uint8)
  
  return ft_I, ft_m, ft_a



# frame per second
fps = 30.0


# define the ROI of the CCD to be used
# the CCD full size is 480x640 by default
roi_h,roi_w = 300,300
roi_h_offset,roi_w_offset = 90,120

# fft parameter
kill_center_lines=False

# start the stream
camera = highgui.cvCreateCameraCapture(0)


# start the display
pygame.init()
window = pygame.display.set_mode((roi_w*2,roi_h*2))
pygame.display.set_caption("TL: Real space | TR: Intensity | BL: Modulus | BR: Angle")
screen = pygame.display.get_surface()


i=0
starttime=time.time()
stop=False
while not(stop):
  
  # loop control (stops on key down)
  events = pygame.event.get()
  for event in events:
    if event.type == QUIT or event.type == KEYDOWN:
      stop=True
      
      
  # grabs the image from the camera and converts it to grayscale
  im = get_image(camera)
  im = im.convert(mode = 'L')
  
  # get the data as an 8 bits array and do the FFT
  im_data=numpy.asarray(im.getdata()).reshape((480,640)).astype(uint8)
  h,w=im_data.shape
  im_data=im_data[roi_h_offset:roi_h_offset+roi_h,roi_w_offset:roi_w_offset+roi_w]
  ft_I, ft_m, ft_a=logfft(im_data,kill_center_lines)
  
  # stitch the real space image and the FT
  newshape = (im_data.shape[1]*2,im_data.shape[0]*1)
  im_top = transpose(append(transpose(im_data),transpose(ft_I)).reshape(newshape))
  im_bot = transpose(append(transpose(ft_m),transpose(ft_a)).reshape(newshape))
  newshape = (im_data.shape[0]*2,im_data.shape[1]*2)
  im_all = append(im_top, im_bot).reshape(newshape) 
  im_mod=Image.fromarray(im_all)
  
  # KLUDGE : convert back to RGB for display
  im_mod=im_mod.convert('RGB')
  
  # insert modified image in stream
  pg_img = pygame.image.frombuffer(im_mod.tostring(), im_mod.size, im_mod.mode)
  screen.blit(pg_img, (0,0))
  pygame.display.flip()
  pygame.time.delay(int(1000 * 1.0/fps))
  i+=1
  print "current frame rate = %2.2f fps"%(i/(time.time()-starttime))
  
  
  
# cleaning
pygame.quit()
highgui.cvReleaseCapture(camera)




#  Webcam properties
#  #    name
#  0    CV_CAP_PROP_POS_MSEC
#  1    CV_CAP_PROP_POS_FRAMES
#  2    CV_CAP_PROP_POS_AVI_RATIO
#  3    CV_CAP_PROP_FRAME_WIDTH
#  4    CV_CAP_PROP_FRAME_HEIGHT
#  5    CV_CAP_PROP_FPS
#  6    CV_CAP_PROP_FOURCC
#  7    CV_CAP_PROP_FRAME_COUNT
#  8    CV_CAP_PROP_FORMAT
#  9    CV_CAP_PROP_MODE
# 10    CV_CAP_PROP_BRIGHTNESS
# 11    CV_CAP_PROP_CONTRAST
# 12    CV_CAP_PROP_SATURATION
# 13    CV_CAP_PROP_HUE
# 14    CV_CAP_PROP_GAIN
# 15    CV_CAP_PROP_EXPOSURE
# 16    CV_CAP_PROP_CONVERT_RGB
# 17    CV_CAP_PROP_WHITE_BALANCE
# 18    CV_CAP_PROP_RECTIFICATION
