utils.py
import scipy.misc, numpy as np, os, sys
import cv2 as cv

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def rgb2luv(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    luv = np.array([[.299, .587, .114],[-.147, -.288, .436],[.615, -.515, -.1]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return luv.transpose(1,2,0)
def luv2rgb(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    rgb = np.array([[1, 0, 1.139],[1, -.395, -.580],[1, 2.03, 0]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return rgb.transpose(1,2,0)


def histogram(content, styles):
    new_styles=[]
    for i in range(len(styles)):
        #content_sub = cv.resize(content, (256,256), interpolation=cv.INTER_CUBIC)/1.0
        #style_sub = cv.resize(style, (256,256), interpolation=cv.INTER_CUBIC)/1.0
        style = styles[i]
        content_sub = content
        style_sub = styles[i]
        mean_c = np.zeros((3))/1.0
        mean_s = np.zeros((3))/1.0
        conv_c = np.zeros((3,3))/1.0
        conv_s = np.zeros((3,3))/1.0
        for i in range (0,3):
        	mean_c[i] = np.mean(content_sub[:,:,i])
        	mean_s[i] = np.mean(style_sub[:,:,i])
        for i in range (0,3):
        	for j in range (0,3):
        		conv_c[i,j] = np.mean((content_sub[:,:,i]-mean_c[i])*(content_sub[:,:,j]-mean_c[j]))
        		conv_s[i,j] = np.mean((style_sub[:,:,i]-mean_s[i])*(style_sub[:,:,j]-mean_s[j]))
        eig_c, vec_c = np.linalg.eig(conv_c)

        eig_s, vec_s = np.linalg.eig(conv_s)

        if (False == np.all(eig_c>0.0001) or False == np.all(eig_s>0.0001)):
            new_styles.append(style.copy())
            continue

        sqrt_conv_c = np.dot(np.dot(vec_c, np.diag(eig_c**0.5)), vec_c.transpose())

        sqrt_conv_s_inv = np.dot(np.dot(vec_s, np.diag(eig_s**-0.5)), vec_s.transpose())

        A_chol = np.dot(sqrt_conv_c, sqrt_conv_s_inv)

        b_chol = mean_c - np.dot(A_chol, mean_s)

        new_style = style.copy()

        new_style_size = new_style.shape[0]*new_style.shape[1]

        new_style_shape = [new_style.shape[0],new_style.shape[1]]

        new_style_newshape = np.zeros((3,new_style_size))/1.0

        new_style_newshape[0,:] = new_style[:,:,0].flatten()

        new_style_newshape[1,:] = new_style[:,:,1].flatten()

        new_style_newshape[2,:] = new_style[:,:,2].flatten()

        new_style_newshape = np.dot(A_chol, new_style_newshape)+b_chol.repeat(new_style_size).reshape(3,new_style_size)

        new_style[:,:,0] = new_style_newshape[0,:].reshape(new_style_shape)

        new_style[:,:,1] = new_style_newshape[1,:].reshape(new_style_shape)

        new_style[:,:,2] = new_style_newshape[2,:].reshape(new_style_shape)
        new_styles.append(new_style)
    return np.array(new_styles)
