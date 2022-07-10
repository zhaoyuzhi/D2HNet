import cv2
import numpy as np


def resize_img(imgs, size):
    """
    imgs: list of numpy array(NHWC) or numpy array(NHWC)
    """
    if isinstance(imgs, list):
        resize_imgs = []
        for i in range(len(imgs)):
            resize_imgs.append([])
            for j in range(imgs[i].shape[0]):
                resize_imgs[i].append(cv2.resize(imgs[i][j], (size, size), interpolation = cv2.INTER_AREA))
            resize_imgs[i] = np.array(resize_imgs[i])
        return tuple(resize_imgs)
    elif isinstance(imgs, np.ndarray):
        resize_imgs = []
        for i in range(len(imgs)):
            resize_imgs.append(cv2.resize(imgs[i], (size, size), interpolation = cv2.INTER_AREA))
        resize_imgs = np.array(resize_imgs)

        return resize_imgs
    else:
        raise NotImplementedError('Type %s is not supported.' % type(imgs))


def read_img(opt, path):
    if hasattr(opt, 'memcache'):
        import mc
        server_list_config_file = opt.memcache.server_path
        client_config_file = opt.memcache.client_path
        mclient = mc.MemcachedClient.GetInstance(
            server_list_config_file, client_config_file)
        value = mc.pyvector()
        mclient.Get(path, value)
        value_str = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_str, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

 
def sobel_xy(img):
    sobelX = cv2.Sobel(img,cv2.CV_64F,1,0)              # x-direction gradient
    sobelY = cv2.Sobel(img,cv2.CV_64F,0,1)              # y-direction gradient
    sobelX = np.uint8(np.absolute(sobelX))              # x-direction gradient, absolute value
    sobelY = np.uint8(np.absolute(sobelY))              # y-direction gradient, absolute value
    sobelCombined = cv2.bitwise_or(sobelX,sobelY)
    return sobelCombined
