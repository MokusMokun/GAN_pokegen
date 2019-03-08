# By @Mirope Yuhao Hu
# This python script take 256x256 pokemon images set and augmented them
# 11x into bunch of folders (for testing purpose)
import os, sys, cv2, time
from PIL import Image
import tensorflow as tf

HEIGHT, WIDTH, CHANNEL = 128, 128, 3

opath = 'original_dataset/' # the original dataset
rbpath = 'resized_backgrounded_dataset/'
vfpath = 'augmented/vfilp/'
hfpath = 'augmented/hflip/'
cpath = 'augmented/rancrop'
bcpath = 'augmented/ranbricon'
r5path = 'augmented/rotate5/'
r10path = 'augmented/rotate10/'
r15path = 'augmented/rotate15/'
r20path = 'augmented/rotate20/'
rn5path = 'augmented/rotate_n5/'
rn10path = 'augmented/rotate_n10/'
rn15path = 'augmented/rotate_n15/'
rn20path = 'augmented/rotate_n20/' 


def path_validation():
    if not os.path.exists(rbpath):
        os.makedirs(rbpath)
    if not os.path.exists(vfpath):
        os.makedirs(vfpath)
    if not os.path.exists(hfpath):
        os.makedirs(hfpath)
    if not os.path.exists(r5path):
        os.makedirs(r5path)
    if not os.path.exists(r10path):
        os.makedirs(r10path)
    if not os.path.exists(r15path):
        os.makedirs(r15path)
    if not os.path.exists(r20path):
        os.makedirs(r20path)
    if not os.path.exists(rn5path):
        os.makedirs(rn5path)
    if not os.path.exists(rn10path):
        os.makedirs(rn10path)
    if not os.path.exists(rn15path):
        os.makedirs(rn15path)
    if not os.path.exists(rn20path):
        os.makedirs(rn20path)
    

def background_and_resize():
    for each in os.listdir(opath):
        png = Image.open(os.path.join(opath, each))
        if png.mode == 'RGBA':
            png.load() # required for png.split()
            background = Image.new("RGB", png.size, (0,0,0))
            background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
            background.save(os.path.join(rbpath,each.split('.')[0] + '.jpg'), 'JPEG')
        else:
            png.convert('RGB')
            png.save(os.path.join(rbpath,each.split('.')[0] + '.jpg'), 'JPEG')

    for each in os.listdir(rbpath):
        img = cv2.imread(os.path.join(rbpath, each))
        img = cv2.resize(img,(128,128))
        cv2.imwrite(os.path.join(rbpath, each), img)
    
def rancrop():
        session = tf.InteractiveSession()
        for each in os.listdir(rbpath):
            img = tf.read_file(rbpath+each)
            image = tf.image.decode_jpeg(img, channels = CHANNEL)
            image = tf.random_crop(image, [HEIGHT-10,WIDTH-10,3])
            size = [HEIGHT, WIDTH]
            image = tf.image.resize_images(image, size)
            image.set_shape([HEIGHT,WIDTH,CHANNEL])
            image = tf.cast(image, tf.uint8)
            imgname = '/crop-'+ each
            image = tf.image.encode_jpeg(image)
            file = tf.write_file(cpath+'4'+imgname, image)
            session.run(file)
        session.run(image)
        session.close()

def ranbright_contrast():
        session = tf.InteractiveSession()
        for each in os.listdir(rbpath):
            img = tf.read_file(rbpath+each)
            image = tf.image.decode_jpeg(img, channels = CHANNEL)
            image = tf.image.random_brightness(image, max_delta = 0.1)
            image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
            image = tf.cast(image, tf.uint8)
            imgname = '/bri_con-'+ each
            image = tf.image.encode_jpeg(image)
            file = tf.write_file(bcpath+'4'+imgname, image)
            session.run(file)
        session.close()


def hflip():
    for each in os.listdir(rbpath):
        img = cv2.imread(os.path.join(rbpath, each))
        img = cv2.flip(img, 1)
        imgname = 'hflip-'+ each
        cv2.imwrite(os.path.join(hfpath + imgname), img)

def vflip():
    for each in os.listdir(rbpath):
        img = cv2.imread(os.path.join(rbpath, each))
        img = cv2.flip(img, 0)
        imgname = 'vflip-'+ each
        cv2.imwrite(os.path.join(vfpath + imgname), img)

def rotate5():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(5)
        imgname = 'r5-'+each
        img.save(os.path.join(r5path + imgname), 'JPEG')

def rotate10():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(10)
        imgname = 'r10-'+each
        img.save(os.path.join(r10path + imgname), 'JPEG')

def rotate15():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(15)
        imgname = 'r15-'+each
        img.save(os.path.join(r15path + imgname), 'JPEG')

def rotate20():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(20)
        imgname = 'r20-'+each
        img.save(os.path.join(r20path + imgname), 'JPEG')

def rotaten5():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(-5)
        imgname = 'rn5-'+each
        img.save(os.path.join(rn5path + imgname), 'JPEG')

def rotaten10():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(-10)
        imgname = 'rn10-'+each
        img.save(os.path.join(rn10path + imgname), 'JPEG')

def rotaten15():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(-15)
        imgname = 'rn15-'+each
        img.save(os.path.join(rn15path + imgname), 'JPEG')

def rotaten20():
    for each in os.listdir(rbpath):
        img = Image.open(os.path.join(rbpath, each))
        img = img.rotate(-20)
        imgname = 'rn20-'+each
        img.save(os.path.join(rn20path + imgname), 'JPEG')

if __name__ == '__main__':
    # each function does separate tasks, comment out if you dont need any of them
    path_validation()
    background_and_resize()
    rancrop()
    ranbright_contrast()
    hflip()
    vflip()
    rotate5()
    rotate10()
    rotate15()
    rotate20()
    rotaten5()
    rotaten10()
    rotaten15()
    rotaten20()