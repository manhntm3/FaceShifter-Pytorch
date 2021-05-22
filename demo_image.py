import time
import torch
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import glob
import PIL
import dlib
import random
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

image_path = "./sample/demo/"
source_image_path = image_path + "source.jpg"
target_image_path = image_path + "target.jpg"
source_input_path = image_path + "source_input.jpg"
target_input_path = image_path + "target_input.jpg"
swap_image_path = image_path + "swap.jpg"
output_image_path = image_path + "output.jpg"
CKPTS = "./checkpoints/all-epochx"
NUMBER_OF_COUPLE = 100

def crop_aligned(detector, sp, output_size, input_img):
    transform_size = 4096
    enable_padding=True
    torch.backends.cudnn.benchmark = False
    img = dlib.load_rgb_image(input_img)
    dets = detector(img, 1)
    if len(dets) < 0:
        print("no face landmark detected")
    else:
        shape = sp(img, dets[0])
        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            points[b, 0] = shape.part(b).x
            points[b, 1] = shape.part(b).y
        lm = points
    # lm = fa.get_landmarks(input_img)[-1]
    # lm = np.array(item['in_the_wild']['face_landmarks'])
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.open(input_img)
    img = img.convert('RGB')

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return img


if __name__ == '__main__':
    output_size = 256
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('./data/preprocess/shape_predictor_68_face_landmarks.dat')

    source_img = crop_aligned(detector, sp, output_size=256, input_img=source_image_path)
    target_img = crop_aligned(detector, sp, output_size=256, input_img=target_image_path)

    source_img.save(source_input_path)
    target_img.save(target_input_path)

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    device = torch.device(f"cuda:{opt.gpu_ids[0]}" if torch.cuda.is_available() else 'cpu')
    model = create_model(opt)      # create a model given opt.model and other options
    model.netG.load_state_dict(torch.load(CKPTS + "/latest_net_G.pth", map_location=str(device)))
    model.netE.load_state_dict(torch.load(CKPTS + "/latest_net_E.pth", map_location=str(device)))
    model.netZ.load_state_dict(torch.load(CKPTS + "/latest_net_Z.pth", map_location=str(device)))
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    model.eval()
    # model.freeze()  # run inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    source_img_orgin = source_img
    target_img_orgin = target_img

    source_img = transform(source_img_orgin).unsqueeze(0).to(device)
    target_img = transform(target_img_orgin).unsqueeze(0).to(device)
    print(source_img.max(), source_img.min())
    start_time = time.time()
    with torch.no_grad():
        model.forward(target_img, source_img)
        output_img = model.fake
    print("Time: ", time.time() - start_time)
    output_img = (output_img + 1) / 2.0 
    output_img = transforms.ToPILImage()(output_img.cpu().squeeze().clamp(0,1)) 
    output_img.save(swap_image_path)

    list_img = [source_img_orgin, target_img_orgin, output_img]
    imgs_comb = np.hstack([np.asarray(img) for img in list_img])
    imgs_comb = Image.fromarray(imgs_comb)

    imgs_comb.save(output_image_path)
