import cv2
import PIL
import os, glob,tqdm

list_img = glob.glob("/root/FaceDataset/FFHQ/images256x256/*.png")

for img in tqdm.tqdm(list_img):
    new_img = img.replace("/root/FaceDataset/FFHQ", "/root/FFHQ_png")
    new_img = new_img.replace(".png", ".jpg")
    image = cv2.imread(img)
    # print(new_img)
    cv2.imwrite(new_img, image)