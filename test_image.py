import time
import torch
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import random
from PIL import Image
import numpy as np
import glob
from torchvision import transforms

input_folder = ["/root/FaceDataset/FFHQ/images256x256/", "/root/FaceDataset/celeba_hq/images256x256/"]
output_folder = "./sample/output/"  
output_visual_folder = "./sample/output_visual/"  
CKPTS = "./checkpoints/all-epochx"
NUMBER_OF_COUPLE = 100

if __name__ == '__main__':

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
    list_source_img = []
    for i in input_folder:
        list_source_img.extend(glob.glob(i + "*.png"))

    source_index_list = random.sample(range(0, len(list_source_img)), NUMBER_OF_COUPLE)
    target_index_list = random.sample(range(0, len(list_source_img)), NUMBER_OF_COUPLE)

    for index in range(1):
        source_path = list_source_img[source_index_list[index]]
        target_path = list_source_img[target_index_list[index]]

        source_img_orgin = Image.open(source_path)
        target_img_orgin = Image.open(target_path)

        source_img = transform(source_img_orgin).unsqueeze(0).to(device)
        target_img = transform(target_img_orgin).unsqueeze(0).to(device)
        print(source_img.max(), source_img.min())
        start_time = time.time()
        with torch.no_grad():
            model.forward(target_img, source_img)
            output_img = model.fake
        print("Time: ", time.time() - start_time)
        output_img = (output_img + 1) / 2.0 
        output_path = os.path.basename(source_path)[:-4] + "to" + os.path.basename(target_path)[:-4]+".jpg"
        # print(output.max(), output.min())
        output_img = transforms.ToPILImage()(output_img.cpu().squeeze().clamp(0,1)) 
        output_img.save(output_folder + output_path)

        list_img = [source_img_orgin, target_img_orgin, output_img]
        imgs_comb = np.hstack(np.asarray(list_img))
        imgs_comb = Image.fromarray(imgs_comb)

        imgs_comb.save(output_visual_folder + output_path)




