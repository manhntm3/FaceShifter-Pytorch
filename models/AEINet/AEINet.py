import torch
import os
import itertools
from torch import nn
from utils import util
import torch.nn.functional as F
from collections import OrderedDict
from InsightFace_Pytorch.get_face_embs import ArcFaceBackbone
from .AADLayer import AADResBlock, AADGenerator, MultiLevelAttributeEncoder
from loss.loss import GANLoss, IdentityLoss, IdentityContrastiveLoss, ReconstructionLoss, AttributeLoss
from models.MultiscaleDiscriminator import MultiscaleDiscriminator
from . import networks

class AEINet(nn.Module):
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for AEI network
        """
        parser.add_argument('--AEI_mode', type=str, default="AEI", choices='(AEI)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--network_identity', type=str, default='ir_se50', help='Network architecture for identity feature extraction model')
        parser.add_argument('--arcface_id_size', type=int, default=512, help='')
        parser.add_argument('--lambda_id', type=float, default=5.0, help='weight for identity loss')
        parser.add_argument('--lambda_att', type=float, default=20.0, help='weight for attribute loss')
        parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for reconstruction loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        # parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        # parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for AEINet
        if opt.AEI_mode.lower() == "aei":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        else:
            raise ValueError(opt.AEI_mode)
        return parser

    def __init__(self, opt):
        super(AEINet, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTra`in
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.optimizers = []
        self.metric=0
        self.loss_names = ['G', 'D', 'GAN', 'E_G', 'ID', 'REC', 'ATT', 'D_real', 'D_fake']
        self.model_names = ['G', 'E', 'Z', 'D']
        self.visual_names = ['source_img', 'target_img', 'fake']
        self.faceid_size = opt.arcface_id_size

        self.netG = AADGenerator(self.faceid_size)
        self.netE = MultiLevelAttributeEncoder()
        self.netZ = ArcFaceBackbone(opt)
        self.netZ.eval()

        if self.isTrain:
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.lambda_ID = opt.lambda_id
            self.lambda_ATT = opt.lambda_att
            self.lambda_REC = opt.lambda_rec
            self.criterionID = IdentityContrastiveLoss().to(self.device)
            self.criterionREC = ReconstructionLoss().to(self.device)
            self.criterionATT = AttributeLoss().to(self.device)

            self.netD = MultiscaleDiscriminator(input_nc=opt.input_nc, n_layers=opt.n_layers_D)

            self.optimizer_GE = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netE.parameters()), lr=opt.lr, betas=(opt.adam_config_b1, opt.adam_config_b2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.adam_config_b1, opt.adam_config_b2))
            self.optimizers.append(self.optimizer_GE)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        pass

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.source_img = input['src_data'].to(self.device)
        self.target_img = input['tgt_data'].to(self.device)
        self.same = input['same'].to(self.device)
        self.src_paths = input['src_paths']
        self.tgt_paths = input['tgt_paths']

    def optimize_parameters(self):
        # forward
        self.forward(self.target_img, self.source_img)

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_GE.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_GE.step()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

    def forward(self, target_img, source_img):
        with torch.no_grad():
            Z_id_real = self.netZ(F.interpolate(source_img[:, :, 19:237, 19:237], size=112, mode='bilinear', align_corners=True))
            # Z_id_target = self.netZ(F.interpolate(target_img[:, :, 19:237, 19:237], size=112, mode='bilinear', align_corners=True))
        Z_id_real = F.normalize(Z_id_real).detach()
        self.Z_id_real = Z_id_real
        # self.Z_id_target = F.normalize(Z_id_target).detach()
        self.feature_map_real = self.netE(target_img)

        self.fake = self.netG(Z_id_real, self.feature_map_real)

        Z_id_fake = self.netZ(F.interpolate(self.fake[:, :, 19:237, 19:237], size=112, mode='bilinear', align_corners=True))
        Z_id_fake = F.normalize(Z_id_fake)
        self.Z_id_fake = Z_id_fake
        self.feature_map_fake = self.netE(self.fake)

    def compute_G_loss(self):
        D_score_fake = self.netD(self.fake)
        self.loss_GAN = self.criterionGAN(D_score_fake, True, for_discriminator=False)

        self.loss_ATT = self.criterionATT(self.feature_map_real, self.feature_map_fake)
        self.loss_ID = self.criterionID(self.Z_id_real, self.Z_id_target, self.Z_id_fake, self.same)
        self.loss_REC = self.criterionREC(self.target_img, self.fake, self.same)

        # if self.loss_D_fake < 0.1:
        #     import pdb; pdb.set_trace()
        # print("Att: ", self.loss_ATT.item(), "Id: ", self.loss_ID.item(), "Rec: ", self.loss_REC.item())

        self.loss_E_G = self.lambda_ID * self.loss_ID + self.lambda_REC * self.loss_REC + self.lambda_ATT * self.loss_ATT

        self.loss_G = self.loss_E_G + self.loss_GAN
        return self.loss_G

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake.detach()
        D_score_real = self.netD(self.target_img)
        D_score_fake = self.netD(fake)

        self.loss_D_real = self.criterionGAN(D_score_real, True)
        self.loss_D_fake = self.criterionGAN(D_score_fake, False)
        # if self.loss_D_fake < 0.1:
        #     import pdb; pdb.set_trace()
        # print("D fake: ", self.loss_D_fake.item(), "D real: ", self.loss_D_real.item())

        self.loss_D = self.loss_D_fake + self.loss_D_real
        return self.loss_D

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # if not self.isTrain or opt.continue_train:
        #     load_suffix = opt.epoch
        #     self.load_networks(load_suffix)
        self.parallelize()
        # self.print_networks(opt.verbose)
        
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def parallelize(self):

        self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids).to(self.device)
        self.netE = torch.nn.DataParallel(self.netE, self.gpu_ids).to(self.device)
        self.netZ = torch.nn.DataParallel(self.netZ, self.gpu_ids).to(self.device)
        if self.isTrain:
            self.netD = torch.nn.DataParallel(self.netD, self.gpu_ids).to(self.device)
        # for name in self.model_names:
        #     if isinstance(name, str):
        #         net = getattr(self, 'net' + name)
        #         setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret