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
        self.Z_id_real = F.normalize(Z_id_real).detach()
        self.feature_map_real = self.netE(target_img)
        self.fake = self.netG(self.Z_id_real, self.feature_map_real)
        Z_id_fake = self.netZ(F.interpolate(self.fake[:, :, 19:237, 19:237], size=112, mode='bilinear', align_corners=True))
        Z_id_fake = F.normalize(Z_id_fake)
        self.Z_id_fake = Z_id_fake
        self.feature_map_fake = self.netE(self.fake)

    def compute_G_loss(self):
        D_score_fake = self.netD(self.fake)
        self.loss_GAN = self.criterionGAN(D_score_fake, True, for_discriminator=False)

        self.loss_ATT = self.criterionATT(self.feature_map_real, self.feature_map_fake)
        self.loss_ID = self.criterionID(self.Z_id_real, self.Z_id_target)
        self.loss_REC = self.criterionREC(self.target_img, self.fake, self.same)

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

        self.loss_D = self.loss_D_fake + self.loss_D_real
        return self.loss_D