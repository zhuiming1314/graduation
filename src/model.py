import network
import torch
import torch.nn as nn

class TwinsNet(nn.Moudle):
    def __init__(self, opts):
        super(TwinsNet, self).__init__()
        
        # parameters
        lr = 0.0001
        betas = (0.5, 0.999)
        weight_decay = 0.0001
        self.nz = 8
        
        # encoders
        self.enc_c = network.ContentEncoder(opts.input_dim_a, opts.input_dim_b)
        self.enc_s = network.StyleEncoder(opts.input_dim_a, opts.input_dim_b)

        # generator
        self.gen = network.Generator(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

        # discirminators
        #self.disA = network.Discriminator(opts.input_dim_a, opts.dis_scale)
        self.disB = network.Discriminator(opts.input_dim_a, opts.dis_scale)

        # optimizers
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.enc_s_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def set_gpu(self, gpu):
        self.gpu = gpu
        self.enc_c.cuda(self.gpu)
        self.enc_s.cuda(self.gpu)
        self.gen.cuda(self.gpu)
        #self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)

    def forward(self):
        # get real content encode
        self.real_content_a, self.real_content_b = self.enc_c.forward(self.input_a, self.input_b)

        # get real style encode
        self.real_style_a, self.real_style_b = self.enc_s.forward(self.input_a, self.input_b)

        # generate imgae of content b and style a
        input_b_to_a = torch.cat((self.content_b, self.style_a), 0)
        self.output_fake_b = self.gen.forward_b(input_b_to_a)

        # get content encode from b and style encode from a
        self.fake_content_b = self.enc_c.forward_b(self.output_fake_b)
        self.fake_style_a = self.enc_s.forward_a(self.output_fake_b)

        # generate image of real content a and fake style a
        input_a_to_a = torch.cat((self.real_content_a, self.fake_style_a), 0)
        self.rec_a = self.gen.forward_a(input_a_to_a)

        # generate image of fake b content and real style b
        input_b_to_b = torch.cat((self.fake_content_b, self.real_style_b), 0)
        self.rec_b = self.gen.forward_b(input_b_to_b)

    def backward_gen(self):
        outs_fake = self.disB.forward(self.output_fake_b)
        loss_gen = 0
        for out in outs_fake:
            o = nn.functional.sigmoid(out)
            all_ones = torch.ones_like(o).cuda(self.gpu)
            loss_gen += nn.functional.binary_cross_entropy(o, all_ones)
        return loss_gen

    def backward_rec(self):
        loss_rec_a = torch.nn.L1Loss(self.input_a, self.rec_a)
        loss_rec_b = torch.nn.L1Loss(self.input_b, self.rec_b)
        
        return loss_rec_a, loss_rec_b


    def update_enc_gen(self):
        self.enc_c_opt.zero_grad()
        self.enc_s_opt.zero_grad()
        self.gen_opt.zero_grad()

        loss_gen = self.backward_gen()
        loss_rec_a, loss_rec_b = self.backward_rec()
        loss = loss_gen + loss_rec_a + loss_rec_b

        loss.backward(retain_graph=True)

        self.enc_c_opt.step()
        self.enc_s_opt.step()
        self.gen_opt.step()

    def backward_dis(self):
        pred_fake = self.disB.forward(self.output_fake_b)
        pred_real = self.disB.forward(self.input_b)

        loss_dis = 0

        for _, (fake, real) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(fake)
            out_real = nn.functional.sigmoid(real)
            all_zeros = torch.zeros_like(out_fake).cuda(self.gpu)
            all_ones = torch.ones_like(out_real).cuda(self.gpu)
            loss_dis_fake = nn.functional.binary_cross_entropy(out_fake, all_zeros)
            loss_dis_real = nn.functional.binary_cross_entropy(out_real, all_ones)

            loss_dis += loss_dis_fake + loss_dis_real

        return loss_dis

    def update_dis(self, input_a, input_b):
        self.input_a = input_a
        self.input_b = input_b

        # encode and generate first
        self.forward()

        # update dis
        self.disB_opt.zero_grad()
        loss_dis = self.backward_dis()
        loss_dis.backward()
        self.disB_opt.step()

    def save_model(self, filename, ep, total_iter):
        state = {
            "enc_c": self.enc_c.state_dict(),
            "enc_s": self.enc_s.state_dict(),
            "gen": self.gen.state_dict(),
            "disB": self.gen.state_dict(),
            "enc_c_opt": self.enc_c_opt.state_dict(),
            "enc_s_opt": self.enc_s_opt.state_dict(),
            "gen_opt": self.gen_opt.state_dict(),
            "disB_opt": self.disB_opt.state_dict(),
            "ep": ep,
            "total_iter": total_iter
        }
        torch.save(state, filename)

    def resume(self, filename, train=True):
        checkpoint = torch.load(filename)

        if train:
            self.enc_c.load_state_dict(checkpoint["enc_c"])
            self.enc_s.load_state_dict(checkpoint["enc_s"])
            self.gen.load_state_dict(checkpoint["gen"])
            self.disB.load_state_dict(checkpoint["disB"])
            self.enc_c_opt.load_state_dict(checkpoint["enc_c_opt"])
            self.enc_s_opt.load_state_dict(checkpoint["enc_s_opt"])
            self.gen_opt.load_state_dict(checkpoint["gen_opt"])
            self.disB_opt.load_state_dic(checkpoint["disB_opt"])

            return checkpoint["ep"], checkpoint["total_iter"]
    
    def assemble_outputs(self):
        img_a = self.input_a.detach()
        img_b = self.input_b.detach()
        img_a_to_b = self.output_fake_b.detach()
        rec_a = self.rec_a.detach()
        rec_b = self.rec_b.detach()
        
        return torch.cat((img_a, img_b, img_a_to_b, rec_a, rec_b), dim=0)

    def forward_transfer(self, input_a, input_b):
        content_a, content_b = self.enc_c(input_a, input_b)
        style_a, style_b = self.enc_s(input_a, input_b)

        output = self.gen.forward_b(content_a, style_b)

        return output




