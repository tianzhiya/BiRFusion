import os

from dataset.RegDataset import RegData

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from TrainOptions import TrainOptions
from Module.model import BiRGenerator
from Tools.saver import Saver
from Module.TII import *

from torch.utils.data import DataLoader

def trainReg(opts):
    # daita loader
    print('\n--- load dataset ---')
    dataset = RegData(opts)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    model = BiRGenerator(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    saver = Saver(opts)

    # train
    print('\n--- train ---')
    ep0 = 0
    for ep in range(ep0, opts.n_ep):

        for it, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) in enumerate(train_loader):
            # input data
            image_ir = image_ir.cuda(opts.gpu).detach()
            image_vi = image_vi.cuda(opts.gpu).detach()
            image_ir_warp = image_ir_warp.cuda(opts.gpu).detach()
            image_vi_warp = image_vi_warp.cuda(opts.gpu).detach()
            deformation = deformation.cuda(opts.gpu).detach()
            if len(image_ir.shape) > 4:
                image_ir = image_ir.squeeze(1)
                image_vi = image_vi.squeeze(1)
                image_ir_warp = image_ir_warp.squeeze(1)
                image_vi_warp = image_vi_warp.squeeze(1)
                deformation = deformation.squeeze(1)


            model.trainReg(image_ir, image_vi, image_ir_warp,
                           image_vi_warp, deformation)

            if (total_it + 1) % 10 == 0:
                Reg_Img_loss = model.loss_reg_img
                Reg_Field_loss = model.loss_reg_field
                Total_loss = model.loss_total
                print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                    total_it, ep, it, model.DM_opt.param_groups[0]['lr'], Total_loss))
                print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}'.format(
                    Reg_Img_loss, Reg_Field_loss))
            total_it += 1

        print(ep)
        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()
        saver.write_model(ep, opts.n_ep, model)

    return


if __name__ == '__main__':
    parser = TrainOptions()
    opts = parser.parse()
    trainReg(opts)
