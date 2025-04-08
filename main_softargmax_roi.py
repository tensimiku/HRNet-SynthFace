from config import config
from config import update_config
import heatmap_hrnet_softargmax
import torch
from torch.export import export
from dataset import SynthFaceDataset_RoI
from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker
import numpy as np
import os

def main():
    config.defrost()
    config.merge_from_file("heatmap_hrnet.yaml")
    config.freeze()

    lr = 5e-5
    num_epoch = 200
    per_gpu_batch_size = 32
    checkpoint_dir = './networks/hrnet_softargmax_roi'
    save_at = 20

    accelerator = Accelerator(log_with="tensorboard", project_dir=checkpoint_dir)

    model = heatmap_hrnet_softargmax.get_cls_net(config.MODEL.HRNET)
    # print([k for k, v in model.named_parameters()])
    # model.init_weights_without_classification_layer("hrnetv2_w64_imagenet_pretrained.pth")
    # model.init_weights_without_classification_layer("hrnetv2_w32_imagenet_pretrained.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dset = SynthFaceDataset_RoI("./synth_face")
    train_dataloader = torch.utils.data.DataLoader(dset, batch_size=per_gpu_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True, persistent_workers=True)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    init_epoch = 0
    if os.path.exists(checkpoint_dir):
        dirs = list(filter(lambda x: x.startswith("ep"), filter(lambda x: os.path.isdir(os.path.join(checkpoint_dir, x)), os.listdir(checkpoint_dir))))
        if len(dirs) > 0:
            sdirs = sorted(dirs)
            accelerator.print("loading from..", os.path.join(checkpoint_dir, sdirs[-1]))
            accelerator.load_state(os.path.join(checkpoint_dir, sdirs[-1]))
            try:
                init_epoch = int(sdirs[-1][2:]) + 1
                accelerator.print("set initial epoch to ", init_epoch)
            except ValueError as e:
                pass
    
    accelerator.init_trackers('logs')
    tracker: TensorBoardTracker = accelerator.get_tracker("tensorboard")

    total_steps = len(dset)*init_epoch
    for epoch in range(init_epoch, num_epoch+1):
        model.train()
        ep_loss = 0.0
        for step, (x, lmk) in enumerate(train_dataloader):
            with accelerator.autocast():
                y = model(x)
                y = y.reshape(-1, 70, 2) # b, l, 2+1 (x, y, sigma)
                loss = torch.nn.functional.smooth_l1_loss(y, lmk)
                accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1)
            accelerator.print(f"current epoch {epoch}: loss: {loss.item()}", end='\r')
            accelerator.log({"step/loss":loss.item()}, total_steps)
            total_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            ep_loss += loss.item()
        
        with torch.no_grad():
            if accelerator.is_main_process:
                ep_loss /= (step+1)
                accelerator.log({"epoch/loss":ep_loss}, epoch)
                x_imgs = []
                y_imgs = []
                gt_imgs = []
                for ix, iy, ilmk in zip(x, y, lmk):
                    ix = ix.cpu().numpy()
                    iy = iy.cpu().numpy()
                    ilmk = ilmk.cpu().numpy()
                    img, img_lmk = dset.draw_landmark(ix, iy)
                    _, img_lmk_gt = dset.draw_landmark(ix, ilmk)
                    x_imgs.append(img)
                    y_imgs.append(img_lmk)
                    gt_imgs.append(img_lmk_gt)
                x_imgs = np.array(x_imgs)
                y_imgs = np.array(y_imgs)
                gt_imgs = np.array(gt_imgs)
                tracker.writer.add_images('epoch/x', x_imgs, epoch, dataformats='NHWC')
                tracker.writer.add_images('epoch/y', y_imgs, epoch, dataformats='NHWC')
                tracker.writer.add_images('epoch/gt', gt_imgs, epoch, dataformats='NHWC')
            
        if epoch % save_at == 0:
            if accelerator.is_main_process:
                accelerator.save_state(os.path.join(checkpoint_dir, 'ep%04d'%epoch))
                torch.save({'x': x, 'y': y}, os.path.join(checkpoint_dir, 'ep%04d'%epoch+".pt"))
    
    accelerator.end_training()

if __name__ == "__main__":
    main()