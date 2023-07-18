import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm

from models.mlp import FCNet
from models.pe import PE
from models.exp import FCNetExposure

def adjust_exposure(image, exposure):
    image = image * exposure
    image = torch.clamp(image, 0, 1)
    return image

def get_coords(res, normalize = False):
    x = y = torch.arange(res)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack([xx, yy], dim=-1)
    if normalize:
        coords = coords / (res - 1)
    return coords

def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

class ImageDataset(Dataset):
    def __init__(self, image_path, image_size, device = torch.device('cpu'), exposure = 1.0):
        self.image = Image.open(image_path).resize(image_size)
        self.rgb_vals = torch.from_numpy(np.array(self.image)).reshape(-1, 3).to(device)
        self.rgb_vals = self.rgb_vals.float() / 255
        self.rgb_vals = adjust_exposure(self.rgb_vals, exposure)
        self.coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)
        self.exposure = torch.full((len(self.rgb_vals), 1), exposure, device=device, dtype=torch.float32)


    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        return self.coords[idx], self.rgb_vals[idx], self.exposure[idx]

class Trainer:
    def __init__(self, image_path, image_size, model_type = 'mlp', use_pe = True, device = torch.device('cpu'), exposure = 2):
        self.dataset = ImageDataset(image_path, image_size, device, exposure)
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)


        if model_type == 'mlp':
           self.model = FCNet().to(device)
        elif model_type == 'mlp_exposure':
            self.model = FCNetExposure().to(device)
        else:
            pass

        # self.load_model()

        lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = 200

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            # exposures = [5]
            # for exposure in exposures:
            #     self.dataset.exposure = torch.full((len(self.dataset.rgb_vals), 1), exposure, device=self.dataset.exposure.device, dtype=torch.float32)
            #     self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)

            self.model.train()
            for coords, rgb_vals, exposure in self.dataloader:
                self.optimizer.zero_grad()
                if isinstance(self.model, FCNetExposure):
                    pred = self.model(coords, exposure)
                else:
                    pred = self.model(coords)
                loss = self.criterion(pred, rgb_vals)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                coords = self.dataset.coords
                exposure = self.dataset.exposure
                if isinstance(self.model, FCNetExposure):
                    pred = self.model(coords, exposure)
                else:
                    pred = self.model(coords)
                gt = self.dataset.rgb_vals
                psnr = get_psnr(pred, gt)
            pbar.set_description(f'Epoch: {epoch}, PSNR: {psnr:.2f}')
            pred = pred.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            pred = (pred * 255).astype(np.uint8)
            gt = self.dataset.rgb_vals.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            gt = (gt * 255).astype(np.uint8)
            save_image = np.hstack([gt, pred])
            save_image = Image.fromarray(save_image)
            #save_image.save(f'output_{epoch}.png')
            self.visualize(np.array(save_image), text = '# params: {}, PSNR: {:.2f}'.format(self.get_num_params(), psnr))
        # self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), 'model_weights.pth')

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load('model_weights.pth'))
            self.model.eval()
            print("Loaded model parameters.")
        except FileNotFoundError:
            print("No saved model parameters found, starting from scratch.")

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def visualize(self, image, text):
        save_image = np.ones((300, 512, 3), dtype=np.uint8) * 255
        img_start = (300 - 256)
        save_image[img_start:img_start + 256, :, :] = image
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, text, position, font, scale, color, thickness)
        cv2.imshow('image', save_image)
        cv2.waitKey(1)



if __name__ == '__main__':
    image_path = 'image.jpg'
    image_size = (256, 256)
    model_type = 'mlp_exposure'
    device = torch.device('cpu')
    
    trainer = Trainer(image_path, image_size, model_type, device)

    print('# params: {}'.format(trainer.get_num_params()))
    trainer.run()

    # trainer_mlp = Trainer(image_path, image_size, 'mlp', device)
    # trainer_mlp_exposure = Trainer(image_path, image_size, 'mlp_exposure', device)

    # print('Training MLP model...')
    # print('# params: {}'.format(trainer_mlp.get_num_params()))
    # trainer_mlp.run()

    # print('Training MLP_Exposure model...')
    # print('# params: {}'.format(trainer_mlp_exposure.get_num_params()))
    # trainer_mlp_exposure.run()

