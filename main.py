import torch
from torch.backends import cudnn

from parameters import *
from trainer import Trainer
from tester import Tester
from data_loader import CustomDataLoader
from utils import make_folder
from augmentations import *


def main(config):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(2020)

    if config.train:
        # Create directories if not exist
        make_folder(config.model_save_path, config.arch)
        make_folder(config.sample_path, config.arch) # test results sample
        make_folder(config.test_pred_label_path, config.arch) # test pred results
        make_folder(config.test_color_label_path, config.arch) # colorful test pred results

        # Transform for Data Augment
        transform = Compose([RandomHorizontallyFlip(p=.5), RandomSized(size=config.imsize), \
            AdjustBrightness(bf=0.1), AdjustContrast(cf=0.1), AdjustHue(hue=0.1), \
            AdjustSaturation(saturation=0.1)])
        
        data_loader = CustomDataLoader(config.img_path, config.label_path, config.imsize,
                                       config.batch_size, num_workers=config.num_workers, 
                                       transform=transform, mode=config.train)
        val_loader = CustomDataLoader(config.val_img_path, config.val_label_path, config.imsize,
                                      config.batch_size, num_workers=config.num_workers, 
                                      transform=None, mode=bool(1 - config.train))
        trainer = Trainer(data_loader.loader(), config, val_loader.loader())
        trainer.train()
    else:
        data_loader = CustomDataLoader(config.test_image_path, config.test_label_path, config.imsize,
                                       config.batch_size, num_workers=config.num_workers, mode=config.train)
        tester = Tester(data_loader.loader(), config)
        tester.test()


if __name__ == '__main__':
    config = get_parameters()
    main(config)
