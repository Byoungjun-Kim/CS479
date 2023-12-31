import json
import math
import time
from pathlib import Path

import torch
import yaml

from .dataset import get_dataset, get_torchvision_transforms
from .models.model import DiffusionAutoEncoders
from .sampler import Sampler
from .trainer import Trainer
from .utils import get_torchvision_unnormalize


class DiffusionAutoEncodersInterface:
    CFG_DIR = Path('./diffae/cfg')

    def __init__(self, args, mode):
        """Setting up config, output directory, model, and dataset.

        Args:
            args (dict): A dict of arguments with the following keys,
                mode == 'train':
                    data_name (str): Dataset name.
                    size (int): Image size.
                    expn (str): Experiment name.
                mode in {'test', 'clf_train', 'clf_test', 'infer'}:
                    output (str): Path to output directory.
        """
        assert mode in {'train', 'test', 'clf_train', 'clf_test', 'infer'}
        self.mode = mode
        self.cfg = self._init_config(args)

        if self.mode == 'train':
            self.output_dir = self._init_output_dir(args)
            saved_cfg_file = self.output_dir / 'model.yml'
            with saved_cfg_file.open('w') as fp:
                yaml.safe_dump(self.cfg, fp, sort_keys=False)
        else:
            self.output_dir = Path(args['output'])

        model_ckpt_path = self.output_dir / 'ckpt' / args['model_ckpt'] if 'model_ckpt' in args else None
        model_ckpt_path = None if self.mode == 'train' else self.output_dir / 'ckpt' / args['model_ckpt']
        self.model = self._init_model(model_ckpt_path)

        if self.mode in {'test', 'infer'}:
            self.sampler = Sampler(self.model, self.cfg)
            self.unnormalize = get_torchvision_unnormalize(
                self.cfg['test']['dataset'][-1]['params']['mean'],
                self.cfg['test']['dataset'][-1]['params']['std'],
            )

        self._init_dataset()

    def _init_config(self, args):
        """Setting up config dict.
        """
        print('Initializing config...')
        # Load config file.
        if self.mode == 'train':
            cfg_path = self.CFG_DIR / f'{args["size"]}_model.yml'
        else:
            cfg_path = Path(args['output']) / 'model.yml'
        with cfg_path.open('r') as fp:
            cfg = yaml.safe_load(fp)

        if self.mode == 'train':
            cfg['general']['dataset_path'] = args['dataset_path']
        
        if not torch.cuda.is_available():
            print('cuda is not available')
            cfg['general']['device'] = 'cpu'

        return cfg

    def _init_output_dir(self, args):
        """Create output directory.
        """
        assert self.mode == 'train'
        print('Initializing output directory...')
        output_root = Path(self.cfg['general']['output_root'])
        output_dir = output_root / time.strftime('%Y%m%d%H%M%S') if args['expn'] is None else output_root / args['expn']
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def _init_model(self, ckpt_path=None):
        """Setting up DiffusionAutoEncoders module.
        """
        print('Initializing Diffusion Autoencoders...')
        model = DiffusionAutoEncoders(self.cfg)
        if ckpt_path is not None:
            print('Loading Diff-AE checkpoint...')
            state = torch.load(ckpt_path)
            model.load_state_dict(state['model'])
        return model

    def _init_dataset(self):
        """Setting up transforms and dataset.
        """
        assert self.mode in {'train', 'test', 'clf_train', 'clf_test', 'infer'}
        print('Initializing dataset...')

        if self.mode in {'train', 'clf_train'}:
            self.transforms = get_torchvision_transforms(self.cfg, 'train')
        else:
            self.transforms = get_torchvision_transforms(self.cfg, 'test')

        dataset_path = self.cfg['general']['dataset_path']
        self.dataset = get_dataset(dataset_path=dataset_path, transform=self.transforms)

    def train(self):
        """DiffusionAutoEncoders training.
        """
        assert self.mode == 'train'

        trainer = Trainer(self.model, self.cfg, self.output_dir, self.dataset)
        trainer.train()

    @torch.inference_mode()
    def test(self):
        """DiffusionAutoEncoders evaluation.
        """
        assert self.mode == 'test'

        print('Evaluation start...')
        result = self.sampler.sample_testdata(self.dataset)
        with (self.output_dir / 'test.json').open('w') as fp:
            json.dump(result, fp, indent=4, sort_keys=True)


    @torch.inference_mode()
    def infer(self, image, xt=None, style_emb=None):
        """Autoencode a single image.

        Args:
            image: (PIL Image): A single PIL Image.
            style_emb (torch.tensor, optional): A tensor of SemanticEncoder embedding.
                You can perform conditional generation with arbitary embedding by passing this argument.

        Returns:
            result (dict): A result of autoencoding which has the following keys,
                input (numpy.ndarray): A input image array.
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        image = self.transforms(image)
        result = self.sampler.sample_one_image(image, xt=xt, style_emb=style_emb)

        # Unnormalize and to numpy.ndarray
        for k, v in result.items():
            if isinstance(v, list):
                for i, x in enumerate(result[k]):
                    assert torch.is_tensor(x)
                    result[k][i] = self.unnormalize(x).permute(1, 2, 0).cpu().detach().numpy()
            elif torch.is_tensor(v):
                result[k] = self.unnormalize(v).permute(1, 2, 0).cpu().detach().numpy()

        return result

    @torch.inference_mode()
    def infer_manipulate(self, image, to_dir):
        """Attribute manipulation using classifier.

        Args:
            image (PIL Image): A single PIL Image.
            target_id (int): Target attribute id.
            s (float or List[float]): Attribute manipulation parameter(s).

        Returns:
            result (dict): A result of autoencoding which has the following keys,
                input (numpy.ndarray): A input image array.
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        device = self.cfg['general']['device']

        x0 = self.transforms(image).unsqueeze(dim=0).to(device)
        xt = self.sampler.encode_stochastic(x0)
        style_emb = self.model.encoder(x0)
        style_emb[:, :to_dir.size(0)] = to_dir
        result = self.infer(image, xt=xt, style_emb=style_emb)

        return result
