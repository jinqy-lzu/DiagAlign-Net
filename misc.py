import shutil
import os
import torch 
import logging  
class CheckpointManager():
    def __init__(self, model, optimizer=None, ema_model=None, save_dir='', keep_num=10, rank=0, additions={}):
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.additions = additions
        self.save_dir = save_dir
        self.keep_num = keep_num
        self.rank = rank
        self.ckpts = []
        if self.rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.metrics_fp = open(os.path.join(save_dir, 'metrics.csv'), 'a')
            self.metrics_fp.write('epoch,train_loss,test_loss,top1,top5\n')

    def update(self, epoch, metrics, score_key='top1'):
        if self.rank == 0:
            self.metrics_fp.write('{},{},{}\n'.format(epoch, metrics['test_loss'], metrics['top1']))
            self.metrics_fp.flush()

        score = metrics[score_key]
        insert_idx = 0
        for ckpt_, score_ in self.ckpts:
            if score > score_:
                break
            insert_idx += 1
        if insert_idx < self.keep_num:
            save_path = os.path.join(self.save_dir, 'checkpoint-{}.pth.tar'.format(epoch))
            self.ckpts.insert(insert_idx, [save_path, score])
            if len(self.ckpts) > self.keep_num:
                remove_ckpt = self.ckpts.pop(-1)[0]
                if self.rank == 0:
                    if os.path.exists(remove_ckpt):
                        os.remove(remove_ckpt)
            self._save(save_path, epoch, is_best=(insert_idx == 0))
        else:
            self._save(os.path.join(self.save_dir, 'last.pth.tar'), epoch)
        return self.ckpts

    def _save(self, save_path, epoch, is_best=False):
        if self.rank != 0:
            return
        save_dict = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict() if self.ema_model else None
            # 'optimizer': self.optimizer.state_dict() if self.optimizer else None,
        }
        for key, value in self.additions.items():
            save_dict[key] = value.state_dict() if hasattr(value, 'state_dict') else value

        torch.save(save_dict, save_path)
        if save_path != os.path.join(self.save_dir, 'last.pth.tar'):
            shutil.copy(save_path, os.path.join(self.save_dir, 'last.pth.tar'))
        if is_best:
            shutil.copy(save_path, os.path.join(self.save_dir, 'best.pth.tar'))

    def load(self, ckpt_path):
        save_dict = torch.load(ckpt_path, map_location='cpu')

        for key, value in self.additions.items():
            if hasattr(value, 'load_state_dict'):
                value.load_state_dict(save_dict[key])
            else:
                self.additions[key] = save_dict[key]

        if 'state_dict' in save_dict and 'model' not in save_dict:
            save_dict['model'] = save_dict['state_dict']
        # if isinstance(self.model, DDP):
        #     missing_keys, unexpected_keys = \
        #         self.model.module.load_state_dict(save_dict['model'], strict=False)
        # else:
        missing_keys, unexpected_keys = \
                self.model.load_state_dict(save_dict['model'], strict=False)
        if len(missing_keys) != 0:
            print(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            print(f'Unexpected keys in source state dict: {unexpected_keys}')
        
        if self.ema_model is not None and 'ema_model' in save_dict:
            self.ema_model.load_state_dict(save_dict['ema_model'])
        if self.optimizer is not None and 'optimizer' in save_dict:
            self.optimizer.load_state_dict(save_dict['optimizer'])

        if 'epoch' in save_dict:
            epoch = save_dict['epoch']
        else:
            epoch = -1

        '''avoid memory leak'''
        del save_dict
        torch.cuda.empty_cache()

        return epoch
import time
def init_logger(args):
    logger = logging.getLogger()
    if not os.path.exists(args.best_model_path):
        os.makedirs(args.best_model_path)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.best_model_path, f'log_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.txt'))
    fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    logger.info(f'Experiment directory: {args.best_model_path}')