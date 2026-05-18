import argparse
import yaml

def parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    default_dicts = {}
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

            for k, v in cfg.items():
                if isinstance(v, dict):
                    default_dicts[k] = v
                    
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    for k, v in default_dicts.items():
        v.update(args.__dict__[k])
    args.__dict__[k] = v

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='/home/user1/bingli/4_gene_his_text/survival/baseline_config/Brain-LGG.yaml', type=str,
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='ImageNet Training')

parser.add_argument('--epochs', default=100, type=int)#500
parser.add_argument('--gene_dir', default='/home/user1/bingli/Lung/cnv_rna_mut_sub_slide_scale.csv')
parser.add_argument('--img_dir', default='/home/user1/bingli/Lung/pt_file')
parser.add_argument('--diagnostic_reports', default=' /home/user1/bingli/nas/bingli/Brain/LGG_deepseek_diagnostic_reports.csv')
parser.add_argument('--best_model_path', default='/home/user1/bingli/gene_histop/Lung_results')

parser.add_argument('--schedule', default=[10, 20, 30], type=int, nargs='+')#[200, 300, 400]
parser.add_argument('--opt', default='adamw', type=str,
                help='Optimizer (default: "adamw"')
parser.add_argument('--decay_epochs', type=float, default=10, 
                help='epoch interval to decay LR')
parser.add_argument('--opt_eps', default=1e-8, type=float,
                help='Optimizer Epsilon (default: 1e-8, use opt default)')
parser.add_argument('--opt_no_filter', action='store_true', default=False,
                help='disable bias and bn filter of weight decay')
parser.add_argument('--sgd_no_nesterov', action='store_true', default=False,
                help='set nesterov=False in SGD optimizer')
parser.add_argument('--dyrep', action='store_true', default=False,
                help='Use DyRep')
parser.add_argument('--sched', default='cosine', type=str, 
                help='LR scheduler (default: "step" "cosine" ')

parser.add_argument('--decay_rate', '--dr', type=float, default=0.1,
                help='LR decay rate')
parser.add_argument('--decay_by_epoch', action='store_true', default=False,
                help='decay LR by epoch, valid only for cosine scheduler')
parser.add_argument('--min-lr', type=float, default=5e-6, 
                help='minimal learning rate (default: 1e-5)')#5e-6
parser.add_argument('--model_ema', action='store_true', default=False,
                help='Enable tracking moving average of model weights')
parser.add_argument('--model_ema_decay', type=float, default=0.99996,
                help='decay factor for model weights moving average (default: 0.9998)')
parser.add_argument('--warmup-epochs', type=int, default=0, 
                help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup-lr', type=float, default=5.0e-7,
                help='warmup learning rate (default: 0.0001)')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--lr_decay', default=0.1, type=float)
parser.add_argument('--gpu_id', default='3', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--trail', default=0, type=int)
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--train_ration', default=0.5, type=float)
#sur_ret_dir
parser.add_argument('--sur_ret_dir', default="", type=str)
parser.add_argument('--task_type', default="survival", type=str)
parser.add_argument('--bag_loss', default="ce_surv", type=str)
parser.add_argument('--alpha_surv', default=0.0, type=float)

parser.add_argument('--lambda_reg', default=1e-5, type=float)
parser.add_argument('--gc', default=16, type=int)

parser.add_argument('--load_model', default='', type=str)
parser.add_argument('--ig_ret_path', default='', type=str)

parser.add_argument('--mode', default=" ", type=str)

#======================attention vision=====================================
	#保存裁剪后图片和热力图
parser.add_argument('--heatmap_dir', type=str, default="")
parser.add_argument('--slide_patch_save_dir', type=str, default="")
	#完整 wsi slide 图的保存位置
parser.add_argument('--slide_dir', type=str, default="")
	#保存样本slide名称文件
parser.add_argument('--sample_file_path', default="", type=str)
parser.add_argument('--risk', default="", type=str)
#=========================================================================