import os
import os.path as osp
import time
import torch
import tqdm
import yaml

from argparse import ArgumentParser
from attrdict import AttrDict
from torch.nn import Module

from data.transitions import Sampler, RaySampler
from utils.log import get_logger, RunningAverage
from utils.misc import load_module


def train(
        model: Module,
        neuboots: bool,
        root: str,
        expid: str,
        eval_config: dict = None,
        batch_size: int = 100,
        max_num_points: int = 300,
        min_num_points: int = 64,
        num_bootstrap: int = 10,
        learning_rate: float = 1e-4,
        num_steps: int = 100000,
        print_freq: int = 200,
        eval_freq: int = 5000,
        save_freq: int = 1000,
        resume: bool = False,
        device: torch.device = torch.device('cpu'),
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    if resume:
        # load check point
        ckpt = torch.load(osp.join(root, 'ckpt.tar'), map_location=device)
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        # scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step

    else:
        logfilename = osp.join(root, f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not resume:
        logger.info(f"MBMRL Experiment: [{model.__class__.__name__.lower()}] {expid}")
        logger.info(f"Device: {device}\n")
        logger.info(f"Total number of params: {sum(p.numel() for p in model.parameters())}\n")

    sampler = RaySampler(device=device)
    for step in range(start_step, num_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=batch_size,
            max_num_points=max_num_points,
            min_num_points=min_num_points
        )

        if neuboots:
            outs = model(batch=batch,
                         num_samples=1,
                         num_bs=num_bootstrap)
        else:
            outs = model(batch=batch,
                         num_samples=num_bootstrap)

        outs.loss.backward()
        optimizer.step()
        # scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % print_freq == 0:
            line = f"{model.__class__.__name__.lower()}: step {step} "
            line += f"lr {optimizer.param_groups[0]['lr']:.3e} "
            line += ravg.info()
            logger.info(line)

            # if step % eval_freq == 0:
            #     line = eval(
            #         dim_problem=dim_problem,
            #         model=model,
            #         neuboots=neuboots,
            #         mode='train',
            #         root=root,
            #         bound=bound,
            #         max_num_points=max_num_points,
            #         device=device,
            #         **eval_config
            #     )
            #     logger.info(line + '\n')

            ravg.reset()

        if step % save_freq == 0 or step == num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            # ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, osp.join(root, "ckpt.tar"))

            if step % 100000 == 0:
                torch.save(ckpt, osp.join(root, f"ckpt_{step}.tar"))

    # eval(
    #     dim_problem=dim_problem,
    #     model=model,
    #     neuboots=neuboots,
    #     mode='eval',
    #     root=root,
    #     bound=bound,
    #     max_num_points=max_num_points,
    #     device=device,
    #     **eval_config
    # )
    #
    # plot_log(logfilename)
    # if num_steps >= 50000:
    #     plot_log(logfilename, 1, num_steps)


def main():
    parser = ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--loc', choices=['aigs', 'nipa', 'local'], default='nipa')

    parser.add_argument('--model', type=str, default='neubanp')
    # if neuboots-based model
    parser.add_argument('--wagg', choices=['mean', 'max'], default='mean')
    parser.add_argument('--wattn', action='store_false', default=True)

    # train
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--max_num_points', type=int, default=300)
    parser.add_argument('--min_num_points', type=int, default=64)
    parser.add_argument('--train_num_bootstrap', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=1000000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    # eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_bootstrap', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    args = parser.parse_args()
    if args.gpu < 0:
        args.loc = 'local'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neuboots = 'neu' in args.model

    with open(f'configs/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    neuboots_config = {}
    if neuboots:
        neuboots_config = {"yenc": True, "wenc": True, "wagg": args.wagg, "wloss": True}
        if args.model == 'neubanp' and args.wattn:
            neuboots_config['wattn'] = True

    model_cls = getattr(load_module(f"models/{args.model}.py"), args.model.upper())
    model = model_cls(**config, **neuboots_config).to(device)

    with open("runner/paths.yaml") as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)[args.loc]
        datasets_path = paths["datasets_path"]
        evalsets_path = paths["evalsets_path"]
        results_path = paths["results_path"]

    expid = f'bs{args.train_num_bootstrap}'
    if neuboots:
        if args.wagg != 'mean':
            expid += f'_{args.wagg}'

    root = osp.join(results_path,
                    'mbmrl',
                    args.model,
                    f'{expid}_batch{args.train_batch_size}_min{args.min_num_points}_max{args.max_num_points}')

    if not osp.isdir(root):
        os.makedirs(root)

    with open(osp.join(root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    eval_config = {
        'eval_path': evalsets_path,
        'num_batch': args.eval_num_batches,
        'batch_size': args.eval_batch_size,
        'num_bootstrap': args.eval_num_bootstrap,
        'eval_logfile': args.eval_logfile,
        'seed': args.eval_seed
    }

    if args.mode == "train":
        train(
            model=model,
            neuboots=neuboots,
            root=root,
            expid=expid,
            eval_config=eval_config,
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            min_num_points=args.min_num_points,
            num_bootstrap=args.train_num_bootstrap,
            learning_rate=args.lr,
            num_steps=args.num_steps,
            print_freq=args.print_freq,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            resume=args.resume,
            device=device
        )

    # elif args.mode == "eval":
    #     eval(
    #         dim_problem=args.dimension,
    #         model=model,
    #         neuboots=neuboots,
    #         mode=args.mode,
    #         root=root,
    #         bound=args.bound,
    #         max_num_points=args.max_num_points,
    #         device=device,
    #         **eval_config
    #     )

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
