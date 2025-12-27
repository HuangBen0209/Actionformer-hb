import argparse
import os
import time
import datetime
import pickle
from pprint import pprint
import torch
import torch.nn as nn
import torch.utils.data
# 用于可视化
from torch.utils.tensorboard import SummaryWriter

# 我们的代码
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

from libs.utils.logger import Logger
# 补充AverageMeter（如果libs.utils中没有，需添加）
class AverageMeter(object):
    """用于统计平均值、总和、计数的工具类"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 补充postprocess_results（如果libs.utils中没有，需添加）
def postprocess_results(results, ext_score_file):
    """空实现，可根据实际需求补充后处理逻辑"""
    return results


################################################################################
def main(args):
    """主函数，处理训练/验证过程，保存最优mAP模型"""

    """1. 设置参数/文件夹"""
    # 解析命令行参数
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("配置文件不存在。")
    pprint(cfg)

    # 准备输出文件夹（基于时间戳）
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            str(cfg['output_folder']), str(cfg_filename + '_' + str(ts)))
    else:
        ckpt_folder = os.path.join(
            str(cfg['output_folder']), str(cfg_filename + '_' + str(args.output)))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # 创建TensorBoard写入器
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # 固定随机种子（这会固定所有随机性）
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # 根据GPU数量重新调整学习率/工作进程数
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. 创建数据集/数据加载器"""
    # 训练集
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # 根据数据集属性更新cfg（针对epic-kitchens数据集）
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    # 验证集（新增）
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers'])  # batch_size=1便于验证
    # 获取验证集属性
    val_db_vars = val_dataset.get_attributes()

    """3. 创建模型、优化器和调度器"""
    # 模型
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    print(f"--------------------cfg['model_name'] = {cfg['model_name']}-------------------------")
    # 多GPU训练
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # 优化器
    optimizer = make_optimizer(model, cfg['opt'])
    # 调度器
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # 启用模型EMA（指数移动平均）
    print("使用模型EMA...")
    model_ema = ModelEma(model)
    best_mAP = 0.0  # 初始化最优mAP（平均精度均值）
    best_epoch = 0  # 记录最优mAP对应的epoch
    """4. 从检查点恢复/初始化最优mAP"""
    # 从检查点恢复？
    if args.resume:
        if os.path.isfile(args.resume):
            # 加载检查点
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            # 如果检查点包含mAP，恢复最优mAP
            if 'best_mAP' in checkpoint:
                best_mAP = checkpoint['best_mAP']
                best_epoch = checkpoint['best_epoch']
            print(
                f"=> 加载检查点 '{args.resume}' (epoch {checkpoint['epoch']})，历史最优mAP: {best_mAP:.4f} (epoch {best_epoch})")
            del checkpoint
        else:
            print("=> 在'{}'未找到检查点".format(args.resume))
            return

    # 保存当前配置
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. 训练/验证循环"""
    print("\n开始训练模型 {:s} ...".format(cfg['model_name']))

    # 开始训练
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    print(f"起始epoch/最大epoch数: {args.start_epoch}, {max_epochs}")

    logger = Logger("training_mAP", os.path.join(ckpt_folder, 'logs'))  # 初始化日志记录器


    for epoch in range(args.start_epoch, max_epochs):
        # ========== 训练一个epoch ==========
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        if epoch>=max_epochs//2:
            val_db_vars = val_dataset.get_attributes() #获取验证数据集属性


        # ========== 验证一个epoch ==========
        # 初始化动作定位评估器
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds'])

        # 执行验证（使用EMA模型，结果更稳定）
        current_mAP = valid_one_epoch(
            val_loader,
            model_ema.module,  # 使用EMA模型验证，
            epoch,
            ext_score_file=cfg['test_cfg'].get('ext_score_file', None),
            evaluator=det_eval,
            output_file=None,  # 不保存预测结果文件
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        #记录当前epoch的mAP到日志
        logger.info(f'epoch: {epoch}, mAP: {current_mAP:.4f}')
        # 更新最优模型
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            best_ep = epoch
            print(f'发现更优模型，mAP提升至 {best_mAP:.4f} (epoch {best_ep})，开始保存...')

            # 准备保存的状态字典
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),  # 模型权重
                'scheduler': scheduler.state_dict(),  # 调度器状态
                'optimizer': optimizer.state_dict(),  # 优化器状态
                'mAP': current_mAP  # 当前mAP值
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()  # EMA模型权重

            # 保存最优模型（覆盖原有最优模型）
            save_checkpoint(
                save_states,
                False,  # 不标记为最新模型
                file_folder=ckpt_folder,
                file_name='bestmodel.pth.tar'
            )

        # 打印当前验证结果
        print(
            f"Epoch [{epoch + 1}/{max_epochs}] - 验证mAP: {current_mAP:.4f}, 最优mAP: {best_mAP:.4f} (epoch {best_epoch})")

        # 记录mAP到TensorBoard
        tb_writer.add_scalar('validation/mAP', current_mAP, epoch)

    # 收尾工作
    tb_writer.close()
    print(f"\n训练完成！最优mAP: {best_mAP:.4f} (epoch {best_epoch})")
    print(f"最优模型路径: {os.path.join(ckpt_folder, 'bestmodel.pth.tar')}")
    return

################################################################################
if __name__ == '__main__':
    """程序入口点"""
    # 参数解析器
    parser = argparse.ArgumentParser(
        description='训练一个基于点的Transformer用于动作定位，每轮验证并保存最优mAP模型')
    parser.add_argument('config', metavar='DIR',
                        help='配置文件路径')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='打印频率（默认：每10次迭代）')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='检查点保存频率（默认：每5个epoch）')
    parser.add_argument('--output', default='', type=str,
                        help='实验文件夹名称（默认：无）')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='检查点路径（默认：无）')
    args = parser.parse_args()
    main(args)