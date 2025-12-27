"""
日志工具模块
============
该模块实现了增强版的日志记录功能，主要包含：
1. basename函数：获取文件路径的基础名称，支持去除文件扩展名
2. CustomTimedRotatingFileHandler类：自定义的按时间轮转的文件处理器，修复了时区和DST问题
3. Logger类：封装了logging.Logger，支持同时输出日志到控制台和按时间轮转的文件
核心特性：
- 支持按时间自动切割日志文件（小时/天/周等）
- 自动处理夏令时(DST)切换问题
- 保留指定数量的备份日志文件
- 同时输出到控制台和文件，格式统一
"""

import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler

__all__ = ["Logger", "basename"]


def basename(filepath, wo_fmt=False):
    """
    获取文件路径的基础名称

    参数:
        filepath (str): 完整的文件路径
        wo_fmt (bool): 是否去除文件扩展名，默认False

    返回:
        str: 文件的基础名称
            - wo_fmt=False: 返回包含扩展名的文件名（如"test.txt"）
            - wo_fmt=True: 返回去除扩展名的文件名（如"test"）

    示例:
        >>> basename("/home/user/test.txt")
        'test.txt'
        >>> basename("/home/user/test.txt", wo_fmt=True)
        'test'
    """
    bname = os.path.basename(filepath)
    if wo_fmt:
        bname = '.'.join(bname.split('.')[:-1])
    return bname


class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    自定义的按时间轮转的日志文件处理器
    继承自TimedRotatingFileHandler，修复了原类在处理夏令时(DST)切换时的时间计算问题
    确保日志文件按指定时间间隔正确轮转，文件名包含轮转周期的起始时间戳
    """

    def doRollover(self):
        """
        执行日志文件轮转操作
        核心逻辑：
        1. 关闭当前日志文件流
        2. 计算轮转周期的起始时间（而非当前时间）作为文件名后缀
        3. 处理夏令时(DST)切换导致的时间偏移
        4. 重命名当前日志文件并创建新文件
        5. 删除超过备份数量的旧日志文件
        6. 计算下一次轮转时间并处理DST调整
        """
        # 关闭当前打开的日志文件流
        if self.stream:
            self.stream.close()
            self.stream = None

        # 获取当前时间和轮转时间点
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]  # 当前是否为夏令时
        t = self.rolloverAt - self.interval  # 轮转周期的起始时间

        # 计算轮转文件的时间戳（处理UTC和本地时间）
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]  # 轮转周期起始时是否为夏令时

            # 处理夏令时切换导致的时间偏移
            if dstNow != dstThen:
                # 夏令时切换补偿：+1小时或-1小时
                addend = 3600 if dstNow else -3600
                timeTuple = time.localtime(t + addend)

        # 构建轮转后的日志文件名（原文件名+时间戳）
        dfn = self.rotation_filename(
            self.baseFilename + "." + time.strftime("%Y-%m-%d_%H-%M-%S", timeTuple)
        )

        # 如果目标文件已存在则先删除
        if os.path.exists(dfn):
            os.remove(dfn)

        # 重命名当前日志文件为轮转文件名
        self.rotate(self.baseFilename, dfn)

        # 删除超过备份数量的旧日志文件
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)

        # 重新打开新的日志文件（非延迟模式）
        if not self.delay:
            self.stream = self._open()

        # 计算下一次轮转时间
        newRolloverAt = self.computeRollover(currentTime)

        # 确保下一次轮转时间在当前时间之后
        while newRolloverAt <= currentTime:
            newRolloverAt += self.interval

        # 处理午夜/每周轮转时的夏令时调整
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]  # 下次轮转时是否为夏令时
            if dstNow != dstAtRollover:
                # 根据夏令时切换方向调整时间
                addend = -3600 if not dstNow else 3600
                newRolloverAt += addend

        # 更新下一次轮转时间
        self.rolloverAt = newRolloverAt


# ------------------------------------------------------------------------------
#   Logger 核心日志类
# ------------------------------------------------------------------------------
class Logger(logging.Logger):
    """
    增强版日志记录器类
    继承自logging.Logger，封装了控制台输出和按时间轮转的文件输出功能
    特性：
    - 自动创建日志目录
    - 同时输出日志到控制台和文件
    - 支持按时间自动切割日志文件
    - 自定义日志格式，包含时间、级别、文件名、行号等信息
    - 自动管理日志文件备份数量

    属性:
        logname (str): 日志器名称
        logdir (str/None): 日志文件保存目录，None则只输出到控制台
        error_id (int): 错误计数ID，初始为0
    """

    def __init__(self, logname, logdir=None, when='H', backupCount=24 * 7):
        """
        初始化日志器

        参数:
            logname (str): 日志器名称
            logdir (str/None): 日志文件保存目录，None表示不输出到文件
            when (str): 日志文件轮转时间单位，可选值：
                - 'S' : 秒
                - 'M' : 分钟
                - 'H' : 小时（默认）
                - 'D' : 天
                - 'W0'-'W6' : 星期几（0=周一）
                - 'midnight' : 午夜
            backupCount (int): 保留的日志文件备份数量，默认24*7=168个（按小时轮转保留7天）
        """
        # 初始化基础属性
        self.logname = logname
        self.logdir = logdir

        # 创建日志目录（如果指定）
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)

        # 错误计数ID（可用于自定义错误处理）
        self.error_id = 0

        # 定义日志格式：时间-级别-名称-文件名-行号: 日志内容
        formatter = logging.Formatter(
            "%(asctime)s-%(levelname)s-%(name)s-%(filename)s-%(lineno)d: %(message)s")

        # 创建文件处理器（如果指定了日志目录）
        filehandler = None
        if logdir is not None:
            # 构建日志文件路径
            logfile = os.path.join(logdir, "%s.log" % (logname))
            # 创建自定义的按时间轮转的文件处理器
            filehandler = CustomTimedRotatingFileHandler(
                logfile, when=when, backupCount=backupCount)
            filehandler.setLevel(logging.INFO)  # 文件日志级别为INFO
            filehandler.setFormatter(formatter)

        # 创建控制台处理器
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.INFO)  # 控制台日志级别为INFO
        streamhandler.setFormatter(formatter)

        # 初始化父类Logger
        super(Logger, self).__init__(logname)
        self.setLevel(logging.INFO)  # 日志器基础级别为INFO

        # 添加处理器到日志器
        if filehandler is not None:
            self.addHandler(filehandler)
        self.addHandler(streamhandler)

        # 记录日志器初始化信息
        if logdir is not None:
            self.info("Logger \'{}\' will be written at {}".format(
                self.logname, logfile))