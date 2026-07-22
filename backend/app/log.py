"""统一日志：stdlib logging，控制台 + 滚动文件双输出，支持分级。

用法：
    from app.log import get_logger
    log = get_logger(__name__)
    log.info("知识库入库 %d 条", n)

服务启动时调用 setup_logging(level) 一次即可（见 app.main.main）。
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"

_CONFIGURED = False

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "info") -> None:
    """配置根日志器（幂等，仅首次生效）。level: debug/info/warning/error。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)
    # 避免把第三方库的冗余日志刷屏（只保留 WARNING 以上）
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    fmt = logging.Formatter(FORMAT, DATEFMT)

    # 控制台
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # 滚动文件（5MB × 3 备份）
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception as e:  # 文件不可写不应阻断服务
        root.warning("日志文件初始化失败，仅输出控制台：%s", e)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
