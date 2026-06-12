"""
安全的 YAML 读写工具模块

提供原子写入和文件锁定功能，防止多进程并发读写导致的文件损坏。

主要功能：
1. 原子写入：先写入临时文件，再原子重命名
2. 文件锁定：使用 fcntl 进行进程级文件锁
3. 重试机制：读取失败时自动重试
"""

import os
import yaml
import fcntl
import tempfile
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 0.1  # 秒
DEFAULT_LOCK_TIMEOUT = 30  # 秒


class YAMLLockError(Exception):
    """YAML 文件锁定错误"""
    pass


class YAMLReadError(Exception):
    """YAML 文件读取错误"""
    pass


class YAMLWriteError(Exception):
    """YAML 文件写入错误"""
    pass


def safe_yaml_read(
    file_path: Union[str, Path],
    retry_count: int = DEFAULT_RETRY_COUNT,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    default: Any = None
) -> Any:
    """
    安全读取 YAML 文件
    
    使用文件锁和重试机制，防止读取到正在写入的不完整文件。
    
    Args:
        file_path: YAML 文件路径
        retry_count: 读取失败时的重试次数
        retry_delay: 重试间隔（秒）
        lock_timeout: 获取锁的超时时间（秒）
        default: 读取失败或文件为空时的默认值
        
    Returns:
        解析后的 YAML 数据，或 default 值
        
    Raises:
        YAMLReadError: 多次重试后仍无法读取
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"YAML文件不存在: {file_path}")
        return default
    
    last_error = None
    
    for attempt in range(retry_count):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 获取共享锁（允许多个读取者）
                _acquire_lock(f, fcntl.LOCK_SH, lock_timeout)
                try:
                    content = f.read()
                    if not content or content.strip() == '':
                        return default
                    
                    # 检查文件是否包含空字节（损坏的标志）
                    if '\x00' in content:
                        raise YAMLReadError(f"文件包含空字节，可能已损坏: {file_path}")
                    
                    data = yaml.safe_load(content)
                    return data if data is not None else default
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except (yaml.YAMLError, YAMLReadError) as e:
            last_error = e
            logger.warning(f"YAML读取失败 (尝试 {attempt + 1}/{retry_count}): {file_path}, 错误: {e}")
            if attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))  # 指数退避
        except IOError as e:
            last_error = e
            logger.warning(f"IO错误 (尝试 {attempt + 1}/{retry_count}): {file_path}, 错误: {e}")
            if attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))
    
    raise YAMLReadError(f"读取YAML文件失败（已重试{retry_count}次）: {file_path}, 最后错误: {last_error}")


def safe_yaml_write(
    file_path: Union[str, Path],
    data: Any,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    allow_unicode: bool = True,
    default_flow_style: bool = False,
    sort_keys: bool = False
) -> None:
    """
    安全写入 YAML 文件
    
    使用原子写入（先写临时文件再重命名）和文件锁，防止写入过程中被其他进程读取到不完整内容。
    
    Args:
        file_path: 目标文件路径
        data: 要写入的数据
        lock_timeout: 获取锁的超时时间（秒）
        allow_unicode: 是否允许 Unicode 字符
        default_flow_style: YAML 流格式
        sort_keys: 是否对键排序
        
    Raises:
        YAMLWriteError: 写入失败
    """
    file_path = Path(file_path)
    
    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成同目录下的临时文件（确保在同一文件系统，以支持原子重命名）
    temp_fd = None
    temp_path = None
    
    try:
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.yaml.tmp',
            prefix=f'.{file_path.stem}_',
            dir=file_path.parent
        )
        
        # 写入临时文件
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            temp_fd = None  # fdopen 接管了文件描述符
            yaml.dump(
                data, f,
                allow_unicode=allow_unicode,
                default_flow_style=default_flow_style,
                sort_keys=sort_keys
            )
            f.flush()
            os.fsync(f.fileno())  # 确保数据写入磁盘
        
        # 如果目标文件存在，先获取排他锁
        if file_path.exists():
            with open(file_path, 'r+', encoding='utf-8') as lock_file:
                _acquire_lock(lock_file, fcntl.LOCK_EX, lock_timeout)
                try:
                    # 原子重命名
                    os.replace(temp_path, file_path)
                    temp_path = None  # 重命名成功，不需要清理
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        else:
            # 文件不存在，直接重命名
            os.replace(temp_path, file_path)
            temp_path = None
            
        logger.debug(f"YAML文件已安全写入: {file_path}")
        
    except Exception as e:
        raise YAMLWriteError(f"写入YAML文件失败: {file_path}, 错误: {e}") from e
    finally:
        # 清理临时文件
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except:
                pass
        if temp_path is not None and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def _acquire_lock(
    file_handle,
    lock_type: int,
    timeout: float
) -> None:
    """
    获取文件锁，带超时
    
    Args:
        file_handle: 文件句柄
        lock_type: 锁类型 (fcntl.LOCK_SH 或 fcntl.LOCK_EX)
        timeout: 超时时间（秒）
        
    Raises:
        YAMLLockError: 获取锁超时
    """
    start_time = time.time()
    
    while True:
        try:
            fcntl.flock(file_handle.fileno(), lock_type | fcntl.LOCK_NB)
            return  # 成功获取锁
        except BlockingIOError:
            if time.time() - start_time > timeout:
                raise YAMLLockError(f"获取文件锁超时（{timeout}秒）")
            time.sleep(0.01)  # 短暂等待后重试


# 便捷函数，兼容原有接口
def load_yaml(file_path: Union[str, Path], default: Any = None) -> Any:
    """加载 YAML 文件（safe_yaml_read 的简化接口）"""
    return safe_yaml_read(file_path, default=default)


def save_yaml(
    file_path: Union[str, Path],
    data: Any,
    **kwargs
) -> None:
    """保存 YAML 文件（safe_yaml_write 的简化接口）"""
    safe_yaml_write(file_path, data, **kwargs)
