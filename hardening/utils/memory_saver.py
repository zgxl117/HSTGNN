# utils/memory_saver.py
import gc
import psutil
import numpy as np
import torch
from typing import List
import sys

class MemorySaver:
    """内存节省器"""
    
    def __init__(self, memory_threshold=0.8):
        """
        参数:
        ----------
        memory_threshold : float
            内存使用阈值，超过此阈值触发清理（0.8 = 80%）
        """
        self.memory_threshold = memory_threshold
        self.total_memory = psutil.virtual_memory().total
        
    def check_and_clean(self, force=False):
        """检查并清理内存"""
        memory_info = psutil.virtual_memory()
        used_percent = memory_info.percent / 100.0
        
        ## print(f"[内存监控] 使用率: {used_percent:.1%} ({memory_info.used/1024**3:.1f}GB/{self.total_memory/1024**3:.1f}GB)")
        
        if used_percent > self.memory_threshold or force:
            ## print(f"[内存监控] 内存使用过高，进行清理...")
            self.force_cleanup()
            return True
        return False
    
    def force_cleanup(self):
        """强制内存清理"""
        original_memory = psutil.virtual_memory().used
        
        # 1. 清理Python垃圾回收
        collected = gc.collect()
        ## print(f"  GC清理: 回收 {collected} 个对象")
        
        # 2. 清理numpy缓存
        try:
            import numpy as np
            np.bytes0  # 触发numpy内部清理
        except:
            pass
        
        # 3. 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ## print(f"  GPU缓存已清理")
        
        # 4. 清理Python模块缓存
        for module in list(sys.modules.keys()):
            if module.startswith('temp_') or module.startswith('_'):
                del sys.modules[module]
        
        # 5. 报告清理效果
        new_memory = psutil.virtual_memory().used
        saved = (original_memory - new_memory) / 1024**3
        ## print(f"  清理释放: {saved:.2f}GB")
        
    def monitor_memory_usage(self, interval=10):
        """监控内存使用"""
        import threading
        import time
        
        def monitor():
            while True:
                self.check_and_clean()
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread

# 全局内存节省器实例
memory_saver = MemorySaver(memory_threshold=0.80)  # 80%阈值