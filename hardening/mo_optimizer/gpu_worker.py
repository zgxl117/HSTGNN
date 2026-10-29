import multiprocessing as mp
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np

def _worker_main(request_q: mp.Queue, result_q: mp.Queue, sim_props: dict, batch_size: int, config: dict = None):
    """
    GPU worker process: 构造 GPUAcceleratedEvaluator 并循环处理评估请求。
    请求格式: (task_id, 'evaluate_population', genes_batch)
    返回格式: (task_id, list_of_results)
    """
    try:
        # 延迟导入以避免在主进程初始化 CUDA
        from .gpu_evaluator import GPUAcceleratedEvaluator
    except Exception as e:
        result_q.put(('__init_error__', str(e)))
        return

    # 构造一个轻量模拟器对象（SimpleNamespace），供 GPUAcceleratedEvaluator 使用
    sim_ns = SimpleNamespace()
    # 将预传的属性赋值
    sim_ns.all_reg_properties = sim_props.get('all_reg_properties', {})
    sim_ns.SETFF_indices = sim_props.get('SETFF_indices', None)
    sim_ns.K = int(sim_props.get('K', len(sim_ns.all_reg_properties.get('area', []))))
    # 【关键修复】：添加 total_num_registers 和 total_sensitivity_sum（用于敏感性计算）
    sim_ns.total_num_registers = int(sim_props.get('total_num_registers', sim_ns.K))
    sim_ns.total_sensitivity_sum = float(sim_props.get('total_sensitivity_sum', 0.0))
    # 原始基准值
    setattr(sim_ns, 'Area_orig', sim_props.get('Area_orig', sim_props.get('design_area', 1e5)))
    setattr(sim_ns, 'Power_orig', sim_props.get('Power_orig', sim_props.get('design_power', 50.0)))
    setattr(sim_ns, 'WNS_orig', sim_props.get('WNS_orig', sim_props.get('design_slack', 0.05)))
    setattr(sim_ns, 'sens_orig', sim_props.get('sens_orig', sim_props.get('design_sensitivity', 1e-5)))

    # 实例化 GPU 评估器，传入配置
    try:
        # 在worker进程中明确设置cuda设备
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 使用第一个GPU
        gpu_eval = GPUAcceleratedEvaluator(sim_ns, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=batch_size, config=config or {})
    except Exception:
        # 尝试不传 device
        gpu_eval = GPUAcceleratedEvaluator(sim_ns, device=None, batch_size=batch_size, config=config or {})

    result_q.put(('__ready__', None))

    # 处理循环
    while True:
        try:
            task = request_q.get()
        except Exception:
            time.sleep(0.01)
            continue

        if task is None:
            break

        task_id, cmd, payload = task
        if cmd == 'evaluate_population':
            # payload: genes_batch (numpy array)
            try:
                genes_batch = payload
                area, power, slack, ser = gpu_eval.evaluate_batch_gpu(genes_batch)
                # 返回 list of tuples per individual
                results = list(zip(area.tolist(), power.tolist(), slack.tolist(), ser.tolist()))
                result_q.put((task_id, results))
            except Exception as e:
                result_q.put((task_id, ('__error__', str(e))))
        elif cmd == 'shutdown':
            break
        else:
            result_q.put((task_id, ('__unknown_cmd__', cmd)))


class GPUWorker:
    """主进程侧的 GPU worker 客户端封装"""
    def __init__(self, sim_props: dict, batch_size: int = 512, config: dict = None):
        # Manager proxies can deadlock during Windows spawn. Native queues use
        # one worker process and avoid the extra manager server process.
        self.ctx = mp.get_context('spawn')
        self.request_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.batch_size = batch_size
        # 设置为非daemon进程，确保CUDA能正常初始化
        self.proc = self.ctx.Process(
            target=_worker_main,
            args=(self.request_q, self.result_q, sim_props, batch_size, config or {}),
            daemon=True,
        )
        self.proc.start()
        self._next_id = 1

        start = time.time()
        while True:
            try:
                status, payload = self.result_q.get(timeout=0.2)
            except Exception:
                if not self.proc.is_alive():
                    raise RuntimeError('GPU worker exited during initialization')
                if time.time() - start > 60.0:
                    self.proc.terminate()
                    self.proc.join(timeout=5)
                    raise TimeoutError('GPU worker initialization timeout')
                continue
            if status == '__ready__':
                break
            if status == '__init_error__':
                self.proc.join(timeout=5)
                raise RuntimeError(f'GPU worker initialization failed: {payload}')

    def evaluate_population(self, genes_batch: np.ndarray, timeout: float = 30.0) -> List[Tuple[float, float, float, float]]:
        task_id = f"task_{self._next_id}"
        self._next_id += 1
        self.request_q.put((task_id, 'evaluate_population', genes_batch))

        start = time.time()
        while True:
            try:
                res_id, payload = self.result_q.get(timeout=0.1)
            except Exception:
                if time.time() - start > timeout:
                    raise TimeoutError('GPUWorker evaluate_population timeout')
                continue

            if res_id == task_id:
                if isinstance(payload, tuple) and payload[0] == '__error__':
                    raise RuntimeError(payload[1])
                return payload
            else:
                # 放回队列（其他任务）
                self.result_q.put((res_id, payload))

    def shutdown(self):
        """优雅地关闭GPU worker进程"""
        if getattr(self, "_closed", False):
            return
        self._closed = True

        # 尽量让子进程自己退出，忽略已关闭管道/文件错误
        try:
            self.request_q.put((None, 'shutdown', None))
        except Exception:
            pass

        try:
            # 等待进程退出（最多等待5秒）
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join()
        except Exception as e:
            try:
                print(f"GPU worker关闭时出错: {e}")
            except Exception:
                pass

        # 关闭 manager（Windows 下可能抛 FileNotFoundError，忽略即可）
        try:
            for queue in (self.request_q, self.result_q):
                queue.close()
                queue.join_thread()
        except FileNotFoundError:
            pass
        except Exception as e:
            try:
                print(f"GPU worker关闭时出错: {e}")
            except Exception:
                pass
