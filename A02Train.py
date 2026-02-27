import tkinter as tk
from tkinter import messagebox, filedialog
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
# 设置字体为宋体 (SimSun) 和 Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块或乱码的问题
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# =================================================================
# 程序名称: A02Train.py
# 功能描述: 使用 Neural ODE 训练 NDVI 演化模型，并外推求解 1-200 时刻数据
# 主要包需求: torch, torchdiffeq, numpy, matplotlib, tkinter
# =================================================================

class ODEFunc(nn.Module):
    """
    定义常微分方程的右端项 f(y, t) = dy/dt
    输入: 当前状态 y (10x10 矩阵), 时间 t
    输出: 变化率 dy/dt (10x10 矩阵)
    """
    def __init__(self, hidden_dim=128):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(100, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 100),
        )

    def forward(self, t, y):
        # y shape: [batch, 100]
        return self.net(y)

class NeuralODETrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("NDVI 神经常微分方程 (Neural ODE) 训练程序")
        self.root.geometry("1100x700")

        # 数据状态
        self.times = None
        self.grids = None
        self.model = None
        self.is_training = False
        self.losses = []

        self.setup_ui()

    def setup_ui(self):
        # 1. 顶部标题
        tk.Label(self.root, text="NDVI 演化建模：神经常微分方程 (Neural ODE)", font=("SimSun", 16, "bold")).pack(pady=10)

        # 主容器
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # 2. 左侧控制面板
        self.control_frame = tk.Frame(self.main_frame, width=250, bd=1, relief=tk.SOLID)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20), pady=10)

        tk.Label(self.control_frame, text="控制面板", font=("SimSun", 12, "bold")).pack(pady=10)

        self.load_btn = tk.Button(self.control_frame, text="读取 raster.json", command=self.load_data, width=20, font=("SimSun", 9))
        self.load_btn.pack(pady=5)

        self.train_btn = tk.Button(self.control_frame, text="开始训练", command=self.start_training_thread, width=20, bg="#4caf50", fg="white", font=("SimSun", 9))
        self.train_btn.pack(pady=5)

        self.stop_btn = tk.Button(self.control_frame, text="停止训练", command=self.stop_training, width=20, state=tk.DISABLED, font=("SimSun", 9))
        self.stop_btn.pack(pady=5)

        # 进度信息
        self.info_frame = tk.LabelFrame(self.control_frame, text="训练状态", padx=10, pady=10, font=("SimSun", 10, "bold"))
        self.info_frame.pack(pady=20, fill=tk.X, padx=10)

        self.epoch_label = tk.Label(self.info_frame, text="迭代次数: 0", font=("SimSun", 9))
        self.epoch_label.pack(anchor="w")
        self.loss_label = tk.Label(self.info_frame, text="当前 Loss: -", font=("SimSun", 9))
        self.loss_label.pack(anchor="w")

        # 保存结果
        self.save_frame = tk.Frame(self.control_frame)
        self.save_frame.pack(pady=20, fill=tk.X, padx=10)
        
        self.save_btn = tk.Button(self.save_frame, text="存储 1-200 预测结果", command=self.save_predictions, width=20, bg="#2196f3", fg="white", state=tk.DISABLED, font=("SimSun", 9))
        self.save_btn.pack(pady=(0, 5))
        
        tk.Label(self.save_frame, text="文件名:", font=("SimSun", 9)).pack(side=tk.LEFT)
        self.result_filename_var = tk.StringVar(value="result.json")
        self.result_filename_entry = tk.Entry(self.save_frame, textvariable=self.result_filename_var, width=15, font=("SimSun", 9))
        self.result_filename_entry.pack(side=tk.LEFT, padx=5)

        # 3. 右侧图表显示区
        self.plot_frame = tk.Frame(self.main_frame, bd=1, relief=tk.SUNKEN)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Training Loss Curve")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("MSE Loss")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        try:
            filename = "raster.json"
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sequences = data['sequences']
            # 将时间点归一化到 [0, 1] 范围内更有利于 ODE 训练，或者保持原始范围
            # 这里我们使用原始时间点除以 200，映射到 [0, 1]
            self.times = torch.tensor([s['time'] / 200.0 for s in sequences], dtype=torch.float32)
            self.grids = torch.tensor([s['grid'] for s in sequences], dtype=torch.float32)
            
            # 打平 10x10 为 100 维
            self.grids = self.grids.view(-1, 100)
            
            messagebox.showinfo("成功", f"成功读取 {len(sequences)} 个时间点的数据")
            self.losses = []
            self.update_plot()
        except Exception as e:
            messagebox.showerror("错误", f"读取失败: {str(e)}")

    def start_training_thread(self):
        if self.times is None:
            messagebox.showwarning("警告", "请先读取 raster.json")
            return
        
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()

    def train_model(self):
        # 初始化模型
        self.model = ODEFunc()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # 初始状态 y0 为第一个时刻的 grid
        y0 = self.grids[0:1] # [1, 100]
        target = self.grids # [N, 100]
        t_target = self.times # [N]

        epoch = 0
        while self.is_training:
            optimizer.zero_grad()
            
            # 使用 ODE 解理器积分
            # 从 t_target[0] 开始积到所有目标时间点
            pred_y = odeint(self.model, y0, t_target, method='dopri5')
            # pred_y shape: [N, 1, 100] -> [N, 100]
            pred_y = pred_y.squeeze(1)
            
            loss = criterion(pred_y, target)
            loss.backward()
            optimizer.step()
            
            epoch += 1
            self.losses.append(loss.item())
            
            # 每隔 10 次迭代更新一次 UI
            if epoch % 10 == 0:
                self.epoch_label.config(text=f"迭代次数: {epoch}")
                self.loss_label.config(text=f"当前 Loss: {loss.item():.6f}")
                self.update_plot()
            
            # 防止 UI 假死
            if epoch % 100 == 0:
                time.sleep(0.01)

    def stop_training(self):
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL)
        messagebox.showinfo("训练停止", "训练已由用户手动停止")

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Training Loss Curve")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("MSE Loss")
        if self.losses:
            self.ax.plot(self.losses, color='red')
            self.ax.set_yscale('log') # 使用对数坐标更清晰
        self.canvas.draw()

    def save_predictions(self):
        """对 1-200 每一个整数时刻进行求解并存储结果"""
        if self.model is None:
            return
            
        try:
            # 准备 1-200 的时间序列
            t_full = torch.linspace(0, 1.0, 200) # 对应原始时刻 1-200 (归一化到 0-1)
            y0 = self.grids[0:1]
            
            with torch.no_grad():
                pred_full = odeint(self.model, y0, t_full, method='dopri5')
                pred_full = pred_full.squeeze(1).numpy() # [200, 100]
                pred_full = pred_full.reshape(200, 10, 10)
            
            # 准备保存数据
            results = {
                "_comment": [
                    "这是使用 Neural ODE 训练后的外推结果",
                    "模型通过已知的几个时间点学习了 NDVI 的演变微分方程",
                    "这里展示了从时刻 1 到时刻 200 的完整演变过程",
                    "数据格式为 200 个 10x10 的矩阵序列"
                ],
                "metadata": {
                    "total_steps": 200,
                    "resolution": "1.0 unit per step",
                    "learned_dynamics": "Neural ODE (dopri5 solver)"
                },
                "predictions": []
            }
            
            for i in range(200):
                results["predictions"].append({
                    "time": i + 1,
                    "grid": pred_full[i].tolist()
                })
            
            save_path = self.result_filename_var.get()
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
            messagebox.showinfo("成功", f"预测结果已保存至 {save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    # 居中显示
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1100) // 2
    y = (screen_height - 700) // 2
    root.geometry(f"1100x700+{x}+{y}")
    
    app = NeuralODETrainer(root)
    root.mainloop()
