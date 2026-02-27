# =================================================================
# C02Train.py
# 机理约束 vs 纯数据驱动 对比训练程序
# 对应研究内容 2.1.3: 机理模型约束下的 Embedding 空间正则化与双向校正
#
# 演示目标 (这是 C 系列最核心的展示):
#   1. 同一网络架构、同一数据，左右对比训练:
#      左 = 纯数据驱动 (L_recon only)
#      右 = 物理信息约束 (L_recon + λ·L_mechanism)
#   2. 在数据缺失期 (60 天):
#      - 纯数据驱动模型: 振荡 / 越界 / 崩溃
#      - 机理约束模型: 平稳穿过缺失区, 符合 Logistic 形状
#   3. 物理信息损失函数由三部分组成:
#      L_mechanism = L_bounds (物理极值) + L_smooth (变化率) + L_shape (Logistic 形状)
#   4. Embedding 空间中特定维度与 LAI/CHL/BIO 解耦对应
#
# 操作: 加载 crop_data.json → 开始训练 → 实时观看对比 → 存储结果
# =================================================================

import tkinter as tk
from tkinter import messagebox
import numpy as np
import json
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

KEYS = ['LAI', 'CHL', 'BIO']
VMAX = {'LAI': 8.0, 'CHL': 80.0, 'BIO': 15.0}
COLORS = {'LAI': '#2e7d32', 'CHL': '#1565c0', 'BIO': '#e65100'}


class CropEmbeddingNet(nn.Module):
    """
    作物 Embedding 网络: 从时间 t → 16 维 Embedding → 3 个物理参数 (LAI, CHL, BIO)
    使用 Fourier 随机特征编码捕获时间序列的高频变化
    """
    def __init__(self, embed_dim=16, hidden=128, n_freq=48):
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer('B', torch.randn(1, n_freq) * 8.0)
        self.encoder = nn.Sequential(
            nn.Linear(n_freq * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, t):
        proj = t @ self.B
        feat = torch.cat([torch.sin(proj * np.pi), torch.cos(proj * np.pi)], dim=1)
        emb = self.encoder(feat)
        out = self.decoder(emb)
        return out, emb


def mechanism_loss(pred_all, mech_tensor):
    """
    物理信息约束损失 (核心创新点):
      L_bounds:  超出 LAI∈[0,8], CHL∈[0,80], BIO∈[0,15] 的惩罚
      L_smooth:  相邻时刻变化率过大的惩罚 (单日生长量超限)
      L_shape:   偏离 Logistic 机理形状的轻度惩罚
    """
    lai, chl, bio = pred_all[:, 0], pred_all[:, 1], pred_all[:, 2]
    # 物理极值惩罚
    l_bounds = (F.relu(-lai).mean() + F.relu(lai - 8).mean() +
                F.relu(-chl).mean() + F.relu(chl - 80).mean() +
                F.relu(-bio).mean() + F.relu(bio - 15).mean())
    # 变化率平滑惩罚
    if pred_all.size(0) > 1:
        diff = pred_all[1:] - pred_all[:-1]
        l_smooth = (diff ** 2).mean()
    else:
        l_smooth = torch.tensor(0.0)
    # Logistic 形状引导
    l_shape = F.mse_loss(pred_all, mech_tensor)
    return l_bounds * 8.0 + l_smooth * 3.0 + l_shape * 0.15


class TrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("C02 - 机理约束 vs 纯数据驱动 对比训练")
        self.root.geometry("1450x880")

        self.raw = None
        self.is_running = False
        self.embed_dim = 16

        self.setup_ui()

    def setup_ui(self):
        top = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        tk.Label(top, text="数据:", font=("SimSun", 10)).pack(side=tk.LEFT, padx=4)
        self.file_var = tk.StringVar(value="crop_data.json")
        tk.Entry(top, textvariable=self.file_var, width=18, font=("SimSun", 10)).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="加载数据", command=self.load_data, font=("SimSun", 9)).pack(side=tk.LEFT, padx=8)

        self.train_btn = tk.Button(top, text="开始对比训练", command=self.start_train,
                                   bg="#4caf50", fg="white", font=("SimSun", 10, "bold"), state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = tk.Button(top, text="停止", command=self.stop_train,
                                  bg="#f44336", fg="white", font=("SimSun", 9), state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.epoch_label = tk.Label(top, text="", font=("Consolas", 9))
        self.epoch_label.pack(side=tk.LEFT, padx=15)

        tk.Button(top, text="存储结果", command=self.save_results,
                  bg="#2196f3", fg="white", font=("SimSun", 9)).pack(side=tk.RIGHT, padx=8)
        self.res_file_var = tk.StringVar(value="crop_result.json")
        tk.Entry(top, textvariable=self.res_file_var, width=16, font=("SimSun", 9)).pack(side=tk.RIGHT, padx=3)

        # 2×2 子图: 上行=曲线拟合, 下行=Loss
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 7.5))
        self.fig.tight_layout(pad=2.5, h_pad=2.0, w_pad=3.0)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 日志
        self.log_text = tk.Text(self.root, height=6, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 8))
        self.log_text.pack(fill=tk.X, padx=8, pady=(0, 5))

    def log(self, msg):
        self.log_text.insert(tk.END, time.strftime("[%H:%M:%S] ") + msg + "\n")
        self.log_text.see(tk.END)

    # ==================== 数据加载 ====================
    def load_data(self):
        try:
            with open(self.file_var.get(), 'r', encoding='utf-8') as f:
                self.raw = json.load(f)
            n = len(self.raw['observations']['days'])
            self.log(f"成功加载数据。观测点 {n} 个, 缺失期 {self.raw['gaps']}")
            self.train_btn.config(state=tk.NORMAL)
            messagebox.showinfo("成功", "数据加载完成")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ==================== 训练 ====================
    def start_train(self):
        if self.is_running:
            return
        self.is_running = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self._train_loop, daemon=True).start()

    def stop_train(self):
        self.is_running = False

    def _train_loop(self):
        obs_days = np.array(self.raw['observations']['days'])
        obs_vals = np.stack([self.raw['observations'][k] for k in KEYS], axis=1)  # [N, 3]

        # 归一化天数到 [0, 1]
        t_obs = torch.tensor(obs_days / 365.0, dtype=torch.float32).unsqueeze(1)
        y_obs = torch.tensor(obs_vals, dtype=torch.float32)

        t_all = torch.linspace(0, 1, 365).unsqueeze(1)
        mech_np = np.stack([self.raw['mechanism_curves'][k] for k in KEYS], axis=1)
        mech_tensor = torch.tensor(mech_np, dtype=torch.float32)

        # 两个模型: 共享相同的架构和初始化种子
        torch.manual_seed(42)
        model_dd = CropEmbeddingNet(embed_dim=self.embed_dim)
        torch.manual_seed(42)
        model_pc = CropEmbeddingNet(embed_dim=self.embed_dim)

        opt_dd = optim.Adam(model_dd.parameters(), lr=0.003)
        opt_pc = optim.Adam(model_pc.parameters(), lr=0.003)
        sched_dd = optim.lr_scheduler.StepLR(opt_dd, 250, 0.5)
        sched_pc = optim.lr_scheduler.StepLR(opt_pc, 250, 0.5)

        n_epochs = 800
        loss_hist_dd, loss_hist_pc, loss_hist_recon, loss_hist_mech = [], [], [], []

        self.log(f"开始训练: {n_epochs} 轮 | Embed={self.embed_dim}d | 左=纯数据驱动 右=机理约束")
        self.log("=" * 60)

        self.result_dd = None
        self.result_pc = None
        self.emb_dd = None
        self.emb_pc = None

        for ep in range(n_epochs):
            if not self.is_running:
                break

            # ---- 纯数据驱动 ----
            model_dd.train()
            pred_dd, _ = model_dd(t_obs)
            loss_dd = F.mse_loss(pred_dd, y_obs)
            opt_dd.zero_grad()
            loss_dd.backward()
            opt_dd.step()
            sched_dd.step()

            # ---- 机理约束 ----
            model_pc.train()
            pred_pc, _ = model_pc(t_obs)
            l_recon = F.mse_loss(pred_pc, y_obs)
            pred_all_pc, emb_all_pc = model_pc(t_all)
            l_mech = mechanism_loss(pred_all_pc, mech_tensor)
            loss_pc = l_recon + 0.5 * l_mech
            opt_pc.zero_grad()
            loss_pc.backward()
            opt_pc.step()
            sched_pc.step()

            loss_hist_dd.append(loss_dd.item())
            loss_hist_pc.append(loss_pc.item())
            loss_hist_recon.append(l_recon.item())
            loss_hist_mech.append(l_mech.item())

            if ep % 25 == 0 or ep == n_epochs - 1:
                # 全域推理
                model_dd.eval()
                model_pc.eval()
                with torch.no_grad():
                    r_dd, e_dd = model_dd(t_all)
                    r_pc, e_pc = model_pc(t_all)
                self.result_dd = r_dd.numpy()
                self.result_pc = r_pc.numpy()
                self.emb_dd = e_dd.numpy()
                self.emb_pc = e_pc.numpy()

                self.root.after(0, lambda e=ep, ld=loss_dd.item(), lp=loss_pc.item(),
                                hd=list(loss_hist_dd), hp=list(loss_hist_pc),
                                hr=list(loss_hist_recon), hm=list(loss_hist_mech):
                                self._update_display(e, ld, lp, hd, hp, hr, hm))
                if ep % 100 == 0:
                    self.log(f"Epoch {ep:4d} | DD Loss: {loss_dd.item():.5f} | "
                             f"PC Total: {loss_pc.item():.5f} (recon={l_recon.item():.5f} mech={l_mech.item():.5f})")

            time.sleep(0.005)

        self.log("=" * 60)
        self.log("训练完成。请观察左右对比: 缺失区域内的预测差异。")
        self.is_running = False
        self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def _update_display(self, epoch, loss_dd, loss_pc, hist_dd, hist_pc, hist_recon, hist_mech):
        self.epoch_label.config(text=f"Epoch {epoch}  |  DD: {loss_dd:.5f}  PC: {loss_pc:.5f}")
        days = np.arange(1, 366)
        obs_days = self.raw['observations']['days']
        gaps = self.raw['gaps']
        mech = self.raw['mechanism_curves']

        # ---- 上行: 曲线拟合 ----
        titles = ['纯数据驱动 (Data-Driven Only)', '物理信息约束 (Physics-Informed)']
        results = [self.result_dd, self.result_pc]
        for col in range(2):
            ax = self.axes[0, col]
            ax.clear()
            res = results[col]
            for g in gaps:
                ax.axvspan(g[0], g[1], alpha=0.22, color='#bdbdbd')
            for i, k in enumerate(KEYS):
                ax.plot(days, mech[k], '--', color=COLORS[k], alpha=0.5, linewidth=1.2)
                ax.plot(days, res[:, i], color=COLORS[k], linewidth=1.8, label=k)
                obs_v = self.raw['observations'][k]
                ax.scatter(obs_days, obs_v, c='red', s=15, zorder=5, alpha=0.6)
            ax.set_title(titles[col], fontsize=10)
            ax.set_ylim(-5, 85)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.legend(fontsize=7, loc='upper left')
            # 缺失区域标注
            for g in gaps:
                ax.annotate('数据缺失', xy=((g[0] + g[1]) / 2, -3), fontsize=7,
                            ha='center', color='gray')

        # ---- 下行: Loss 曲线 ----
        ax_ld = self.axes[1, 0]
        ax_ld.clear()
        ax_ld.plot(hist_dd, color='#e53935', linewidth=1.2)
        ax_ld.set_title('Data-Driven Loss', fontsize=9)
        ax_ld.set_xlabel('Epoch')
        ax_ld.set_ylabel('MSE')
        ax_ld.grid(True, alpha=0.2)

        ax_lp = self.axes[1, 1]
        ax_lp.clear()
        ax_lp.plot(hist_pc, color='#1e88e5', linewidth=1.2, label='L_total')
        ax_lp.plot(hist_recon, color='#43a047', linewidth=1.0, alpha=0.7, label='L_recon')
        ax_lp.plot(hist_mech, color='#fb8c00', linewidth=1.0, alpha=0.7, label='λ·L_mech')
        ax_lp.set_title('Physics-Constrained Loss (Breakdown)', fontsize=9)
        ax_lp.set_xlabel('Epoch')
        ax_lp.legend(fontsize=7)
        ax_lp.grid(True, alpha=0.2)

        self.fig.tight_layout(pad=2.0, h_pad=2.0)
        self.canvas_fig.draw()

    # ==================== 存储 ====================
    def save_results(self):
        if self.result_dd is None:
            messagebox.showwarning("提示", "请先完成训练")
            return
        data = {
            "_comment": ("C02 对比训练结果。data_driven: 纯数据驱动预测 (365×3); "
                         "physics_constrained: 物理约束预测 (365×3); "
                         "embeddings_*: 对应的 16 维 Embedding 向量 (365×16); "
                         "mechanism_curves: 原始 Logistic 机理曲线; observations: 原始观测; gaps: 缺失期"),
            "data_driven": {k: self.result_dd[:, i].tolist() for i, k in enumerate(KEYS)},
            "physics_constrained": {k: self.result_pc[:, i].tolist() for i, k in enumerate(KEYS)},
            "embeddings_dd": self.emb_dd.tolist(),
            "embeddings_pc": self.emb_pc.tolist(),
            "mechanism_curves": self.raw['mechanism_curves'],
            "observations": self.raw['observations'],
            "gaps": self.raw['gaps'],
        }
        try:
            fn = self.res_file_var.get()
            with open(fn, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", f"结果已保存至 {fn}")
        except Exception as e:
            messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1450x880+{(sw - 1450) // 2}+{(sh - 880) // 2}")
    app = TrainApp(root)

    def on_close():
        app.is_running = False
        plt.close('all')
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
