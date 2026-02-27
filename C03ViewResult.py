# =================================================================
# C03ViewResult.py
# 双向同化效果展示 + Embedding 解耦可视化
# 对应研究内容 2.1.3: 机理模型约束下的 Embedding 空间正则化与双向校正
#
# 演示目标:
#   1. 综合对比: 机理模型 vs 数据驱动 vs 物理约束 (三线并列)
#   2. Embedding 解耦可视化: 热图显示 16 个 Embedding 维度与 LAI/CHL/BIO 的相关系数
#   3. 双向同化动画: "数据校正 ↔ 机理校正" 交替迭代, 逐步收敛
#   4. 误差分析: 分区段统计 (观测区 / 缺失区) 的 RMSE 对比
#
# 操作: 加载 crop_result.json → 自动绘制对比图 → 点击"双向同化"观看动画
# =================================================================

import tkinter as tk
from tkinter import messagebox
import numpy as np
import json
import time
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

KEYS = ['LAI', 'CHL', 'BIO']
LABELS = {'LAI': 'LAI (m²/m²)', 'CHL': 'Chlorophyll (μg/cm²)', 'BIO': 'Biomass (t/ha)'}
VMAX = {'LAI': 8.0, 'CHL': 80.0, 'BIO': 15.0}
COLORS_MECH = '#888888'
COLORS_DD = '#e53935'
COLORS_PC = '#1e88e5'


class ViewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("C03 - 双向同化效果展示 + Embedding 解耦可视化")
        self.root.geometry("1500x920")

        self.data = None
        self.is_running = False
        self.setup_ui()

    def setup_ui(self):
        # 顶部
        top = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        tk.Label(top, text="结果文件:", font=("SimSun", 10)).pack(side=tk.LEFT, padx=4)
        self.file_var = tk.StringVar(value="crop_result.json")
        tk.Entry(top, textvariable=self.file_var, width=18, font=("SimSun", 10)).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="加载并绘制", command=self.load_and_draw,
                  bg="#4caf50", fg="white", font=("SimSun", 10, "bold")).pack(side=tk.LEFT, padx=10)

        self.assim_btn = tk.Button(top, text="启动双向同化演示", command=self.start_assimilation,
                                   bg="#9c27b0", fg="white", font=("SimSun", 10, "bold"), state=tk.DISABLED)
        self.assim_btn.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(top, text="", font=("Consolas", 9))
        self.status_label.pack(side=tk.LEFT, padx=15)

        # 主内容: 左右分栏
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 左栏: 3 行对比图 + 误差柱状图
        left = tk.Frame(paned)
        paned.add(left, width=900)
        self.fig_left, self.axes_left = plt.subplots(4, 1, figsize=(9, 8),
                                                      gridspec_kw={'height_ratios': [1, 1, 1, 0.7]})
        self.fig_left.tight_layout(pad=2.0, h_pad=1.5)
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=left)
        self.canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 右栏: Embedding 相关性热图 + 双向同化
        right = tk.Frame(paned)
        paned.add(right, width=550)
        self.fig_right, self.axes_right = plt.subplots(2, 1, figsize=(5.5, 8),
                                                        gridspec_kw={'height_ratios': [1, 1]})
        self.fig_right.tight_layout(pad=2.5, h_pad=3.0)
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=right)
        self.canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ==================== 加载 ====================
    def load_and_draw(self):
        try:
            with open(self.file_var.get(), 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.draw_comparison()
            self.draw_embedding_heatmap()
            self.assim_btn.config(state=tk.NORMAL)
            self.status_label.config(text="数据已加载, 可启动双向同化")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ==================== 左栏: 三参数对比 + 误差 ====================
    def draw_comparison(self):
        days = np.arange(1, 366)
        gaps = self.data['gaps']
        obs = self.data['observations']
        mech = self.data['mechanism_curves']
        dd = self.data['data_driven']
        pc = self.data['physics_constrained']

        # 构建缺失区掩码
        gap_mask = np.zeros(365, dtype=bool)
        for g in gaps:
            gap_mask[max(g[0] - 1, 0):min(g[1], 365)] = True

        rmse_dd_obs, rmse_dd_gap, rmse_pc_obs, rmse_pc_gap = [], [], [], []

        for i, k in enumerate(KEYS):
            ax = self.axes_left[i]
            ax.clear()

            m = np.array(mech[k])
            d = np.array(dd[k])
            p = np.array(pc[k])

            for g in gaps:
                ax.axvspan(g[0], g[1], alpha=0.2, color='#bdbdbd')

            ax.plot(days, m, '--', color=COLORS_MECH, linewidth=1.5, label='机理模型', alpha=0.7)
            ax.plot(days, d, color=COLORS_DD, linewidth=1.6, label='纯数据驱动', alpha=0.85)
            ax.plot(days, p, color=COLORS_PC, linewidth=1.6, label='物理约束', alpha=0.85)

            obs_d = np.array(obs['days'])
            obs_v = np.array(obs[k])
            ax.scatter(obs_d, obs_v, c='black', s=18, zorder=6, label='观测', marker='x', linewidths=1)

            ax.set_ylabel(LABELS[k], fontsize=9)
            ax.grid(True, alpha=0.2)
            if i == 0:
                ax.legend(fontsize=7, loc='upper left', ncol=4)

            # 计算分区段 RMSE (以机理模型为参考真值)
            rmse_dd_obs.append(np.sqrt(np.mean((d[~gap_mask] - m[~gap_mask]) ** 2)))
            rmse_dd_gap.append(np.sqrt(np.mean((d[gap_mask] - m[gap_mask]) ** 2)))
            rmse_pc_obs.append(np.sqrt(np.mean((p[~gap_mask] - m[~gap_mask]) ** 2)))
            rmse_pc_gap.append(np.sqrt(np.mean((p[gap_mask] - m[gap_mask]) ** 2)))

        # 误差柱状图
        ax_err = self.axes_left[3]
        ax_err.clear()
        x = np.arange(3)
        w = 0.18
        ax_err.bar(x - 1.5 * w, rmse_dd_obs, w, label='DD (观测区)', color=COLORS_DD, alpha=0.6)
        ax_err.bar(x - 0.5 * w, rmse_dd_gap, w, label='DD (缺失区)', color=COLORS_DD, alpha=1.0)
        ax_err.bar(x + 0.5 * w, rmse_pc_obs, w, label='PC (观测区)', color=COLORS_PC, alpha=0.6)
        ax_err.bar(x + 1.5 * w, rmse_pc_gap, w, label='PC (缺失区)', color=COLORS_PC, alpha=1.0)
        ax_err.set_xticks(x)
        ax_err.set_xticklabels(KEYS)
        ax_err.set_ylabel('RMSE vs Mechanism', fontsize=8)
        ax_err.set_title('Error: Observed Zone vs Gap Zone', fontsize=9, fontname='Times New Roman')
        ax_err.legend(fontsize=6.5, ncol=4)
        ax_err.grid(True, alpha=0.2)

        self.fig_left.tight_layout(pad=1.5, h_pad=1.2)
        self.canvas_left.draw()

    # ==================== 右上: Embedding 相关性热图 ====================
    def draw_embedding_heatmap(self):
        emb_dd = np.array(self.data['embeddings_dd'])  # [365, 16]
        emb_pc = np.array(self.data['embeddings_pc'])
        mech = self.data['mechanism_curves']
        phys = np.stack([mech[k] for k in KEYS], axis=1)  # [365, 3]

        ax = self.axes_right[0]
        ax.clear()

        # 计算相关系数矩阵: [embed_dim, 3] for both models
        n_dim = emb_pc.shape[1]
        corr_dd = np.zeros((n_dim, 3))
        corr_pc = np.zeros((n_dim, 3))
        for d in range(n_dim):
            for p in range(3):
                corr_dd[d, p] = np.corrcoef(emb_dd[:, d], phys[:, p])[0, 1]
                corr_pc[d, p] = np.corrcoef(emb_pc[:, d], phys[:, p])[0, 1]

        # 并排显示: 左半 DD, 右半 PC
        combined = np.hstack([corr_dd, np.full((n_dim, 1), np.nan), corr_pc])  # [16, 7]
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(combined.T, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')

        ax.set_yticks(range(7))
        ax.set_yticklabels(['DD-LAI', 'DD-CHL', 'DD-BIO', '─────',
                            'PC-LAI', 'PC-CHL', 'PC-BIO'], fontsize=7)
        ax.set_xticks(range(n_dim))
        ax.set_xticklabels([f'd{i}' for i in range(n_dim)], fontsize=6)
        ax.set_xlabel('Embedding Dimension', fontsize=8, fontname='Times New Roman')
        ax.set_title('Embedding-Physical Param Correlation\n(DD=Data-Driven, PC=Physics-Constrained)',
                     fontsize=9, fontname='Times New Roman')
        self.fig_right.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

        self.fig_right.tight_layout(pad=2.0, h_pad=3.0)
        self.canvas_right.draw()

    # ==================== 右下: 双向同化动画 ====================
    def start_assimilation(self):
        if self.is_running:
            return
        self.is_running = True
        self.assim_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_assimilation, daemon=True).start()

    def _run_assimilation(self):
        """
        双向同化演示:
          从机理模型出发, 交替进行:
            (A) 数据校正: 在有观测的位置向观测值靠拢
            (B) 机理校正: 整体向 Logistic 形状回拉 + 平滑
          迭代 ~12 轮, 逐步收敛到"兼顾数据与机理"的最优曲线
        """
        mech = {k: np.array(self.data['mechanism_curves'][k]) for k in KEYS}
        obs_days = np.array(self.data['observations']['days']) - 1  # 转为 0-indexed
        obs_vals = {k: np.array(self.data['observations'][k]) for k in KEYS}
        gaps = self.data['gaps']

        current = {k: mech[k].copy() for k in KEYS}
        history = {k: [mech[k].copy()] for k in KEYS}

        n_iter = 12
        for it in range(1, n_iter + 1):
            if not self.is_running:
                break

            alpha = 0.35  # 数据校正强度
            beta = 0.15   # 机理回拉强度

            for k in KEYS:
                c = current[k].copy()

                # (A) 数据校正: 向观测值靠拢
                for j, d in enumerate(obs_days):
                    if 0 <= d < 365:
                        c[d] = (1 - alpha) * c[d] + alpha * obs_vals[k][j]

                # 平滑 (简易高斯核)
                kernel = np.exp(-0.5 * (np.arange(21) - 10) ** 2 / 3.0 ** 2)
                kernel /= kernel.sum()
                c = np.convolve(c, kernel, mode='same')

                # (B) 机理校正: 向 Logistic 形状回拉
                c = (1 - beta) * c + beta * mech[k]

                # 物理极值裁剪
                c = np.clip(c, 0, VMAX[k])

                current[k] = c
                history[k].append(c.copy())

            self.root.after(0, lambda it_=it, h={k: [x.copy() for x in v] for k, v in history.items()}:
                            self._draw_assimilation(it_, h))
            time.sleep(0.5)

        self.is_running = False
        self.root.after(0, lambda: self.assim_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.status_label.config(text="双向同化完成"))

    def _draw_assimilation(self, iteration, history):
        ax = self.axes_right[1]
        ax.clear()

        days = np.arange(1, 366)
        gaps = self.data['gaps']
        k = 'LAI'  # 以 LAI 为主要展示对象

        for g in gaps:
            ax.axvspan(g[0], g[1], alpha=0.2, color='#bdbdbd')

        # 机理模型 (虚线)
        ax.plot(days, history[k][0], '--', color=COLORS_MECH, linewidth=1.2, alpha=0.5, label='机理模型')

        # 历史迭代轨迹 (渐深)
        n_hist = len(history[k])
        for i in range(1, n_hist):
            alpha_val = 0.15 + 0.6 * (i / n_hist)
            ax.plot(days, history[k][i], color='#7b1fa2', linewidth=0.8, alpha=alpha_val)

        # 当前结果 (粗线)
        ax.plot(days, history[k][-1], color='#7b1fa2', linewidth=2.2, label=f'同化 iter={iteration}')

        # 观测点
        obs_d = np.array(self.data['observations']['days'])
        obs_v = np.array(self.data['observations'][k])
        ax.scatter(obs_d, obs_v, c='black', s=20, zorder=6, marker='x', linewidths=1, label='观测')

        ax.set_ylabel(LABELS[k], fontsize=9)
        ax.set_xlabel('Day of Year', fontsize=9, fontname='Times New Roman')
        ax.set_title(f'Bidirectional Assimilation (iter {iteration}/{12})',
                     fontsize=9, fontname='Times New Roman')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.3, VMAX[k] * 1.05)

        self.fig_right.tight_layout(pad=2.0, h_pad=3.0)
        self.canvas_right.draw()


if __name__ == "__main__":
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1500x920+{(sw - 1500) // 2}+{(sh - 920) // 2}")
    app = ViewApp(root)

    def on_close():
        app.is_running = False
        plt.close('all')
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
