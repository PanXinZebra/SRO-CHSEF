# =================================================================
# C01CreateData.py
# 作物生长参数数据建立程序
# 对应研究内容 2.1.3: 机理模型约束下的 Embedding 空间正则化与双向校正
#
# 演示目标:
#   1. 展示 Logistic 机理生长模型(含衰老期)作为"先验知识"的角色
#   2. 模拟真实场景下稀疏、含噪、含长期数据缺失的观测
#   3. 为 C02 的"有/无机理约束对比训练"提供输入数据
#
# 操作流程: 调整参数 → 生成/刷新 → 存储 crop_data.json → 供 C02 加载
# =================================================================

import tkinter as tk
from tkinter import messagebox
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 三个核心生物物理参数的默认机理模型配置
PARAM_DEFS = {
    'LAI': {'K': 6.5, 'r': 0.06, 't0': 155, 't_dec': 265, 'r_dec': 0.018,
            'label': 'LAI (m²/m²)', 'color': '#2e7d32', 'vmax': 8.0},
    'CHL': {'K': 58.0, 'r': 0.07, 't0': 145, 't_dec': 250, 'r_dec': 0.022,
            'label': 'Chlorophyll (μg/cm²)', 'color': '#1565c0', 'vmax': 80.0},
    'BIO': {'K': 11.0, 'r': 0.045, 't0': 175, 't_dec': 310, 'r_dec': 0.006,
            'label': 'Biomass (t/ha)', 'color': '#e65100', 'vmax': 15.0},
}


def crop_growth(t, K, r, t0, t_dec, r_dec):
    """带衰老期的作物 Logistic 生长曲线: 上升阶段为 Logistic, 衰老阶段为指数衰减"""
    growth = K / (1.0 + np.exp(-r * (t - t0)))
    decline = np.where(t > t_dec, np.exp(-r_dec * (t - t_dec)), 1.0)
    return np.maximum(growth * decline, 0.0)


class CropDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("C01 - 作物生长参数建立 (Logistic 机理模型 + 稀疏观测模拟)")
        self.root.geometry("1200x800")

        self.days = np.arange(1, 366)
        self.curves = {}
        self.obs = {'days': [], 'LAI': [], 'CHL': [], 'BIO': []}
        self.gaps = [(120, 180), (245, 270)]

        self.setup_ui()
        self.refresh()

    # ==================== UI ====================
    def setup_ui(self):
        left = tk.Frame(self.root, width=285, bd=1, relief=tk.GROOVE)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left.pack_propagate(False)

        tk.Label(left, text="── 机理模型参数 ──", font=("SimSun", 10, "bold")).pack(pady=(8, 4))

        self.param_vars = {}
        for key in ['LAI', 'CHL', 'BIO']:
            p = PARAM_DEFS[key]
            frm = tk.LabelFrame(left, text=p['label'], font=("SimSun", 9))
            frm.pack(fill=tk.X, padx=5, pady=2)
            vd = {}
            for name, default in [('K', p['K']), ('r', p['r']), ('t0', p['t0']),
                                  ('t_dec', p['t_dec']), ('r_dec', p['r_dec'])]:
                row = tk.Frame(frm)
                row.pack(fill=tk.X, padx=2)
                tk.Label(row, text=f"{name}:", font=("SimSun", 8), width=5, anchor='e').pack(side=tk.LEFT)
                v = tk.DoubleVar(value=default)
                tk.Entry(row, textvariable=v, width=8, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
                vd[name] = v
            self.param_vars[key] = vd

        tk.Label(left, text="── 观测数据模拟 ──", font=("SimSun", 10, "bold")).pack(pady=(8, 4))
        of = tk.Frame(left)
        of.pack(fill=tk.X, padx=8)
        tk.Label(of, text="观测点数:", font=("SimSun", 9)).grid(row=0, column=0, sticky='e')
        self.n_obs_var = tk.IntVar(value=20)
        tk.Spinbox(of, from_=5, to=80, textvariable=self.n_obs_var, width=5,
                   font=("SimSun", 9)).grid(row=0, column=1, padx=3)
        tk.Label(of, text="噪声系数:", font=("SimSun", 9)).grid(row=1, column=0, sticky='e')
        self.noise_var = tk.DoubleVar(value=0.08)
        tk.Entry(of, textvariable=self.noise_var, width=6, font=("SimSun", 9)).grid(row=1, column=1, padx=3)

        tk.Label(left, text="── 数据缺失期 (云遮挡) ──", font=("SimSun", 10, "bold")).pack(pady=(8, 2))
        self.gap_lb = tk.Listbox(left, height=3, font=("SimSun", 9))
        self.gap_lb.pack(fill=tk.X, padx=8)
        self._refresh_gap_list()

        gf = tk.Frame(left)
        gf.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(gf, text="起:", font=("SimSun", 8)).pack(side=tk.LEFT)
        self.gs_var = tk.IntVar(value=120)
        tk.Entry(gf, textvariable=self.gs_var, width=5, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
        tk.Label(gf, text="止:", font=("SimSun", 8)).pack(side=tk.LEFT)
        self.ge_var = tk.IntVar(value=180)
        tk.Entry(gf, textvariable=self.ge_var, width=5, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(gf, text="添加", font=("SimSun", 8), command=self.add_gap).pack(side=tk.LEFT, padx=2)
        tk.Button(gf, text="删除", font=("SimSun", 8), command=self.del_gap).pack(side=tk.LEFT, padx=2)

        tk.Button(left, text="生成 / 刷新", command=self.refresh,
                  bg="#4caf50", fg="white", font=("SimSun", 10, "bold")).pack(fill=tk.X, padx=10, pady=(15, 5))

        tk.Label(left, text="── 文件 ──", font=("SimSun", 10, "bold")).pack(pady=(10, 2))
        ff = tk.Frame(left)
        ff.pack(fill=tk.X, padx=8)
        self.file_var = tk.StringVar(value="crop_data.json")
        tk.Entry(ff, textvariable=self.file_var, width=18, font=("SimSun", 9)).pack(side=tk.LEFT, padx=2)
        bf = tk.Frame(left)
        bf.pack(fill=tk.X, padx=8, pady=3)
        tk.Button(bf, text="存储", command=self.save_data, bg="#2196f3", fg="white",
                  font=("SimSun", 9)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(bf, text="加载", command=self.load_data, bg="#ff9800", fg="white",
                  font=("SimSun", 9)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # 右侧: 三个子图 (LAI / CHL / BIO)
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        self.fig.tight_layout(pad=2.5, h_pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ==================== 缺失期管理 ====================
    def _refresh_gap_list(self):
        self.gap_lb.delete(0, tk.END)
        for gs, ge in self.gaps:
            self.gap_lb.insert(tk.END, f"Day {gs} ~ {ge}  ({ge - gs} 天)")

    def add_gap(self):
        gs, ge = self.gs_var.get(), self.ge_var.get()
        if 1 <= gs < ge <= 365:
            self.gaps.append((gs, ge))
            self.gaps.sort()
            self._refresh_gap_list()

    def del_gap(self):
        sel = self.gap_lb.curselection()
        if sel:
            del self.gaps[sel[0]]
            self._refresh_gap_list()

    # ==================== 数据生成 ====================
    def refresh(self):
        self._gen_curves()
        self._gen_observations()
        self._update_plots()

    def _gen_curves(self):
        t = self.days.astype(float)
        for key in ['LAI', 'CHL', 'BIO']:
            pv = self.param_vars[key]
            self.curves[key] = crop_growth(
                t, pv['K'].get(), pv['r'].get(), pv['t0'].get(),
                pv['t_dec'].get(), pv['r_dec'].get())

    def _gen_observations(self):
        n_obs = self.n_obs_var.get()
        noise = self.noise_var.get()
        valid = np.ones(365, dtype=bool)
        for gs, ge in self.gaps:
            valid[max(gs - 1, 0):min(ge, 365)] = False
        vi = np.where(valid)[0]
        if len(vi) < n_obs:
            n_obs = len(vi)
        chosen = np.sort(np.random.choice(vi, n_obs, replace=False))
        self.obs = {'days': (self.days[chosen]).tolist()}
        for key in ['LAI', 'CHL', 'BIO']:
            vals = self.curves[key][chosen]
            noisy = vals + np.random.randn(n_obs) * noise * PARAM_DEFS[key]['K']
            self.obs[key] = np.clip(noisy, 0, PARAM_DEFS[key]['vmax']).tolist()

    # ==================== 可视化 ====================
    def _update_plots(self):
        for i, key in enumerate(['LAI', 'CHL', 'BIO']):
            ax = self.axes[i]
            ax.clear()
            p = PARAM_DEFS[key]
            # 生长季淡绿底色
            ax.axvspan(60, 310, alpha=0.06, color='green')
            # 缺失期灰色区域
            for j, (gs, ge) in enumerate(self.gaps):
                ax.axvspan(gs, ge, alpha=0.28, color='#bdbdbd',
                           label='数据缺失 (云遮挡)' if i == 0 and j == 0 else None)
            # 机理模型曲线
            ax.plot(self.days, self.curves[key], color=p['color'], linewidth=2.2,
                    label='Logistic 机理模型', zorder=3)
            # 观测点
            ax.scatter(self.obs['days'], self.obs[key], c='red', s=28, zorder=5,
                       label='稀疏观测', edgecolors='darkred', linewidths=0.5)
            ax.set_ylabel(p['label'], fontsize=9)
            ax.set_ylim(-0.3, p['vmax'] * 1.05)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(loc='upper left', fontsize=7.5)
            if i == 2:
                ax.set_xlabel('Day of Year (DOY)', fontname='Times New Roman', fontsize=10)
        self.fig.suptitle('Crop Growth: Logistic Mechanism Model + Sparse Noisy Observations',
                          fontname='Times New Roman', fontsize=11, y=0.99)
        self.fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.97])
        self.canvas.draw()

    # ==================== 存储 / 加载 ====================
    def save_data(self):
        params_out = {}
        for key in ['LAI', 'CHL', 'BIO']:
            pv = self.param_vars[key]
            params_out[key] = {n: pv[n].get() for n in ['K', 'r', 't0', 't_dec', 'r_dec']}
        data = {
            "_comment": ("C01 作物生长参数数据。mechanism_params: Logistic 机理模型参数; "
                         "mechanism_curves: 365 天理论曲线; observations: 稀疏含噪观测 "
                         "(排除缺失期); gaps: 数据缺失期 (模拟云遮挡)"),
            "mechanism_params": params_out,
            "mechanism_curves": {k: self.curves[k].tolist() for k in ['LAI', 'CHL', 'BIO']},
            "observations": self.obs,
            "gaps": self.gaps,
            "noise_level": self.noise_var.get()
        }
        try:
            fn = self.file_var.get()
            with open(fn, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", f"数据已保存至 {fn}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def load_data(self):
        try:
            fn = self.file_var.get()
            with open(fn, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in ['LAI', 'CHL', 'BIO']:
                mp = data['mechanism_params'][key]
                for n in ['K', 'r', 't0', 't_dec', 'r_dec']:
                    self.param_vars[key][n].set(mp[n])
                self.curves[key] = np.array(data['mechanism_curves'][key])
            self.obs = data['observations']
            self.gaps = [tuple(g) for g in data['gaps']]
            self.noise_var.set(data.get('noise_level', 0.08))
            self._refresh_gap_list()
            self._update_plots()
            messagebox.showinfo("成功", f"已加载 {fn}")
        except Exception as e:
            messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1200x800+{(sw - 1200) // 2}+{(sh - 800) // 2}")
    app = CropDataApp(root)

    def on_close():
        plt.close('all')
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
