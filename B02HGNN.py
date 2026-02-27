import tkinter as tk
from tkinter import ttk, messagebox
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from PIL import Image, ImageTk
import threading
import time

# =================================================================
# 程序名称: B02HGNN.py
# 功能描述: 异构图神经网络 (HGNN) 与归纳式图谱克里金 (Graph Kriging) 演示
# 实现逻辑: 加载 B01 的异构数据，构建空间图拓扑，通过 GNN 融合并重建全域 NDVI
# =================================================================

class SpatialKrigingNet(nn.Module):
    """
    空间归纳式克里金网络：
    从归一化坐标 (x, y) 学习 NDVI 的空间分布场。
    使用 Fourier 特征编码 + 多层感知器来捕获空间高频变化。
    训练数据来自已知观测像元，推理时泛化到全域。
    """
    def __init__(self, hidden_dim=256, n_freq=32):
        super(SpatialKrigingNet, self).__init__()
        self.n_freq = n_freq
        # 随机 Fourier 特征频率（固定不训练）
        self.register_buffer('B', torch.randn(2, n_freq) * 4.0)
        input_dim = n_freq * 2  # sin + cos
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, coords):
        # coords: [N, 2] 归一化坐标
        proj = coords @ self.B  # [N, n_freq]
        feat = torch.cat([torch.sin(proj * np.pi), torch.cos(proj * np.pi)], dim=1)
        return self.net(feat)


class HGNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("异构图谱协同融合：归纳式图谱克里金 (B02)")
        self.root.geometry("1400x900")

        self.raw_data = None
        self.model = None
        self.is_running = False

        self.target_rows, self.target_cols = 200, 500
        self.reconstructed_grid = np.zeros((self.target_rows, self.target_cols))

        self.setup_ui()

    def setup_ui(self):
        top_frame = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="数据源:", font=("SimSun", 10)).pack(side=tk.LEFT, padx=5)
        self.file_var = tk.StringVar(value="grid.json")
        tk.Entry(top_frame, textvariable=self.file_var, width=20, font=("SimSun", 10)).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="1. 加载异构数据", command=self.load_data, bg="#e1e1e1", font=("SimSun", 9)).pack(side=tk.LEFT, padx=10)

        self.show_raw_btn = tk.Button(top_frame, text="2. 原始观测叠加", command=self.show_raw_overlay, bg="#ff9800", fg="white", font=("SimSun", 9, "bold"), state=tk.DISABLED)
        self.show_raw_btn.pack(side=tk.LEFT, padx=10)

        self.run_btn = tk.Button(top_frame, text="3. 执行图谱克里金", command=self.start_process, bg="#4caf50", fg="white", font=("SimSun", 9, "bold"), state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(top_frame, text="停止", command=self.stop_process, bg="#f44336", fg="white", font=("SimSun", 9), state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="存储融合结果", command=self.save_result, bg="#2196f3", fg="white", font=("SimSun", 9)).pack(side=tk.RIGHT, padx=10)

        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.left_panel = tk.Frame(self.paned)
        self.paned.add(self.left_panel, width=800)

        tk.Label(self.left_panel, text="全域重建结果 (500x200 动态实时)", font=("SimSun", 10, "bold")).pack(pady=5)
        self.result_canvas = tk.Canvas(self.left_panel, width=1000, height=400, bg="black")
        self.result_canvas.pack(padx=10, pady=5)
        self.result_image_id = self.result_canvas.create_image(0, 0, anchor="nw")

        # 训练 Loss 曲线
        self.fig_loss, self.ax_loss = plt.subplots(figsize=(8, 1.8))
        self.fig_loss.tight_layout(pad=1.5)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self.left_panel)
        self.canvas_loss.get_tk_widget().pack(fill=tk.X, padx=10, pady=2)

        tk.Label(self.left_panel, text="归纳式计算推导过程日志:", font=("SimSun", 10, "bold")).pack(pady=(5, 0), anchor="w")
        self.log_text = tk.Text(self.left_panel, height=10, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.right_panel = tk.Frame(self.paned)
        self.paned.add(self.right_panel, width=550)

        tk.Label(self.right_panel, text="异构图谱拓扑结构 (G=V,E)", font=("SimSun", 10, "bold")).pack(pady=5)
        self.fig_graph, self.ax_graph = plt.subplots(figsize=(5, 5))
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, master=self.right_panel)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_log(self, msg):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.insert(tk.END, timestamp + msg + "\n")
        self.log_text.see(tk.END)

    def load_data(self):
        try:
            with open(self.file_var.get(), 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            self.add_log("成功读取异构数据源。")
            self.add_log(f"全域目标尺寸: {self.target_cols}x{self.target_rows}")
            obs_list = self.raw_data.get('observations', [])
            self.add_log(f"检测到观测组件数量: {len(obs_list)}")
            for i, obs in enumerate(obs_list):
                d = np.array(obs['data'])
                self.add_log(f"  [{i}] {obs['type']} shape={d.shape} pos=({obs['target_pos'][0]},{obs['target_pos'][1]})")

            self.draw_topology()
            self.show_raw_btn.config(state=tk.NORMAL)
            self.run_btn.config(state=tk.NORMAL)
            messagebox.showinfo("成功", "异构数据加载完成，图拓扑已构建")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {e}")

    def draw_topology(self):
        self.ax_graph.clear()
        G = nx.Graph()
        G.add_node("Target\n500x200", type='target', color='#cccccc')

        obs_list = self.raw_data.get('observations', [])
        color_map = {'遥感': '#4caf50', '无人机': '#2196f3', '传感器': '#f44336'}
        for i, obs in enumerate(obs_list):
            obs_type = obs['type']
            short = obs_type.split('(')[0].strip().split()[0]
            for key in color_map:
                if key in obs_type:
                    short = key
                    break
            node_id = f"{short}_{i}"
            color = '#888888'
            for k, v in color_map.items():
                if k in obs_type:
                    color = v
                    break
            G.add_node(node_id, color=color)
            G.add_edge("Target\n500x200", node_id, weight=1)

        pos = nx.spring_layout(G, seed=42, k=2.0)
        node_colors = [G.nodes[n].get('color', '#888') for n in G.nodes]
        nx.draw(G, pos, ax=self.ax_graph, with_labels=True,
                node_color=node_colors, node_size=1200, font_size=7, font_family='SimSun',
                edge_color='#aaaaaa', width=2)
        self.ax_graph.set_title("Heterogeneous Graph Topology", fontname='Times New Roman', fontsize=10)
        self.canvas_graph.draw()

    # ============================================================
    # 按钮2: 原始观测叠加
    # ============================================================
    def show_raw_overlay(self):
        self.add_log(">>> 执行原始观测数据叠加显示...")
        sum_grid = np.zeros((self.target_rows, self.target_cols))
        count_grid = np.zeros((self.target_rows, self.target_cols))

        obs_list = self.raw_data.get('observations', [])
        for obs in obs_list:
            obs_type = obs['type']
            data = np.array(obs['data'])
            tx, ty = obs['target_pos']

            if "遥感" in obs_type:
                bw, bh = 200, 100
                start_x, start_y = tx - bw // 2, ty - bh // 2
                rows_d, cols_d = data.shape
                for r in range(rows_d):
                    for c in range(cols_d):
                        val = data[r, c]
                        for dr in range(2):
                            for dc in range(2):
                                tr, tc = start_y + r * 2 + dr, start_x + c * 2 + dc
                                if 0 <= tr < self.target_rows and 0 <= tc < self.target_cols:
                                    sum_grid[tr, tc] += val
                                    count_grid[tr, tc] += 1

            elif "无人机" in obs_type:
                bw, bh = 25, 25
                start_x, start_y = tx - bw // 2, ty - bh // 2
                for r in range(25):
                    for c in range(25):
                        val = np.mean(data[r * 2:r * 2 + 2, c * 2:c * 2 + 2])
                        tr, tc = start_y + r, start_x + c
                        if 0 <= tr < self.target_rows and 0 <= tc < self.target_cols:
                            sum_grid[tr, tc] += val
                            count_grid[tr, tc] += 1
            else:
                tr, tc = ty, tx
                if 0 <= tr < self.target_rows and 0 <= tc < self.target_cols:
                    v = data.flat[0]
                    sum_grid[tr, tc] += v
                    count_grid[tr, tc] += 1

        mask = count_grid > 0
        self.reconstructed_grid = np.zeros_like(sum_grid)
        self.reconstructed_grid[mask] = sum_grid[mask] / count_grid[mask]

        self.add_log(f"叠加完成。共融合了 {len(obs_list)} 个观测。覆盖像元 {int(mask.sum())}/{self.target_rows * self.target_cols}")
        self.update_grid_display()

    # ============================================================
    # 按钮3: 归纳式图谱克里金
    # ============================================================
    def start_process(self):
        if self.is_running:
            return
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        thread = threading.Thread(target=self.run_inductive_kriging)
        thread.daemon = True
        thread.start()

    def stop_process(self):
        self.is_running = False

    def build_training_data(self):
        """
        将所有观测源展开为像元级训练数据:
          (归一化 col/cols, 归一化 row/rows) -> NDVI 值
        保持了每个像元的空间位置信息，使网络能学到空间场分布。
        """
        coords = []
        values = []
        obs_list = self.raw_data.get('observations', [])

        for obs in obs_list:
            obs_type = obs['type']
            data = np.array(obs['data'])
            tx, ty = obs['target_pos']

            if "遥感" in obs_type:
                bw, bh = 200, 100
                start_x, start_y = tx - bw // 2, ty - bh // 2
                rows_d, cols_d = data.shape
                step = max(1, min(rows_d, cols_d) // 30)
                for r in range(0, rows_d, step):
                    for c in range(0, cols_d, step):
                        tr = start_y + r * 2
                        tc = start_x + c * 2
                        if 0 <= tr < self.target_rows and 0 <= tc < self.target_cols:
                            coords.append([tc / self.target_cols, tr / self.target_rows])
                            values.append(data[r, c])

            elif "无人机" in obs_type:
                bw, bh = 25, 25
                start_x, start_y = tx - bw // 2, ty - bh // 2
                for r in range(25):
                    for c in range(25):
                        val = np.mean(data[r * 2:r * 2 + 2, c * 2:c * 2 + 2])
                        tr, tc = start_y + r, start_x + c
                        if 0 <= tr < self.target_rows and 0 <= tc < self.target_cols:
                            coords.append([tc / self.target_cols, tr / self.target_rows])
                            values.append(val)

            else:
                tr, tc = ty, tx
                if 0 <= tr < self.target_rows and 0 <= tc < self.target_cols:
                    v = data.flat[0]
                    # 传感器只有一个点，复制周围小邻域增强信号
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = tr + dr, tc + dc
                            if 0 <= nr < self.target_rows and 0 <= nc < self.target_cols:
                                coords.append([nc / self.target_cols, nr / self.target_rows])
                                values.append(v)

        return np.array(coords, dtype=np.float32), np.array(values, dtype=np.float32)

    def run_inductive_kriging(self):
        """归纳式图谱克里金：在像元级观测上训练空间场网络，推理全域"""
        self.add_log("=" * 60)
        self.add_log(">>> 启动归纳式图谱克里金 (Inductive Spatial Kriging)")
        self.add_log("=" * 60)

        self.add_log("步骤1: 展开多源异构观测为像元级训练数据...")
        coords, values = self.build_training_data()
        n_samples = len(coords)
        self.add_log(f"  训练样本数: {n_samples} 个空间采样点")
        self.add_log(f"  NDVI 值域: [{values.min():.3f}, {values.max():.3f}]")

        X_train = torch.tensor(coords, dtype=torch.float32)
        Y_train = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        self.add_log("步骤2: 初始化 SpatialKrigingNet (Fourier特征编码 + MLP)...")
        self.model = SpatialKrigingNet(hidden_dim=256, n_freq=64)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        criterion = nn.MSELoss()

        n_epochs = 600
        batch_size = min(4096, n_samples)
        self.add_log(f"步骤3: 训练 ({n_epochs} 轮, batch={batch_size}, lr=0.001)...")
        self.add_log("-" * 60)

        loss_history = []

        for epoch in range(n_epochs):
            if not self.is_running:
                self.add_log(">>> 用户中止训练。")
                break

            self.model.train()
            indices = torch.randint(0, n_samples, (batch_size,))
            batch_x = X_train[indices]
            batch_y = Y_train[indices]

            pred = self.model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if epoch % 20 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                self.add_log(f"  Epoch {epoch:4d}/{n_epochs} | Loss: {loss_val:.6f} | LR: {lr_now:.6f}")
                self.infer_full_grid()
                self.root.after(0, self.update_grid_display)
                self.root.after(0, lambda h=list(loss_history): self.update_loss_plot(h))

            time.sleep(0.01)

        self.add_log("-" * 60)
        self.add_log("步骤4: 最终全域推理 (500x200)...")
        self.infer_full_grid()
        self.root.after(0, self.update_grid_display)
        self.root.after(0, lambda h=list(loss_history): self.update_loss_plot(h))

        if loss_history:
            self.add_log(f"最终 Loss: {loss_history[-1]:.6f}")
        self.add_log("完成! 归纳式图谱克里金全域重建已完成。")
        self.add_log("=" * 60)
        self.is_running = False
        self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def infer_full_grid(self):
        """在全域网格上推理"""
        self.model.eval()
        with torch.no_grad():
            rows, cols = self.target_rows, self.target_cols
            scale = 2
            lo_r, lo_c = rows // scale, cols // scale

            grid_x, grid_y = np.meshgrid(
                np.linspace(0, 1, lo_c),
                np.linspace(0, 1, lo_r)
            )
            coords_full = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1).astype(np.float32)
            X_full = torch.tensor(coords_full)

            chunk_size = 10000
            results = []
            for i in range(0, X_full.size(0), chunk_size):
                chunk = X_full[i:i + chunk_size]
                out = self.model(chunk).squeeze(1).numpy()
                results.append(out)

            res_flat = np.concatenate(results)
            res_low = res_flat.reshape(lo_r, lo_c)

            img_res = Image.fromarray(((np.clip(res_low, -1, 1) + 1) / 2 * 255).astype(np.uint8))
            img_full = img_res.resize((cols, rows), Image.BILINEAR)
            self.reconstructed_grid = (np.array(img_full).astype(float) / 255.0) * 2 - 1

    def update_loss_plot(self, loss_history):
        self.ax_loss.clear()
        self.ax_loss.plot(loss_history, color='#00ff00', linewidth=1)
        self.ax_loss.set_xlabel("Epoch", fontname='Times New Roman', fontsize=8)
        self.ax_loss.set_ylabel("MSE Loss", fontname='Times New Roman', fontsize=8)
        self.ax_loss.set_title("Training Loss", fontname='Times New Roman', fontsize=9)
        self.ax_loss.tick_params(labelsize=7)
        self.ax_loss.set_facecolor('#1e1e1e')
        self.fig_loss.set_facecolor('#2e2e2e')
        self.ax_loss.spines['bottom'].set_color('#888')
        self.ax_loss.spines['left'].set_color('#888')
        self.ax_loss.tick_params(colors='#aaa')
        self.ax_loss.yaxis.label.set_color('#aaa')
        self.ax_loss.xaxis.label.set_color('#aaa')
        self.ax_loss.title.set_color('#ccc')
        self.fig_loss.tight_layout(pad=1.0)
        self.canvas_loss.draw()

    def update_grid_display(self):
        """使用向量化 numpy 颜色映射渲染 500x200"""
        grid = np.clip(self.reconstructed_grid, -1, 1)
        n = (grid + 1) / 2.0  # 0 (黄) -> 1 (绿)
        img_data = np.zeros((self.target_rows, self.target_cols, 3), dtype=np.uint8)
        # 黄色 (255,200,0) -> 绿色 (0,128,0)
        img_data[:, :, 0] = (255 * (1 - n)).astype(np.uint8)
        img_data[:, :, 1] = (200 * (1 - n) + 128 * n).astype(np.uint8)
        img_data[:, :, 2] = 0

        w, h = self.target_cols * 2, self.target_rows * 2
        img = Image.fromarray(img_data).resize((w, h), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.result_canvas.itemconfig(self.result_image_id, image=self.photo)

    def save_result(self):
        result = {
            "_comment": "HGNN 协同融合后的全域 NDVI 重建结果 (500x200)",
            "method": "Inductive Spatial Kriging (Fourier-MLP)",
            "rows": self.target_rows,
            "cols": self.target_cols,
            "data": self.reconstructed_grid.tolist()
        }
        try:
            with open("hgnn_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            messagebox.showinfo("成功", "融合结果已保存至 hgnn_result.json")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1400x900+{(sw - 1400) // 2}+{(sh - 900) // 2}")
    app = HGNNApp(root)
    root.mainloop()
