import tkinter as tk
from tkinter import messagebox, ttk
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

# 设置字体为宋体 (SimSun) 和 Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False 

# =================================================================
# 程序名称: A03viewResult.py
# 功能描述: 对比展示 raster.json (原始) 和 result.json (预测) 的结果
# 主要功能: 3D 曲线演化图、时间切片对比、10x10 栅格还原
# =================================================================

class NDVIResultViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("NDVI 演化结果对比可视化")
        self.root.geometry("1200x800")
        
        self.raster_data = None
        self.result_data = None
        self.current_time_idx = 0 # 0-199 对应 1-200 时刻
        
        self.setup_ui()

    def setup_ui(self):
        # 1. 顶部文件输入区
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        tk.Label(top_frame, text="原始文件:", font=("SimSun", 10)).pack(side=tk.LEFT)
        self.raster_file_var = tk.StringVar(value="raster.json")
        tk.Entry(top_frame, textvariable=self.raster_file_var, width=20, font=("SimSun", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Label(top_frame, text="结果文件:", font=("SimSun", 10)).pack(side=tk.LEFT, padx=(10, 0))
        self.result_file_var = tk.StringVar(value="result.json")
        tk.Entry(top_frame, textvariable=self.result_filename_var if hasattr(self, 'result_filename_var') else self.result_file_var, width=20, font=("SimSun", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="加载并刷新", command=self.load_all_data, font=("SimSun", 10), bg="#e1e1e1").pack(side=tk.LEFT, padx=15)

        # 2. 底部时间拉动条 - 【关键修改：先 pack 底部，确保其可见】
        self.bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)
        
        tk.Label(self.bottom_frame, text="时间步拖动 (1-200):", font=("SimSun", 11, "bold"), bg="#f0f0f0").pack(side=tk.TOP, pady=(5, 0))
        
        # 用于放置特殊时间标记的 Canvas
        self.marks_canvas = tk.Canvas(self.bottom_frame, height=20, bg="#f0f0f0", highlightthickness=0)
        self.marks_canvas.pack(side=tk.TOP, fill=tk.X, padx=10)

        self.time_var = tk.IntVar(value=1)
        self.time_slider = tk.Scale(self.bottom_frame, from_=1, to=200, orient=tk.HORIZONTAL, 
                                   variable=self.time_var, command=self.on_time_change, 
                                   showvalue=True, length=1000, sliderlength=30,
                                   font=("Times New Roman", 10), bg="#f0f0f0", highlightthickness=0)
        self.time_slider.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 10))

        # 绑定窗口大小改变事件，自动更新标记位置
        self.bottom_frame.bind("<Configure>", lambda e: self.update_special_marks())

        # 3. 中部主显示区 - 【后 pack 中部，并设置 expand=True 填充剩余空间】
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20)

        # 右侧栅格对比区 - 【关键修改：先 pack 右侧，确保其优先占据空间】
        self.right_frame = tk.Frame(main_frame, width=320, bd=1, relief=tk.SOLID)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 为右侧增加滚动条，防止窗口高度不够时被遮挡
        self.right_canvas = tk.Canvas(self.right_frame, width=300, highlightthickness=0)
        self.right_scrollbar = ttk.Scrollbar(self.right_frame, orient=tk.VERTICAL, command=self.right_canvas.yview)
        self.right_scrollable_inner = tk.Frame(self.right_canvas)
        
        self.right_scrollable_inner.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        )
        self.right_canvas.create_window((0, 0), window=self.right_scrollable_inner, anchor="nw")
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)
        
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.right_scrollable_inner, text="数据切片视图", font=("SimSun", 12, "bold")).pack(pady=10)
        
        # 自定义黄色到绿色的 Colormap
        from matplotlib.colors import LinearSegmentedColormap
        self.custom_cmap = LinearSegmentedColormap.from_list('yg', [(1, 1, 0), (0, 0.39, 0)], N=256)

        # 预测结果方格
        self.pred_grid_frame = tk.LabelFrame(self.right_scrollable_inner, text="预测结果 (result.json)", font=("SimSun", 10))
        self.pred_grid_frame.pack(pady=5, padx=10, fill=tk.BOTH)
        self.fig_pred, self.ax_pred = plt.subplots(figsize=(2.8, 2.8)) # 稍微调小尺寸
        self.fig_pred.tight_layout() 
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=self.pred_grid_frame)
        self.canvas_pred.get_tk_widget().pack(pady=5)
        tk.Label(self.pred_grid_frame, text="[模型推理序列]", fg="gray", font=("SimSun", 9)).pack()

        # 原始数据方格 (对比用)
        self.orig_grid_frame = tk.LabelFrame(self.right_scrollable_inner, text="原始输入 (raster.json)", font=("SimSun", 10))
        self.orig_grid_frame.pack(pady=5, padx=10, fill=tk.BOTH)
        self.fig_orig, self.ax_orig = plt.subplots(figsize=(2.8, 2.8)) # 稍微调小尺寸
        self.fig_orig.tight_layout()
        self.canvas_orig = FigureCanvasTkAgg(self.fig_orig, master=self.orig_grid_frame)
        self.canvas_orig.get_tk_widget().pack(pady=5)
        self.orig_match_label = tk.Label(self.orig_grid_frame, text="[非训练采样点]", fg="gray", font=("SimSun", 9))
        self.orig_match_label.pack()

        # 左侧 3D 曲线图 - 【后 pack 左侧】
        self.left_frame = tk.Frame(main_frame, bd=1, relief=tk.SUNKEN)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 初始化 3D 图表对象
        self.fig_3d = plt.figure(figsize=(8, 6))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.view_init(elev=20, azim=-60)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.left_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_all_data(self):
        try:
            # 加载 raster.json
            with open(self.raster_file_var.get(), 'r', encoding='utf-8') as f:
                self.raster_data = json.load(f)
            # 创建原始时间到 grid 的映射
            self.raster_map = {s['time']: np.array(s['grid']) for s in self.raster_data['sequences']}
            
            # 记录特殊时间点
            self.special_times = sorted(list(self.raster_map.keys()))
            
            # 加载 result.json
            with open(self.result_file_var.get(), 'r', encoding='utf-8') as f:
                self.result_data = json.load(f)
            
            self.all_predictions = [np.array(s['grid']) for s in self.result_data['predictions']]
            self.all_times = [s['time'] for s in self.result_data['predictions']]
            
            # 更新拖动条刻度
            self.time_slider.config(tickinterval=0) # 先清除默认大刻度
            # 在标签中显示特殊时间点，或者通过 tooltip，但在 Tkinter Scale 中，最简单的做法是动态更新一个 Label
            self.update_special_marks()

            messagebox.showinfo("成功", f"数据加载完成，包含 {len(self.special_times)} 个原始采样点")
            self.init_3d_plot()
            self.on_time_change(self.time_var.get())
            
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")

    def update_special_marks(self):
        """在拖动条上方通过 Canvas 绘制特殊时间点的标记（倒三角符号）"""
        if not hasattr(self, 'special_times') or not self.special_times:
            return
            
        self.marks_canvas.delete("all")
        self.root.update_idletasks() # 确保布局已完成
        
        # 获取 Scale 的实际显示宽度
        slider_width = self.time_slider.winfo_width()
        # Scale 组件左右会有一定的内边距 (通常是 sliderlength 的一半)
        # 加上我们 pack 时的 padx=20
        margin = 20 + 15 # 基本边距 + 滑块一半宽度
        track_width = slider_width - 2 * margin
        
        if track_width <= 0: return # 防止极端情况

        for t in self.special_times:
            # 计算 t (1-200) 在轨道上的横坐标
            x = margin + (t - 1) * (track_width / 199)
            # 绘制绿色倒三角
            self.marks_canvas.create_polygon(x-5, 0, x+5, 0, x, 10, fill="#2e7d32", outline="white")
            # 绘制文字
            self.marks_canvas.create_text(x, 15, text=str(t), font=("Times New Roman", 8, "bold"), fill="#2e7d32")

        if hasattr(self, 'marks_label'):
            self.marks_label.destroy()
        
        marks_text = "原始采样点时刻: " + ", ".join(map(str, self.special_times))
        self.marks_label = tk.Label(self.bottom_frame, text=marks_text, 
                                   font=("SimSun", 9, "bold"), fg="#2e7d32", bg="#f0f0f0")
        self.marks_label.pack(side=tk.TOP, pady=(0, 5))

    def get_ndvi_color(self, val):
        """
        颜色映射逻辑：
        +1 -> 深绿色 (0, 100, 0) -> (0, 0.39, 0)
        -1 -> 黄色 (255, 255, 0) -> (1, 1, 0)
        """
        n = (val + 1) / 2 # 0 to 1
        r = 1.0 - n
        g = (1.0 - n) + 0.39 * n
        b = 0
        return (r, g, b)

    def init_3d_plot(self):
        self.ax_3d.clear()
        self.ax_3d.set_title("NDVI Time Evolution (3D Curves)", fontname='Times New Roman', fontsize=14)
        self.ax_3d.set_xlabel("Time (1-200)", fontname='Times New Roman')
        self.ax_3d.set_ylabel("Pixel Path (0-99)", fontname='Times New Roman')
        self.ax_3d.set_zlabel("NDVI Value", fontname='Times New Roman')
        self.ax_3d.set_zlim(-1.1, 1.1)

        # 提取 100 条曲线
        # data_3d: [Time, 100]
        self.data_3d = np.array([p.flatten() for p in self.all_predictions])
        self.times_arr = np.array(self.all_times)
        
        # 绘制 100 条曲线
        for i in range(100):
            y_vals = self.data_3d[:, i]
            color = self.get_ndvi_color(np.mean(y_vals))
            # 增加 linewidth 使曲线更宽/更明显
            self.ax_3d.plot(self.times_arr, np.full_like(self.times_arr, i), y_vals, 
                          color=color, alpha=0.6, linewidth=2.4)

        # 初始化切片线
        self.slice_lines = []
        self.canvas_3d.draw()

    def on_time_change(self, val):
        if not hasattr(self, 'all_predictions') or not self.all_predictions:
            return
            
        t = int(val)
        idx = t - 1
        if idx >= len(self.all_predictions): idx = len(self.all_predictions) - 1
        
        # 1. 更新 3D 图中的切片线
        # 先删除旧的切片线
        while self.slice_lines:
            line = self.slice_lines.pop()
            line.remove()
        
        # 在当前时间点 t 绘制一条横跨所有像素的线
        current_y_vals = self.data_3d[idx, :]
        line, = self.ax_3d.plot([t]*100, np.arange(100), current_y_vals, 
                               color='red', linewidth=4.0, label='Current Time', zorder=10)
        self.slice_lines.append(line)
        self.canvas_3d.draw_idle()
        
        # 2. 更新右侧预测方格
        grid_pred = self.all_predictions[idx]
        self.ax_pred.clear()
        self.ax_pred.imshow(grid_pred, vmin=-1, vmax=1, cmap=self.custom_cmap)
        self.ax_pred.set_title(f"Predicted T={t}", fontname='Times New Roman')
        self.ax_pred.axis('off')
        self.canvas_pred.draw()

        # 3. 更新右侧原始方格对照
        self.ax_orig.clear()
        if t in self.raster_map:
            grid_orig = self.raster_map[t]
            self.ax_orig.imshow(grid_orig, vmin=-1, vmax=1, cmap=self.custom_cmap)
            self.ax_orig.set_title(f"Original T={t}", fontname='Times New Roman')
            self.orig_match_label.config(text="[匹配采样点]", fg="green", font=("SimSun", 9, "bold"))
        else:
            # 显示一个空的或占位图
            self.ax_orig.text(0.5, 0.5, "No Sampling\nData", ha='center', va='center', fontname='Times New Roman')
            self.ax_orig.set_title(f"Original T={t} (N/A)", fontname='Times New Roman')
            self.orig_match_label.config(text="[非训练采样点]", fg="gray", font=("SimSun", 9))
        
        self.ax_orig.axis('off')
        self.canvas_orig.draw()

if __name__ == "__main__":
    root = tk.Tk()
    # 居中
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1200) // 2
    y = (screen_height - 800) // 2
    root.geometry(f"1200x800+{x}+{y}")
    
    app = NDVIResultViewer(root)
    root.mainloop()
