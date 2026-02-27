import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import random

# =================================================================
# 程序名称: A01CreateData.py
# 功能描述: 10*10 NDVI 栅格数据建立程序，支持多时间点数据模拟与编辑
# 主要包需求: numpy, tkinter
# =================================================================

class NDVIGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("10*10 NDVI栅格建立程序")
        self.root.geometry("1100x700")

        # 1. 顶部标题区域
        self.header_label = tk.Label(root, text="10*10的NDVI栅格建立程序，时间范围1-200", font=("SimSun", 16, "bold"))
        self.header_label.pack(pady=15)

        # 2. 建立区域：设置表象（时间点）个数
        self.build_frame = tk.Frame(root)
        self.build_frame.pack(pady=5)
        
        tk.Label(self.build_frame, text="状态个数 (2-20):", font=("SimSun", 10)).pack(side=tk.LEFT)
        self.num_items_var = tk.StringVar(value="3")
        self.num_items_entry = tk.Spinbox(self.build_frame, from_=2, to=20, textvariable=self.num_items_var, width=5, font=("SimSun", 10))
        self.num_items_entry.pack(side=tk.LEFT, padx=10)
        
        self.build_btn = tk.Button(self.build_frame, text="建立", command=self.build_workspace, bg="#e1e1e1", width=10, font=("SimSun", 10))
        self.build_btn.pack(side=tk.LEFT, padx=5)

        # 主界面容器（包含左侧工具栏和右侧滚动显示区）
        self.main_container = tk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 3. 左侧控制区：画笔功能
        self.left_frame = tk.Frame(self.main_container, width=150, bd=1, relief=tk.SOLID)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        tk.Label(self.left_frame, text="控制台", font=("SimSun", 12, "bold")).pack(pady=10)
        
        self.brush_active = tk.BooleanVar(value=False)
        self.brush_btn = tk.Checkbutton(self.left_frame, text="画笔 (点击激活)", variable=self.brush_active, 
                                        indicatoron=False, width=15, height=2, bg="#f0f0f0", selectcolor="#90ee90", font=("SimSun", 10))
        self.brush_btn.pack(pady=10, padx=10)
        
        tk.Label(self.left_frame, text="画笔对应值:", font=("SimSun", 9)).pack()
        tk.Label(self.left_frame, text="(+1 绿 | -1 黄)", font=("SimSun", 8), fg="gray").pack()
        
        self.brush_value = tk.DoubleVar(value=1.0)
        # 滑动条：值范围 -1 到 +1
        self.brush_slider = tk.Scale(self.left_frame, from_=1.0, to=-1.0, resolution=0.1, 
                                    variable=self.brush_value, orient=tk.VERTICAL, length=250, 
                                    tickinterval=0.5, showvalue=True, font=("SimSun", 8))
        self.brush_slider.pack(pady=5)

        # 4. 右侧横向滚动区域
        self.right_frame = tk.Frame(self.main_container)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas_scroll = tk.Canvas(self.right_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.right_frame, orient=tk.HORIZONTAL, command=self.canvas_scroll.xview)
        self.scrollable_frame = tk.Frame(self.canvas_scroll)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        )
        
        self.canvas_scroll_win = self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_scroll.configure(xscrollcommand=self.scrollbar.set)
        
        self.canvas_scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # 5. 底部存储区域
        self.bottom_frame = tk.Frame(root, bd=1, relief=tk.GROOVE)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)
        
        self.save_btn = tk.Button(self.bottom_frame, text="存储数据", command=self.save_data, 
                                 bg="#4caf50", fg="white", font=("SimSun", 10, "bold"), width=12)
        self.save_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.load_btn = tk.Button(self.bottom_frame, text="加载数据", command=self.load_data, 
                                 bg="#2196f3", fg="white", font=("SimSun", 10, "bold"), width=12)
        self.load_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(self.bottom_frame, text="保存文件名:", font=("SimSun", 10)).pack(side=tk.LEFT)
        self.filename_var = tk.StringVar(value="raster.json")
        self.filename_entry = tk.Entry(self.bottom_frame, textvariable=self.filename_var, width=40, font=("SimSun", 10))
        self.filename_entry.pack(side=tk.LEFT, padx=10)

        self.grid_items = []

    def load_data(self):
        """从JSON文件加载数据并恢复界面状态"""
        filename = self.filename_var.get()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'sequences' not in data:
                raise ValueError("文件格式错误：未找到 'sequences' 字段")
            
            sequences = data['sequences']
            num = len(sequences)
            
            # 更新状态个数输入框
            self.num_items_var.set(str(num))
            
            # 清除旧的组件
            for item in self.grid_items:
                item.destroy()
            self.grid_items.clear()
            
            # 根据加载的数据创建组件
            for i, seq in enumerate(sequences):
                time_val = seq['time']
                grid_data = np.array(seq['grid'])
                
                item = GridItem(self.scrollable_frame, i, time_val, self)
                item.grid = grid_data
                item.update_canvas()
                item.pack(side=tk.LEFT, padx=15, pady=10)
                self.grid_items.append(item)
                
            messagebox.showinfo("成功", f"成功从 {filename} 加载了 {num} 个时间点的数据")
            
        except FileNotFoundError:
            messagebox.showerror("错误", f"找不到文件: {filename}")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")

    def build_workspace(self):
        """点击建立按钮，初始化列表"""
        # 清除旧的组件
        for item in self.grid_items:
            item.destroy()
        self.grid_items.clear()
        
        try:
            num = int(self.num_items_var.get())
            if not (2 <= num <= 20):
                raise ValueError
        except ValueError:
            messagebox.showwarning("警告", "请输入2-20之间的数字")
            return

        # 随机分配初始时间，并确保顺序
        times = sorted(random.sample(range(1, 201), num))
        
        # 创建多个时间点的矩阵编辑器
        for i in range(num):
            item = GridItem(self.scrollable_frame, i, times[i], self)
            item.pack(side=tk.LEFT, padx=15, pady=10)
            self.grid_items.append(item)

    def save_data(self):
        """将所有时间点的矩阵数据保存为JSON文件"""
        if not self.grid_items:
            messagebox.showwarning("警告", "请先点击'建立'生成数据")
            return
            
        data = {
            "_comment": [
                "NDVI栅格数据存储文件",
                "grid 为 10x10 的二维数组，取值范围 [-1, 1]",
                "time 为该矩阵对应的时间戳，范围 [1, 200]",
                "保证 time[i] < time[i+1]"
            ],
            "metadata": {
                "program": "A01CreateData.py",
                "description": "10*10 NDVI栅格序列",
                "time_range": [1, 200],
                "value_mapping": {"+1": "dark_green", "-1": "yellow"}
            },
            "sequences": []
        }
        
        for item in self.grid_items:
            data["sequences"].append({
                "time": item.time_var.get(),
                "grid": item.grid.tolist()
            })
            
        filename = self.filename_var.get()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("成功", f"数据已成功保存至: {filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

class GridItem(tk.Frame):
    """单个10x10矩阵编辑器组件"""
    def __init__(self, parent, index, initial_time, app):
        super().__init__(parent, bd=2, relief=tk.GROOVE, padx=5, pady=5)
        self.index = index
        self.app = app
        self.grid_size = 10
        self.pixel_size = 15  # 每个方格15*15像元
        
        # 内部数据矩阵 (10x10)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # 界面：序号标题
        tk.Label(self, text=f"时间点序列 {index + 1}", font=("SimSun", 10, "bold")).pack()
        
        # 界面：10x10 绘图画布
        self.canvas = tk.Canvas(self, width=self.grid_size * self.pixel_size, 
                               height=self.grid_size * self.pixel_size, bg="white", highlightthickness=1)
        self.canvas.pack(pady=10)
        self.rects = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # 初始化画布方格
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x1, y1 = c * self.pixel_size, r * self.pixel_size
                x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
                self.rects[r][c] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.get_color(0), outline="#dcdcdc")
        
        # 绑定鼠标绘制事件
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<Button-1>", self.on_draw)
        
        # 按钮控制区
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="随机", command=self.random_grid, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Copy前一个", command=self.copy_from_previous, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="再随机", command=self.randomize_again, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
        
        # 时间控制区
        time_label_frame = tk.Frame(self)
        time_label_frame.pack(fill=tk.X)
        tk.Label(time_label_frame, text="时间点:", font=("SimSun", 8)).pack(side=tk.LEFT)
        self.time_label = tk.Label(time_label_frame, text=str(initial_time), font=("SimSun", 8, "bold"))
        self.time_label.pack(side=tk.RIGHT)

        self.time_var = tk.IntVar(value=initial_time)
        self.time_slider = tk.Scale(self, from_=1, to=200, variable=self.time_var, 
                                   orient=tk.HORIZONTAL, showvalue=False, command=self.validate_time)
        self.time_slider.pack(fill=tk.X, pady=(0, 5))

    def get_color(self, val):
        """
        颜色映射逻辑：
        +1 -> 深绿色 (0, 100, 0)
        -1 -> 黄色 (255, 255, 0)
        0  -> 中间色
        """
        # 归一化 val 到 [0, 1] 范围
        n = (val + 1) / 2
        # 插值计算 R 和 G 分量
        r = int(255 * (1 - n))
        g = int(255 * (1 - n) + 100 * n)
        b = 0
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        return f'#{r:02x}{g:02x}{b:02x}'

    def update_canvas(self):
        """同步数据矩阵到画布显示"""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self.canvas.itemconfig(self.rects[r][c], fill=self.get_color(self.grid[r, c]))

    def on_draw(self, event):
        """画笔绘制：将当前画笔值复制到对应方格"""
        if not self.app.brush_active.get():
            return
        
        c, r = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            val = self.app.brush_value.get()
            self.grid[r, c] = val
            self.canvas.itemconfig(self.rects[r][c], fill=self.get_color(val))

    def random_grid(self):
        """
        随机生成：大部分连续接近绿色的区域，小部分其他。
        使用均值滤波平滑随机噪声，模拟地理空间连续性。
        """
        raw = np.random.uniform(0.5, 1.0, (self.grid_size, self.grid_size))
        for _ in range(3):
            # 简单的 3x3 邻域平滑
            temp = np.pad(raw, 1, mode='edge')
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    raw[r, c] = np.mean(temp[r:r+3, c:c+3])
        
        # 随机添加一些“异常”低值点（黄色区域）
        if random.random() > 0.4:
            for _ in range(random.randint(1, 3)):
                r, c = random.randint(0, 9), random.randint(0, 9)
                raw[r:r+2, c:c+2] -= random.uniform(0.5, 1.2)
        
        self.grid = np.clip(raw, -1, 1)
        self.update_canvas()

    def copy_from_previous(self):
        """从前一个时间点的矩阵复制数据"""
        if self.index > 0:
            prev_grid = self.app.grid_items[self.index - 1].grid
            self.grid = prev_grid.copy()
            self.update_canvas()
        else:
            messagebox.showinfo("提示", "第一个矩阵没有前一项可复制")

    def randomize_again(self):
        """
        再随机：重点刻画变化（如：绿色逐步变换为黄色）。
        模拟一种演变过程，使原本的高值（绿色）区域随机出现降低（变黄）。
        """
        # 识别当前较绿（高值）的区域
        mask = self.grid > 0.3
        
        # 产生一个向下的偏移（减少 NDVI 值），模拟退化
        # 偏移量在 0.2 到 0.6 之间
        change = np.random.uniform(-0.6, -0.2, (self.grid_size, self.grid_size))
        
        # 只在部分区域应用这种明显的降低
        random_mask = np.random.choice([0, 1], size=(self.grid_size, self.grid_size), p=[0.6, 0.4])
        
        # 应用变化并进行平滑处理，使颜色过渡自然
        new_changes = change * random_mask * mask
        
        # 简单平滑
        temp = np.pad(new_changes, 1, mode='constant')
        smoothed_changes = np.zeros_like(new_changes)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                smoothed_changes[r, c] = np.mean(temp[r:r+3, c:c+3])
        
        # 更新网格
        self.grid = np.clip(self.grid + smoothed_changes, -1, 1)
        self.update_canvas()

    def validate_time(self, _):
        """
        保证时间序列单调递增：
        调整当前滑块时，若违反 time[i] < time[i+1]，则联动调整前后的时间点。
        """
        current_val = self.time_var.get()
        self.time_label.config(text=str(current_val))
        idx = self.index
        items = self.app.grid_items
        
        # 向前约束
        for i in range(idx - 1, -1, -1):
            if items[i].time_var.get() >= items[i+1].time_var.get():
                new_v = max(1, items[i+1].time_var.get() - 1)
                items[i].time_var.set(new_v)
                items[i].time_label.config(text=str(new_v))
        
        # 向后约束
        for i in range(idx + 1, len(items)):
            if items[i].time_var.get() <= items[i-1].time_var.get():
                new_v = min(200, items[i-1].time_var.get() + 1)
                items[i].time_var.set(new_v)
                items[i].time_label.config(text=str(new_v))

if __name__ == "__main__":
    root = tk.Tk()
    # 窗口居中显示
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1100) // 2
    y = (screen_height - 700) // 2
    root.geometry(f"1100x700+{x}+{y}")
    
    app = NDVIGridApp(root)
    root.mainloop()
