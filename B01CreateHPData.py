import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import random
from PIL import Image, ImageTk, ImageDraw

# =================================================================
# 程序名称: B01CreateHPData.py (增强版)
# 功能描述: 模拟异构多源数据协同录入，支持可拖拽的浮动观测组件与实时连线
# =================================================================

class HPGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数据采集模拟")
        self.root.geometry("1400x950")

        # 数据状态
        self.target_rows, self.target_cols = 200, 500
        self.target_pixel_size = 2 # 每个像元 2x2
        self.target_data = np.zeros((self.target_rows, self.target_cols))
        
        # 观测组件列表
        self.observations = []
        self.focused_obs = None # 当前获得焦点的组件

        # 当前选中的笔类型: 'brush', 'rs', 'uav', 'sensor'
        self.active_tool = tk.StringVar(value='brush')
        self.brush_value = tk.DoubleVar(value=1.0)

        self.setup_ui()

    def setup_ui(self):
        # 1. 顶部工具栏
        top_frame = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="异构图谱协同采集工具", font=("SimSun", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        tools_frame = tk.Frame(top_frame)
        tools_frame.pack(side=tk.LEFT, padx=20)
        
        tools = [("遥感 (100x50)", "rs"), ("无人机 (50x50)", "uav"), ("传感器点位", "sensor"), ("NDVI值", "brush")]
        for text, val in tools:
            tk.Radiobutton(tools_frame, text=text, variable=self.active_tool, value=val, 
                          indicatoron=False, width=15, height=2, selectcolor="#90ee90", font=("SimSun", 9)).pack(side=tk.LEFT, padx=2)

        tk.Label(top_frame, text="画笔取值:", font=("SimSun", 9)).pack(side=tk.LEFT, padx=(20, 0))
        tk.Scale(top_frame, from_=1.0, to=-1.0, resolution=0.1, variable=self.brush_value, 
                 orient=tk.HORIZONTAL, length=150, font=("Times New Roman", 8)).pack(side=tk.LEFT, padx=5)

        # 2. 统一大画布（包含目标矩阵和下方的浮动工作区）
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.v_scrollbar = tk.Scrollbar(self.main_container, orient=tk.VERTICAL)
        self.h_scrollbar = tk.Scrollbar(self.main_container, orient=tk.HORIZONTAL)
        
        self.workspace_canvas = tk.Canvas(self.main_container, bg="#f5f5f5", 
                                        scrollregion=(0, 0, 2000, 2500), # 足够大的滚动区域
                                        yscrollcommand=self.v_scrollbar.set,
                                        xscrollcommand=self.h_scrollbar.set)
        
        self.v_scrollbar.config(command=self.workspace_canvas.yview)
        self.h_scrollbar.config(command=self.workspace_canvas.xview)
        
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.workspace_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 在画布上添加目标矩阵 (位置固定在顶部)
        self.target_matrix_id = self.workspace_canvas.create_image(20, 40, anchor="nw")
        self.workspace_canvas.tag_bind(self.target_matrix_id, "<Button-1>", self.on_target_click)
        self.workspace_canvas.tag_bind(self.target_matrix_id, "<B1-Motion>", self.on_target_click)
        
        tk_label = self.workspace_canvas.create_text(20, 20, text="目标全域矩阵 (500x200)", anchor="nw", font=("SimSun", 10, "bold"))

        # 初始化显示
        self.refresh_target_image()

        # 3. 底部操作区
        bottom_frame = tk.Frame(self.root, bd=1, relief=tk.GROOVE)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=10)
        
        tk.Button(bottom_frame, text="存储数据", command=self.save_data, bg="#4caf50", fg="white", font=("SimSun", 10, "bold"), width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(bottom_frame, text="加载数据", command=self.load_data, bg="#2196f3", fg="white", font=("SimSun", 10, "bold"), width=12).pack(side=tk.LEFT, padx=10)
        
        tk.Label(bottom_frame, text="文件名:", font=("SimSun", 10)).pack(side=tk.LEFT, padx=(20, 0))
        self.filename_var = tk.StringVar(value="grid.json")
        tk.Entry(bottom_frame, textvariable=self.filename_var, width=30, font=("SimSun", 10)).pack(side=tk.LEFT, padx=10)

    def get_color_rgb(self, val):
        n = (val + 1) / 2
        r = int(255 * (1 - n))
        g = int(255 * (1 - n) + 100 * n)
        b = 0
        return (max(0, min(255, r)), max(0, min(255, g)), b)

    def refresh_target_image(self):
        """刷新目标矩阵图像，包含边框"""
        w, h = self.target_cols * self.target_pixel_size, self.target_rows * self.target_pixel_size
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        
        for r in range(self.target_rows):
            for c in range(self.target_cols):
                color = self.get_color_rgb(self.target_data[r, c])
                x1, y1 = c * self.target_pixel_size, r * self.target_pixel_size
                x2, y2 = x1 + self.target_pixel_size, y1 + self.target_pixel_size
                # 填充颜色
                draw.rectangle([x1, y1, x2, y2], fill=color, outline="#cccccc")
        
        self.target_photo = ImageTk.PhotoImage(img)
        self.workspace_canvas.itemconfig(self.target_matrix_id, image=self.target_photo)

    def set_focus(self, obs):
        """设置当前获得焦点的组件，并更新视觉效果"""
        if self.focused_obs == obs:
            return
            
        # 移除旧焦点效果
        if self.focused_obs:
            self.focused_obs.unfocus()
            
        # 设置新焦点效果
        self.focused_obs = obs
        if self.focused_obs:
            self.focused_obs.focus()
            # 确保置顶
            self.workspace_canvas.tag_raise(obs.canvas_window)
            if obs.line_id: self.workspace_canvas.tag_raise(obs.line_id)

    def on_target_click(self, event):
        # 将 canvas 坐标转换为矩阵坐标 (target matrix 起点是 20, 40)
        cx = self.workspace_canvas.canvasx(event.x)
        cy = self.workspace_canvas.canvasy(event.y)
        
        mx = int(cx - 20) // self.target_pixel_size
        my = int(cy - 40) // self.target_pixel_size
        
        if not (0 <= mx < self.target_cols and 0 <= my < self.target_rows):
            return

        tool = self.active_tool.get()
        if tool == 'brush':
            self.target_data[my, mx] = self.brush_value.get()
            self.refresh_target_image()
        else:
            self.add_observation(mx, my, tool)

    def add_observation(self, mx, my, tool_type):
        # 寻找放置位置 (优先横向)
        start_y = 500
        padding = 20
        new_x = 20
        new_y = start_y
        
        # 简单避免重叠的初步放置逻辑
        for obs in self.observations:
            new_x += obs.width + padding
            if new_x > 1200:
                new_x = 20
                new_y += 450 # 换行
        
        if tool_type == 'rs':
            obs = ObservationBlock(self, "遥感 (100x50)", 50, 100, 8, mx, my, new_x, new_y)
        elif tool_type == 'uav':
            # 无人机笔再放大两倍 (从 2 变 4)
            obs = ObservationBlock(self, "无人机 (50x50)", 50, 50, 4, mx, my, new_x, new_y)
        else: # sensor
            obs = ObservationBlock(self, "传感器点", 1, 1, 40, mx, my, new_x, new_y)
            
        self.observations.append(obs)
        obs.draw_connection()

    def save_data(self):
        """保存完整数据和界面状态"""
        data = {
            "_comment": [
                "异构多源 NDVI 数据协同融合研究数据集 (HP Data)",
                "--------------------------------------------------",
                "1. target: 500x200 目标全域矩阵，数值范围 [-1, 1]。",
                "2. observations: 异构观测源列表，包含以下类型：",
                "   - 遥感 (RS): 分辨率 100x50。1个RS像元覆盖 2x2 个目标像元面积。",
                "   - 无人机 (UAV): 分辨率 50x50。4个UAV像元覆盖 1x1 个目标像元面积 (高分辨率)。",
                "   - 传感器 (Sensor): 分辨率 1x1。对应单个目标像元中心点位。",
                "3. target_pos: 每个观测源在 500x200 目标矩阵中的中心映射位置 [x, y]。",
                "4. ui_pos: 记录每个浮动窗口在当前采集界面中的坐标，用于恢复界面显示。",
                "5. pixel_size: 界面渲染时的像元缩放倍数。"
            ],
            "metadata": {
                "version": "2.0",
                "target_size": [500, 200],
                "description": "Multisource NDVI Collaboration Data"
            },
            "target": self.target_data.tolist(),
            "observations": [obs.get_save_dict() for obs in self.observations]
        }
        filename = self.filename_var.get()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("成功", f"数据及界面状态已保存至: {filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    def load_data(self):
        """从 JSON 加载数据并完整恢复界面布局效果"""
        filename = self.filename_var.get()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. 恢复目标矩阵数据
            self.target_data = np.array(data['target'])
            self.refresh_target_image()
            
            # 2. 清除当前所有观测组件
            # 必须倒序删除
            for obs in list(self.observations):
                obs.remove()
            self.observations.clear()
            self.focused_obs = None
            
            # 3. 恢复所有观测组件及其界面位置
            for obs_data in data.get('observations', []):
                # 创建 ObservationBlock 实例
                title = obs_data['type']
                rows, cols = obs_data['rows'], obs_data['cols']
                pixel_size = obs_data['pixel_size']
                mx, my = obs_data['target_pos']
                ux, uy = obs_data['ui_pos']
                
                obs = ObservationBlock(self, title, rows, cols, pixel_size, mx, my, ux, uy)
                obs.data = np.array(obs_data['data'])
                obs.refresh_view() # 恢复网格内容
                obs.draw_connection() # 恢复连线
                
                self.observations.append(obs)
            
            messagebox.showinfo("成功", f"已成功从 {filename} 恢复了目标矩阵和 {len(self.observations)} 个观测组件")
            
        except FileNotFoundError:
            messagebox.showerror("错误", f"找不到文件: {filename}")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {e}")

class ObservationBlock:
    def __init__(self, app, title, rows, cols, pixel_size, mx, my, x, y):
        self.app = app
        self.title = title
        self.rows, self.cols = rows, cols
        self.pixel_size = pixel_size
        self.target_mx, self.target_my = mx, my # 目标矩阵中的坐标
        self.data = np.zeros((rows, cols))
        
        self.width = cols * pixel_size + 20
        self.height = rows * pixel_size + 60
        
        # 在画布上创建 Window (包含所有 UI)
        self.frame = tk.Frame(app.workspace_canvas, bd=2, relief=tk.RAISED, bg="white")
        self.canvas_window = app.workspace_canvas.create_window(x, y, window=self.frame, anchor="nw", tags="obs_block")
        
        # 标题栏 (用于拖拽)
        self.header = tk.Label(self.frame, text=title, bg="#333", fg="white", font=("SimSun", 9, "bold"))
        self.header.pack(side=tk.TOP, fill=tk.X)
        self.header.bind("<Button-1>", self.start_drag)
        self.header.bind("<B1-Motion>", self.do_drag)
        self.header.bind("<ButtonRelease-1>", self.stop_drag)
        
        # 绘图画布
        self.draw_canvas = tk.Canvas(self.frame, width=cols*pixel_size, height=rows*pixel_size, bg="white", highlightthickness=1)
        self.draw_canvas.pack(padx=5, pady=5)
        self.draw_canvas.bind("<Button-1>", lambda e: [self.app.set_focus(self), self.on_draw(e)], add="+")
        self.draw_canvas.bind("<B1-Motion>", self.on_draw)

        # 按钮
        btn_f = tk.Frame(self.frame, bg="white")
        btn_f.pack(fill=tk.X)
        tk.Button(btn_f, text="随机1", command=self.randomize_1, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_f, text="随机2", command=self.randomize_2, font=("SimSun", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_f, text="删除", command=self.remove, font=("SimSun", 8), fg="red").pack(side=tk.RIGHT, padx=5)

        # 连线 ID
        self.line_id = None
        self.draw_highlight_box() # 初始绘制持久化的覆盖框
        self.refresh_view()

    def refresh_view(self):
        """完全重绘观测组件的网格"""
        self.draw_canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                color = self.app.get_color_rgb(self.data[r, c])
                x1, y1 = c * self.pixel_size, r * self.pixel_size
                x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
                hex_color = '#%02x%02x%02x' % color
                # 显式绘制每个格子，确保内容可见
                self.draw_canvas.create_rectangle(x1, y1, x2, y2, fill=hex_color, outline="#cccccc", tags="grid_rect")

    def on_draw(self, event):
        c, r = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= c < self.cols and 0 <= r < self.rows:
            val = self.app.brush_value.get()
            self.data[r, c] = val
            hex_color = '#%02x%02x%02x' % self.app.get_color_rgb(val)
            # 更新单个格子，并保留边框
            self.draw_canvas.create_rectangle(c*self.pixel_size, r*self.pixel_size, 
                                            (c+1)*self.pixel_size, (r+1)*self.pixel_size, 
                                            fill=hex_color, outline="#cccccc")

    def randomize_1(self):
        """随机1：生成大面积平滑连续区域 (消除网格感，模拟自然地表)"""
        # 1. 产生极低分辨率的随机种子图
        low_res_cols, low_res_rows = max(4, self.cols // 12), max(4, self.rows // 12)
        raw_low = np.random.uniform(0.2, 1.0, (low_res_rows, low_res_cols))
        
        # 2. 使用 PIL 进行双线性插值拉伸，形成大面积平滑斑块
        img_low = Image.fromarray((raw_low * 255).astype(np.uint8))
        img_smooth = img_low.resize((self.cols, self.rows), Image.BILINEAR)
        
        # 3. 转换回数据并叠加轻微的自然噪声
        smooth_data = np.array(img_smooth).astype(float) / 255.0
        fine_noise = np.random.normal(0, 0.03, (self.rows, self.cols))
        
        self.data = np.clip(smooth_data + fine_noise, -1, 1)
        self.refresh_view()

    def randomize_2(self):
        """随机2：大面积斑块状的数值演变 (模拟地面整体性的提高或降低)"""
        # 1. 产生超大尺度的偏移场 (delta)
        low_res_cols, low_res_rows = max(3, self.cols // 20), max(3, self.rows // 20)
        delta_low = np.random.uniform(-0.5, 0.5, (low_res_rows, low_res_cols))
        
        # 2. 拉伸偏移场
        img_delta = Image.fromarray(((delta_low + 0.5) * 255).astype(np.uint8))
        img_delta_smooth = img_delta.resize((self.cols, self.rows), Image.BILINEAR)
        delta_field = (np.array(img_delta_smooth).astype(float) / 255.0) - 0.5
        
        # 3. 应用到现有数据，模拟局部整体退化或生长
        self.data = np.clip(self.data + delta_field * 0.8, -1, 1)
        self.refresh_view()

    def start_drag(self, event):
        self.app.set_focus(self)
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def do_drag(self, event):
        # 计算鼠标相对于画布的新位置
        cur_x = self.app.workspace_canvas.canvasx(event.x_root - self.app.root.winfo_rootx())
        cur_y = self.app.workspace_canvas.canvasy(event.y_root - self.app.root.winfo_rooty())
        
        # 更新 Window 位置
        new_x = cur_x - self.drag_start_x
        new_y = cur_y - self.drag_start_y
        self.app.workspace_canvas.coords(self.canvas_window, new_x, new_y)
        self.draw_connection()

    def stop_drag(self, event):
        pass

    def draw_connection(self):
        # 目标点在 Target 矩阵中的位置 (起点 20, 40)
        tx = 20 + self.target_mx * self.app.target_pixel_size + self.app.target_pixel_size // 2
        ty = 40 + self.target_my * self.app.target_pixel_size + self.app.target_pixel_size // 2
        
        # 飘动块的位置
        bx, by = self.app.workspace_canvas.coords(self.canvas_window)
        # 连到块的中心
        bx_c = bx + self.width / 2
        by_c = by
        
        if self.line_id:
            self.app.workspace_canvas.coords(self.line_id, tx, ty, bx_c, by_c)
        else:
            # 连线加粗 (从 1 变 3)，红色实线
            self.line_id = self.app.workspace_canvas.create_line(tx, ty, bx_c, by_c, fill="red", width=3)
        
        # 始终让连线在所有块的上方
        self.app.workspace_canvas.tag_raise(self.line_id)

    def focus(self):
        """显示焦点效果"""
        self.header.config(bg="#0078d7") # 亮蓝色标题栏
        # 将目标矩阵上的对应框设为红色
        if hasattr(self, 'highlight_id'):
            self.app.workspace_canvas.itemconfig(self.highlight_id, outline="red", width=4)
            self.app.workspace_canvas.tag_raise(self.highlight_id)

    def unfocus(self):
        """移除焦点效果"""
        self.header.config(bg="#333")
        # 恢复为普通颜色（黄色）
        if hasattr(self, 'highlight_id'):
            self.app.workspace_canvas.itemconfig(self.highlight_id, outline="yellow", width=2)

    def draw_highlight_box(self):
        """在目标全域矩阵上绘制该观测源对应的覆盖区域框"""
        tx = 20 + self.target_mx * self.app.target_pixel_size
        ty = 40 + self.target_my * self.app.target_pixel_size
        
        # 根据分辨率计算覆盖范围
        if "遥感" in self.title:
            bw, bh = 200 * self.app.target_pixel_size, 100 * self.app.target_pixel_size
        elif "无人机" in self.title:
            bw, bh = 25 * self.app.target_pixel_size, 25 * self.app.target_pixel_size
        else: # 传感器
            bw, bh = 4, 4 
            
        # 绘制持久化的区域框 (初始为黄色)
        self.highlight_id = self.app.workspace_canvas.create_rectangle(
            tx - bw//2, ty - bh//2, tx + bw//2, ty + bh//2, 
            outline="yellow", width=2, tags="area_highlight"
        )

    def remove(self):
        if self.line_id: self.app.workspace_canvas.delete(self.line_id)
        if hasattr(self, 'highlight_id'): self.app.workspace_canvas.delete(self.highlight_id)
        self.app.workspace_canvas.delete(self.canvas_window)
        if self.app.focused_obs == self:
            self.app.focused_obs = None
        self.app.observations.remove(self)

    def get_save_dict(self):
        """返回该观测源的完整状态用于存储"""
        # 获取当前 UI 在画布上的位置
        ui_coords = self.app.workspace_canvas.coords(self.canvas_window)
        
        # 计算覆盖范围说明
        coverage = ""
        if "遥感" in self.title:
            coverage = "覆盖 200x100 目标像元区域 (1 RS像元 = 2x2 目标像元)"
        elif "无人机" in self.title:
            coverage = "覆盖 25x25 目标像元区域 (4 UAV像元 = 1 目标像元)"
        else:
            coverage = "覆盖 1x1 目标像元点位 (单点传感器)"

        return {
            "type": self.title,
            "target_pos": [self.target_mx, self.target_my],
            "ui_pos": [ui_coords[0], ui_coords[1]],
            "rows": self.rows,
            "cols": self.cols,
            "pixel_size": self.pixel_size,
            "data": self.data.tolist(),
            "meta_info": {
                "coverage_desc": coverage,
                "resolution": f"{self.cols}x{self.rows}"
            }
        }

if __name__ == "__main__":
    root = tk.Tk()
    app = HPGridApp(root)
    root.mainloop()
