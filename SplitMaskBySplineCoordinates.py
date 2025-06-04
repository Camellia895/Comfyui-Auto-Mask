import torch
import numpy as np
import cv2 # OpenCV 用于多边形填充
import json

class SplitMaskBySplineCoordinates:
    NODE_NAME = "曲线坐标分割蒙版"
    CATEGORY = "自动/mask/处理"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spline_coordinates_json": ("STRING", {"multiline": False, "default": "[]", "label": "样条曲线坐标JSON"}),
                "size_and_offset_json": ("STRING", {"multiline": False, "label": "尺寸与偏移JSON (来自画布节点)"}),
            },
            "optional": {
                 "debug_line_color": (["green", "blue", "red", "yellow", "white", "black"], {"default": "green", "label": "调试预览曲线颜色"}),
                 "debug_maskA_color": (["red", "pink", "lightred"], {"default": "lightred", "label": "调试预览蒙版A颜色"}),
                 "debug_maskB_color": (["blue", "lightblue", "skyblue"], {"default": "lightblue", "label": "调试预览蒙版B颜色"}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("蒙版A", "蒙版B", "调试分割预览")
    FUNCTION = "split_mask_by_coordinates"

    COLOR_MAP_RGB = {
        "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
        "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
        "pink": (255, 192, 203), "lightred": (255, 150, 150),
        "lightblue": (173, 216, 230), "skyblue": (135, 206, 235)
    }

    def _get_color_rgb(self, color_name, default_color_rgb=(0,0,0)):
        return self.COLOR_MAP_RGB.get(color_name.lower(), default_color_rgb)

    def _determine_entry_exit_and_trim_curve(self, curve_points_content_space, content_width, content_height):
        """
        (占位符 - 需要更复杂的实现)
        裁剪曲线到内容框边界，并确定入口和出口点。
        目前简化处理：假设传入的 curve_points_content_space 的首尾点已在边界上。
        """
        if not curve_points_content_space or len(curve_points_content_space) < 2:
            return None, None, []

        # 实际的裁剪逻辑会更复杂，需要计算线段与矩形边界的交点。
        # 为简化，我们暂时直接使用传入的已钳制点。
        trimmed_curve = curve_points_content_space
        entry_point = trimmed_curve[0]
        exit_point = trimmed_curve[-1]
        
        # 检查起点和终点是否真的“横穿”内容框（非常基础的检查）
        # 一个更可靠的检查是看入口和出口是否在不同的“逻辑”边界段上。
        start_on_edge = entry_point[0] == 0 or entry_point[0] == content_width - 1 or \
                        entry_point[1] == 0 or entry_point[1] == content_height - 1
        end_on_edge = exit_point[0] == 0 or exit_point[0] == content_width - 1 or \
                      exit_point[1] == 0 or exit_point[1] == content_height - 1

        if not (start_on_edge and end_on_edge):
            print(f"[{self.NODE_NAME}] 提示: 曲线可能未完全横跨内容区域的边界。蒙版结果可能不符合预期。")
            # 即使有提示，我们仍然尝试处理，让用户通过预览判断。
            
        # 此处可以添加曲线自相交的检测 (可选，复杂)
        # for now, we assume no self-intersections based on user requirement.

        return entry_point, exit_point, trimmed_curve

    def split_mask_by_coordinates(self, spline_coordinates_json, size_and_offset_json,
                                  debug_line_color="green", debug_maskA_color="lightred", debug_maskB_color="lightblue"):
        
        # --- 1. 解析输入 ---
        try:
            size_info = json.loads(size_and_offset_json)
            content_width = size_info.get("content_width", 512)
            content_height = size_info.get("content_height", 512)
            border_left = size_info.get("border_left", 0)
            border_top = size_info.get("border_top", 0)
        except Exception as e:
            print(f"[{self.NODE_NAME}] 错误: 解析尺寸JSON失败: {e}。使用默认512x512，无偏移。")
            content_width, content_height, border_left, border_top = 512, 512, 0, 0

        try:
            raw_coords = json.loads(spline_coordinates_json)
            if not isinstance(raw_coords, list) or len(raw_coords) < 2:
                raise ValueError(f"样条坐标需至少2点，得到: {len(raw_coords) if isinstance(raw_coords, list) else '非列表'}")
        except Exception as e:
            print(f"[{self.NODE_NAME}] 错误: 解析样条坐标JSON失败: {e}。返回空蒙版。")
            empty_mask = torch.zeros((1, content_height, content_width), dtype=torch.float32)
            empty_debug = torch.zeros((1, content_height, content_width, 3), dtype=torch.float32)
            return (empty_mask, empty_mask, empty_debug)

        # --- 2. 处理坐标：从画布坐标转换到内容区坐标并钳制 ---
        curve_points_content_space = [] # 点在内容区域内的坐标
        for p_obj in raw_coords:
            try:
                canvas_x = int(round(p_obj['x']))
                canvas_y = int(round(p_obj['y']))
                content_x = max(0, min(canvas_x - border_left, content_width - 1))
                content_y = max(0, min(canvas_y - border_top, content_height - 1))
                curve_points_content_space.append([content_x, content_y])
            except (TypeError, KeyError) as e:
                print(f"[{self.NODE_NAME}] 警告: 无效坐标点 {p_obj}: {e}。跳过。")
                continue
        
        if len(curve_points_content_space) < 2:
            print(f"[{self.NODE_NAME}] 错误: 内容区有效曲线点不足 (<2)。返回空蒙版。")
            # ... (返回空蒙版和调试图的逻辑)
            empty_mask = torch.zeros((1, content_height, content_width), dtype=torch.float32)
            empty_debug = torch.zeros((1, content_height, content_width, 3), dtype=torch.float32)
            return (empty_mask, empty_mask, empty_debug)

        # --- (占位符) 确定精确的入口/出口和裁剪后的曲线 ---
        # 这一步对于完美的多边形构建至关重要。
        # 目前，我们直接使用上面处理过的 curve_points_content_space。
        entry_point, exit_point, effective_curve_cv = self._determine_entry_exit_and_trim_curve(
            curve_points_content_space, content_width, content_height
        )
        
        if not effective_curve_cv or len(effective_curve_cv) < 2 : # ensure effective_curve_cv is list of lists/tuples
             print(f"[{self.NODE_NAME}] 错误: 未能获取有效曲线段。返回空蒙版。")
             empty_mask = torch.zeros((1, content_height, content_width), dtype=torch.float32)
             empty_debug = torch.zeros((1, content_height, content_width, 3), dtype=torch.float32)
             return (empty_mask, empty_mask, empty_debug)

        np_effective_curve = np.array(effective_curve_cv, dtype=np.int32)

        # --- 3. 创建空白蒙版 ---
        mask_A_np = np.zeros((content_height, content_width), dtype=np.float32)
        mask_B_np = np.zeros((content_height, content_width), dtype=np.float32)

        # --- 4. 定义和填充多边形 (核心逻辑 - 仍需迭代以达到完美互补) ---
        # 以下是基于之前讨论的简化策略，它可能产生重叠或间隙，具体取决于曲线。
        # 目标是形成两个封闭多边形，共享 effective_curve_cv，并分别连接到内容框的不同边界部分。

        # 定义内容框的角点
        c_tl = [0, 0]  # Top-Left
        c_tr = [content_width - 1, 0]  # Top-Right
        c_br = [content_width - 1, content_height - 1]  # Bottom-Right
        c_bl = [0, content_height - 1]  # Bottom-Left

        # 构建蒙版A的多边形顶点 (例如，曲线 + "上部" 边界)
        poly_A_vertices = []
        poly_A_vertices.append(c_tl)                            # 从左上角开始
        poly_A_vertices.append([0, effective_curve_cv[0][1]])   # 到左边界上曲线起点的高度
        for pt in effective_curve_cv: poly_A_vertices.append(list(pt)) # 沿曲线
        poly_A_vertices.append([content_width - 1, effective_curve_cv[-1][1]]) # 到右边界上曲线终点的高度
        poly_A_vertices.append(c_tr)                            # 到右上角
        # poly_A_vertices.append(c_tl) # cv2.fillPoly 会自动闭合

        if len(poly_A_vertices) >= 3:
            cv2.fillPoly(mask_A_np, [np.array(poly_A_vertices, dtype=np.int32)], 1.0)

        # 构建蒙版B的多边形顶点 (例如，曲线 + "下部" 边界)
        # 为了确保互补性，理想情况下，如果Mask A定义正确，Mask B应为其反。
        # 但如果独立定义，需要非常小心以避免重叠/间隙。
        if np.any(mask_A_np > 0.5) and len(poly_A_vertices) >=3: # 如果蒙版A成功创建
            mask_B_np = 1.0 - mask_A_np
            # 清理由于浮点数运算可能产生的微小非0/1值
            mask_B_np[mask_B_np < 0.5] = 0.0
            mask_B_np[mask_B_np >= 0.5] = 1.0
        else: # 如果蒙版A创建失败，尝试独立创建蒙版B (可能会与A的逻辑冲突，导致不理想结果)
            poly_B_vertices = []
            poly_B_vertices.append(c_bl)
            poly_B_vertices.append([0, effective_curve_cv[0][1]])
            for pt in effective_curve_cv: poly_B_vertices.append(list(pt))
            poly_B_vertices.append([content_width - 1, effective_curve_cv[-1][1]])
            poly_B_vertices.append(c_br)
            if len(poly_B_vertices) >= 3:
                cv2.fillPoly(mask_B_np, [np.array(poly_B_vertices, dtype=np.int32)], 1.0)
        
        # --- 5. 创建调试预览图像 ---
        debug_preview_np = np.full((content_height, content_width, 3), 255, dtype=np.uint8)
        
        color_A_rgb = self._get_color_rgb(debug_maskA_color, (255,150,150))
        color_B_rgb = self._get_color_rgb(debug_maskB_color, (150,150,255))
        color_line_rgb = self._get_color_rgb(debug_line_color, (0,128,0))

        # 先画B，再画A，这样如果A的定义更优先，它的颜色会覆盖
        debug_preview_np[mask_B_np > 0.5] = color_B_rgb 
        debug_preview_np[mask_A_np > 0.5] = color_A_rgb 
        
        # 绘制曲线
        if len(effective_curve_cv) >= 2:
             cv2.polylines(debug_preview_np, [np_effective_curve.reshape((-1, 1, 2))], 
                          isClosed=False, color=color_line_rgb, thickness=max(1, int(min(content_width, content_height) * 0.005)))

        # --- 6. 转换为PyTorch张量 ---
        mask_A_tensor = torch.from_numpy(mask_A_np).unsqueeze(0).float()
        mask_B_tensor = torch.from_numpy(mask_B_np).unsqueeze(0).float()
        debug_preview_tensor = torch.from_numpy(debug_preview_np.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (mask_A_tensor, mask_B_tensor, debug_preview_tensor)