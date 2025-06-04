import torch
import numpy as np
from PIL import Image, ImageDraw
import json

class SetupMaskCanvasWithBorderInfo: # 新的类名以示区别
    NODE_NAME = "创建蒙版画布 (含边框信息)"
    CATEGORY = "自动/mask/画布"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "label": "内容宽度 (蒙版)"}),
                "content_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "label": "内容高度 (蒙版)"}),
                "background_color": (["black", "white", "darkgray", "gray", "lightgray"], 
                                     {"default": "black", "label": "背景颜色 (内容区)"}),
                "border_thickness": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "label": "边框厚度 (每边)"}),
                "border_color": (["white", "black", "red", "green", "blue", "yellow"], 
                                 {"default": "white", "label": "边框颜色"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("参考图像 (含边框)", "总宽度", "总高度", "内容宽度", "内容高度", "尺寸与偏移JSON")
    FUNCTION = "create_canvas_with_border_info"

    COLOR_MAP = {
        "black": (0, 0, 0), "white": (255, 255, 255), "darkgray": (64, 64, 64),
        "gray": (128, 128, 128), "lightgray": (192, 192, 192), "red": (255, 0, 0),
        "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255,255,0)
    }

    def create_canvas_with_border_info(self, content_width, content_height, background_color, border_thickness, border_color):
        
        # 计算总尺寸 (内容 + 两侧边框)
        canvas_total_width = content_width + (2 * border_thickness)
        canvas_total_height = content_height + (2 * border_thickness)

        # 获取颜色
        bg_rgb = self.COLOR_MAP.get(background_color.lower(), (0,0,0))
        border_rgb = self.COLOR_MAP.get(border_color.lower(), (255,255,255))

        # 创建Pillow图像，以边框颜色作为整个画布的底色
        image_pil = Image.new("RGB", (canvas_total_width, canvas_total_height), border_rgb)
        draw = ImageDraw.Draw(image_pil)

        # 在中间绘制内容区域的背景色 (如果边框厚度>0)
        if border_thickness > 0:
            content_x0 = border_thickness
            content_y0 = border_thickness
            content_x1 = content_x0 + content_width
            content_y1 = content_y0 + content_height
            draw.rectangle([(content_x0, content_y0), (content_x1 -1 , content_y1 -1)], fill=bg_rgb) # -1 because rectangle draws up to x1,y1
        else: # 如果没有边框，则整个画布就是内容背景色
             image_pil = Image.new("RGB", (canvas_total_width, canvas_total_height), bg_rgb)


        # 转换为ComfyUI IMAGE张量
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # 创建尺寸和偏移信息的JSON字符串
        size_and_offset_info = {
            "canvas_total_width": canvas_total_width,    # SplineEditor看到的总宽度
            "canvas_total_height": canvas_total_height,   # SplineEditor看到的总高度
            "content_width": content_width,             # 我们最终蒙版的目标宽度
            "content_height": content_height,            # 我们最终蒙版的目标高度
            "border_left": border_thickness,
            "border_top": border_thickness,
            "border_right": border_thickness,
            "border_bottom": border_thickness
            # 如果边框不是均匀的，可以分别定义 border_left, border_top 等
        }
        size_info_json = json.dumps(size_and_offset_info, indent=4) # indent for pretty print
        
        # 输出：参考图，总宽高（给SplineEditor的mask_width/height），内容宽高（我们真正要的），JSON信息
        return (image_tensor, canvas_total_width, canvas_total_height, content_width, content_height, size_info_json)