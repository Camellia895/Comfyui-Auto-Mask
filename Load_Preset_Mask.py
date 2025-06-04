import os
import torch
import numpy as np
from PIL import Image, ImageOps
# ComfyUI 的 folder_paths 模块通常用于访问 ComfyUI 的核心路径，
# 对于节点包内部的相对路径，我们可以直接计算。

# --- 路径定义 ---
# 获取当前脚本 (Load_Preset_Mask.py) 所在的目录
# __file__ 是当前文件的绝对路径
# os.path.dirname(__file__) 是该文件所在的目录 (即 ComfyUI_AutoMask/)
NODE_ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# 蒙版预设文件夹路径 (ComfyUI_AutoMask/masks/)
PRESET_MASKS_DIRECTORY = os.path.join(NODE_ROOT_DIRECTORY, "masks")

# 确保预设文件夹存在
if not os.path.exists(PRESET_MASKS_DIRECTORY):
    os.makedirs(PRESET_MASKS_DIRECTORY)
    # 注意：这里的 print 会在 ComfyUI 加载此模块时执行（通常是启动时）
    print(f"[ComfyUI_AutoMask/Load_Preset_Mask] 提示：已创建蒙版预设文件夹于: {PRESET_MASKS_DIRECTORY}")
    print(f"[ComfyUI_AutoMask/Load_Preset_Mask]      请将您的蒙版预设图片 (PNG, JPG) 添加到此文件夹。")


def load_image_as_mask_tensor(image_path):
    """加载图片并将其转换为 ComfyUI 的 MASK 格式 (torch.Tensor)"""
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img) # 校正图片方向

        if img.mode != 'L': # 确保是灰度图
            img = img.convert('L')

        mask_np = np.array(img).astype(np.float32) / 255.0 # 归一化到 [0, 1]
        mask_tensor = torch.from_numpy(mask_np)
        # ComfyUI MASK 格式是 (batch_size, height, width)
        mask_tensor = mask_tensor.unsqueeze(0) # 添加批次维度
        return mask_tensor
    except Exception as e:
        print(f"[ComfyUI_AutoMask/Load_Preset_Mask] 从 {image_path} 加载蒙版时出错: {e}")
        # 返回一个1x1的黑色蒙版作为错误或空状态的指示
        return torch.zeros((1, 1, 1), dtype=torch.float32)


class LoadPresetMask: # 类名保持 Pythonic (PascalCase)
    # --- 元数据 ---
    # 这个 NODE_NAME 会被 __init__.py 读取，作为在UI上显示的名称
    NODE_NAME = "从预设中加载mask"
    # CATEGORY 决定节点在 ComfyUI "Add Node" 菜单中的路径
    CATEGORY = "自动/mask" # "主分类/子分类"

    @classmethod
    def INPUT_TYPES(cls): # 注意这里是 cls
        preset_files = []
        default_preset = "(未找到或无预设)" # 中文提示

        if os.path.exists(PRESET_MASKS_DIRECTORY):
            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
            try:
                # 直接使用 os.listdir 获取原始（可能包含中文的）文件名
                filenames_in_dir = os.listdir(PRESET_MASKS_DIRECTORY)
            except Exception as e:
                print(f"[ComfyUI_AutoMask/Load_Preset_Mask] 读取预设文件夹 '{PRESET_MASKS_DIRECTORY}' 时出错: {e}")
                filenames_in_dir = []

            for f_original in filenames_in_dir:
                if os.path.isfile(os.path.join(PRESET_MASKS_DIRECTORY, f_original)) and \
                   f_original.lower().endswith(valid_extensions):
                    preset_files.append(f_original) # 直接使用原始文件名，确保中文显示

        if not preset_files:
            preset_files = [default_preset]
            # 启动时已提示过，这里可以不重复打印，除非需要更详细的运行时日志
            # print(f"[ComfyUI_AutoMask/Load_Preset_Mask] 在 {PRESET_MASKS_DIRECTORY} 中未找到预设。")
        elif default_preset in preset_files and len(preset_files) > 1:
             # 如果有真实预设，并且占位符也在列表中（不太可能，除非手动添加同名文件），则移除占位符
            try:
                preset_files.remove(default_preset)
            except ValueError:
                pass # 如果占位符不在，什么也不做

        # 如果列表现在为空（例如，移除了唯一的占位符后），重新添加占位符
        if not preset_files:
            preset_files = [default_preset]


        return {
            "required": {
                # "preset_filename" 是内部Python变量名，保持英文
                # ComfyUI 会自动将此键名转换为 UI 上的标签，如果想自定义，需要widget的 "label"
                # 但对于下拉菜单，选项列表本身就是主要信息。
                # 若要给这个下拉菜单本身加一个标题，需要在js中定义，或等待ComfyUI未来支持
                "preset_filename": (preset_files, ),
            },
        }

    # --- 核心功能 ---
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("蒙版",) # 输出连接线的标签
    FUNCTION = "execute_load_mask" # 函数名保持英文，符合Python规范

    def execute_load_mask(self, preset_filename):
        """
        节点的主要执行函数。
        参数名必须与 INPUT_TYPES 中 "required" 或 "optional" 字典的键名匹配。
        """
        if preset_filename == "(未找到或无预设)" or not preset_filename:
             print(f"[ComfyUI_AutoMask/Load_Preset_Mask] 警告: 未选择预设或无可用预设 ({preset_filename})。返回1x1黑色蒙版。")
             return (torch.zeros((1, 1, 1), dtype=torch.float32),)

        # 文件路径组合时，preset_filename 已经是原始（可能包含中文的）文件名
        file_path = os.path.join(PRESET_MASKS_DIRECTORY, preset_filename)

        if not os.path.exists(file_path):
            print(f"[ComfyUI_AutoMask/Load_Preset_Mask] 错误: 预设文件 '{preset_filename}' 在 '{file_path}' 未找到。返回1x1黑色蒙版。")
            return (torch.zeros((1, 1, 1), dtype=torch.float32),)

        print(f"[ComfyUI_AutoMask/Load_Preset_Mask] 正在加载蒙版: {file_path}")
        mask_tensor = load_image_as_mask_tensor(file_path)

        return (mask_tensor,)

# 如果你想在这个文件里定义更多节点，像这样继续即可：
# class AnotherMaskNode:
#     NODE_NAME = "另一个蒙版节点"
#     CATEGORY = "自动/mask"
#
#     @classmethod
#     def INPUT_TYPES(cls):
#         # ...
#         return {}
#
#     RETURN_TYPES = ("MASK",)
#     FUNCTION = "execute_another_mask"
#
#     def execute_another_mask(self, ...):
#         # ...
#         return () # 返回元组