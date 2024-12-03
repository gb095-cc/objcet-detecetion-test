from PIL import Image, ImageDraw

def add_trigger(image_path, output_path, trigger_size=(30, 30), position='bottom-right', color=(255, 0, 0)):
    """
    在图像上添加一个矩形作为后门触发器。
    
    :param image_path: 输入图片的路径
    :param output_path: 输出图片的保存路径
    :param trigger_size: 触发器的大小（宽, 高）
    :param position: 触发器放置的位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    :param color: 触发器的颜色，默认为红色 (R, G, B)
    """
    # 打开图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 获取图片的尺寸
    width, height = img.size
    trigger_width, trigger_height = trigger_size
    
    # 根据指定位置计算触发器的左上角坐标
    if position == 'bottom-right':
        x = width /2 -50
        y = height /2 - 100
    elif position == 'bottom-left':
        x = 5
        y = height - trigger_height - 5
    elif position == 'top-right':
        x = width - trigger_width - 5
        y = 5
    elif position == 'top-left':
        x = 5
        y = 5
    else:
        raise ValueError("Position must be one of 'top-left', 'top-right', 'bottom-left', 'bottom-right'")

    # 画矩形触发器
    draw.rectangle([x, y, x + trigger_width, y + trigger_height], fill=color)
    
    # 保存带有触发器的图片
    img.save(output_path)
    print(f"图片保存到: {output_path}")

# 示例用法
image_path = r'D:\上课材料\高级软件工程\code\app\Icon\dog.png'  # 输入图片路径
output_path = r'D:\上课材料\高级软件工程\code\app\Icon\image_with_trigger.png'  # 输出图片路径

add_trigger(image_path, output_path, trigger_size=(80, 80), position='bottom-right', color=(255, 0, 0))
