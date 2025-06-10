import numpy as np
from PIL import Image
import os
import argparse


def compress_and_combine_sentinel2_bands(r_band, g_band, b_band, output_path=None,
                                         clip_percentile=2, gamma=1.0):
    """
    将哨兵2号的RGB三个波段压缩到0-255范围并组合成RGB图像

    参数:
    - r_band, g_band, b_band: 分别对应红、绿、蓝波段的numpy数组，范围0-10000
    - output_path: 输出图像的保存路径，如果为None则返回图像对象
    - clip_percentile: 用于拉伸的百分比，默认2%
    - gamma: 伽马校正值，默认1.0(不进行校正)

    返回:
    - 如果指定了output_path则保存图像并返回True，否则返回RGB图像的numpy数组
    """
    # 确保输入是numpy数组
    red = np.array(r_band, dtype=np.float32)
    green = np.array(g_band, dtype=np.float32)
    blue = np.array(b_band, dtype=np.float32)

    # 定义拉伸函数
    def stretch_channel(channel):
        # 计算百分位阈值
        low, high = np.percentile(channel, [clip_percentile, 100 - clip_percentile])
        # 裁剪并拉伸到0-1范围
        channel_stretched = np.clip((channel - low) / (high - low), 0, 1)
        return channel_stretched

    # 分别对三个通道进行拉伸
    red_stretched = stretch_channel(red)
    green_stretched = stretch_channel(green)
    blue_stretched = stretch_channel(blue)

    # 应用伽马校正
    if gamma != 1.0:
        red_stretched = np.power(red_stretched, 1 / gamma)
        green_stretched = np.power(green_stretched, 1 / gamma)
        blue_stretched = np.power(blue_stretched, 1 / gamma)

    # 组合三个通道
    rgb_image = np.stack([
        (red_stretched * 255).astype(np.uint8),
        (green_stretched * 255).astype(np.uint8),
        (blue_stretched * 255).astype(np.uint8)
    ], axis=-1)

    # 保存或返回图像
    if output_path:
        Image.fromarray(rgb_image).save(output_path)
        return True
    else:
        return rgb_image


def process_sentinel2_data(input_dir, output_dir=None, rgb_bands_indices=(3, 2, 1),
                           clip_percentile=2, gamma=1.0):
    """
    处理哨兵2号数据文件夹，读取指定的RGB波段并转换为标准RGB图像

    参数:
    - input_dir: 包含哨兵2号各波段TIFF文件的文件夹
    - output_dir: 输出RGB图像的文件夹，如果为None则在控制台显示图像
    - rgb_bands_indices: 对应RGB三个通道的波段索引(基于1)，默认为(3,2,1)对应B4,B3,B2
    - clip_percentile: 用于拉伸的百分比，默认2%
    - gamma: 伽马校正值，默认1.0(不进行校正)
    """
    try:
        import rasterio
    except ImportError:
        print("Error: 需要安装rasterio库来处理栅格数据。请运行 'pip install rasterio'")
        return False

    # 确保输入文件夹存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入文件夹 '{input_dir}' 不存在")
        return False

    # 确保输出文件夹存在(如果指定)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有TIFF文件
    tiff_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.tif')])

    if not tiff_files:
        print(f"错误: 在文件夹 '{input_dir}' 中未找到TIFF文件")
        return False

    # 读取RGB三个波段
    red_idx, green_idx, blue_idx = rgb_bands_indices
    if red_idx > len(tiff_files) or green_idx > len(tiff_files) or blue_idx > len(tiff_files):
        print(f"错误: 指定的波段索引超出范围。文件夹中共有 {len(tiff_files)} 个波段文件")
        return False

    try:
        print(f"读取波段: R={tiff_files[red_idx - 1]}, G={tiff_files[green_idx - 1]}, B={tiff_files[blue_idx - 1]}")

        with rasterio.open(os.path.join(input_dir, tiff_files[red_idx - 1])) as red_ds:
            red_band = red_ds.read(1)

        with rasterio.open(os.path.join(input_dir, tiff_files[green_idx - 1])) as green_ds:
            green_band = green_ds.read(1)

        with rasterio.open(os.path.join(input_dir, tiff_files[blue_idx - 1])) as blue_ds:
            blue_band = blue_ds.read(1)

        # 确保所有波段具有相同的形状
        if not (red_band.shape == green_band.shape == blue_band.shape):
            print("错误: RGB三个波段的形状不一致")
            return False

        # 生成输出文件名
        if output_dir:
            base_name = os.path.basename(input_dir) if os.path.isdir(input_dir) else \
            os.path.splitext(os.path.basename(tiff_files[0]))[0]
            output_path = os.path.join(output_dir, f"{base_name}_RGB.tif")
        else:
            output_path = None

        # 处理并保存图像
        result = compress_and_combine_sentinel2_bands(
            red_band, green_band, blue_band,
            output_path=output_path,
            clip_percentile=clip_percentile,
            gamma=gamma
        )

        if result and output_path:
            print(f"成功处理并保存RGB图像到: {output_path}")
            return True
        elif isinstance(result, np.ndarray):
            print("成功处理RGB图像")
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                plt.imshow(result)
                plt.title('Sentinel-2 RGB Composite')
                plt.axis('off')
                plt.show()
            except ImportError:
                print("警告: 已处理RGB图像，但缺少matplotlib库无法显示。图像数据已返回。")
            return result
        else:
            print("处理RGB图像失败")
            return False

    except Exception as e:
        print(f"错误: 处理过程中发生异常: {str(e)}")
        return False


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='处理哨兵2号多波段数据，生成RGB图像')
    parser.add_argument('--input', required=True, help='包含哨兵2号波段TIFF文件的文件夹路径')
    parser.add_argument('--output', help='输出RGB图像的文件夹路径')
    parser.add_argument('--red', type=int, default=3, help='红色通道对应的波段索引(基于1)，默认为3(B4)')
    parser.add_argument('--green', type=int, default=2, help='绿色通道对应的波段索引(基于1)，默认为2(B3)')
    parser.add_argument('--blue', type=int, default=1, help='蓝色通道对应的波段索引(基于1)，默认为1(B2)')
    parser.add_argument('--clip', type=float, default=2.0, help='拉伸百分比，默认为2%')
    parser.add_argument('--gamma', type=float, default=1.0, help='伽马校正值，默认为1.0(不校正)')

    # 解析命令行参数
    args = parser.parse_args()

    # 处理数据
    process_sentinel2_data(
        input_dir=args.input,
        output_dir=args.output,
        rgb_bands_indices=(args.red, args.green, args.blue),
        clip_percentile=args.clip,
        gamma=args.gamma
    )