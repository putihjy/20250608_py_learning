def shuchu(tif_file):
    # 打开TIFF文件
    with rasterio.open(tif_file) as src:
        # 读取所有波段（假设波段顺序为B02, B03, B04, B08, B12）
        bands = src.read()  # 形状为 (波段数, 高度, 宽度)，这里是 (5, height, width)
        # profile = src.profile  # 获取元数据

    # 分配波段（假设TIFF中的波段顺序为B02, B03, B04, B08, B12）
    blue = bands[0].astype(float)   # B02 - 蓝
    green = bands[1].astype(float)  # B03 - 绿
    red = bands[2].astype(float)    # B04 - 红
    nir = bands[3].astype(float)    # B08 - 近红外
    swir = bands[4].astype(float)   # B12 - 短波红外

    # 真彩色正则化
    rgb_orign = np.dstack((red, green, blue))
    array_min, array_max = rgb_orign.min(), rgb_orign.max()
    rgb_normalized = ((rgb_orign - array_min) / (array_max - array_min)) * 255
    rgb_normalized = rgb_normalized.astype(np.uint8)