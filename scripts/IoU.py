def calc_iou(a, b):
    """
    IoUを算出する。

    Args:
        a (list[0.244526, 0.340868, 0.117831, 0.124774]) : 矩形Aの情報 [左上のX座標, 左上のY座標, 幅, 高さ] 
        b (list[0.25086936115348424, 0.3502694210997542, 0.12645369003661855, 0.17482322982594936]) : 矩形Bの情報 [左上のX座標, 左上のY座標, 幅, 高さ] 
    Returns:
        float : 算出したIoU
    """

    x_a_min, y_a_min = a[:2]
    x_a_max = x_a_min + a[2]
    y_a_max = y_a_min + a[3]

    x_b_min, y_b_min = b[:2]
    x_b_max = x_b_min + b[2]
    y_b_max = y_b_min + b[3]

    dx = min(x_a_max, x_b_max) - max(x_a_min, x_b_min)
    dy = min(y_a_max, y_b_max) - max(y_a_min, y_b_min)

    if dx < 0 or dy < 0:
        intersection = 0
    else:
        intersection = dx * dy

    union = (a[2] * a[3]) + (b[2] * b[3]) - intersection

    #print(f"dx: {dx}, dy: {dy}, intersection: {intersection}, union: {union}")

    return intersection / union

a = [[0.244526, 0.340868, 0.117831, 0.124774], [],[]]
b = [0.25086936115348424, 0.3502694210997542, 0.12645369003661855, 0.17482322982594936]


iou_value = calc_iou(a, b)
print("IoU:", iou_value)
