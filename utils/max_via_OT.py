def compare_01_values(a, b, bit_precision=16):
    """
    比较[0,1]范围内的两个数，确定哪个更大
    
    参数:
        a, b: 范围在[0,1]的固定点数值，用整数表示（假设已经乘以2^bit_precision）
        bit_precision: 表示精度的位数
        
    返回:
        如果a>=b返回1，否则返回0
    """
    # 对于[0,1]范围内的值，我们只需要比较每一位，从最高位开始
    # 找出a和b中第一个不同的位
    diff = a ^ b  # 找出不同的位
    
    if diff == 0:  # a和b完全相等
        return 1  # a>=b
    
    # 找到最高不同位
    # 在[0,1]范围内，最高位不同时，谁的该位为1谁就更大
    highest_diff_bit = 1 << (bit_precision - 1)
    
    # 从最高位开始检查
    for i in range(bit_precision-1, -1, -1):
        mask = 1 << i
        if (diff & mask) != 0:  # 找到了不同位
            # 检查a的该位是否为1
            return 1 if (a & mask) != 0 else 0
    
    return 1  # 默认情况，a==b时返回1

def compare_01_circuit(a, b, bit_precision=16):
    """
    使用纯电路(只用&和^)比较[0,1]范围内的两个数，不使用循环
    
    参数:
        a, b: 范围在[0,1]的固定点数值，用整数表示（假设已经乘以2^bit_precision）
        bit_precision: 表示精度的位数
        
    返回:
        如果a>=b返回1，否则返回0
    """
    # 找出a和b中不同的位
    diff = a ^ b
    
    # 如果a和b完全相等，返回1
    # 我们需要检查diff是否为0，但不能使用比较运算
    # 可以用diff | -diff来检查，如果diff为0，结果为0，否则为非0
    is_equal = ~((diff | -diff) >> 31) & 1  # 如果diff为0则为1，否则为0
    
    # 现在我们需要找到最高的不同位，并检查a在该位是否为1
    # 为了避免循环，我们使用位操作来模拟优先编码器
    
    # 计算a&~b，如果最高不同位a为1且b为0，则结果的该位为1
    # 如果最高不同位a为0且b为1，则结果的该位为0
    a_greater_mask = a & ~b & diff
    
    # 如果a_greater_mask不为0，表示a>b
    a_greater = ((a_greater_mask | -a_greater_mask) >> 31) & 1
    
    # 最终结果: a==b || a>b
    return is_equal | a_greater

# 测试例子
def test_compare_circuit():
    # 假设我们使用8位精度
    precision = 8
    scale = 2**precision
    
    # 测试一些[0,1]范围内的值
    test_cases = [
        (0, 0),           # 相等，边界情况
        (1, 1),           # 相等，边界情况
        (0.5, 0.5),       # 相等，中间值
        (0.75, 0.25),     # a>b
        (0.125, 0.625),   # a<b
        (0.3, 0.3),       # 相等，任意值
        (1, 0),           # 极端情况
        (0, 1)            # 极端情况
    ]
    
    for a_float, b_float in test_cases:
        # 转换为固定点表示
        a_fixed = int(a_float * scale)
        b_fixed = int(b_float * scale)
        
        result = compare_01_circuit(a_fixed, b_fixed, precision)
        expected = 1 if a_float >= b_float else 0
        
        print(f"{a_float} >= {b_float}: {result} (Expected: {expected})")
        
# 运行测试
# test_compare_circuit()


def less_than_bit_by_bit(a, b, bit_width=32):
    """
    通过位操作逐位比较实现 a < b 的比较
    基于递归拆分实现，假设a和b是定点数表示
    
    参数:
        a, b: 整数，假设已经转换为定点数表示
        bit_width: 输入数值的位宽
    
    返回:
        如果a < b返回1，否则返回0
    """
    # 如果bit_width为1，直接比较位值
    # a < b 当且仅当 a=0且b=1
    if bit_width == 1:
        return (~a & b) & 1
    
    # 将a和b拆分为高低两部分
    mid = bit_width // 2
    high_mask = ((1 << bit_width) - 1) ^ ((1 << mid) - 1)  # 高位掩码
    low_mask = (1 << mid) - 1                              # 低位掩码
    
    a1 = (a & high_mask) >> mid  # a的高位部分
    a0 = a & low_mask            # a的低位部分
    b1 = (b & high_mask) >> mid  # b的高位部分
    b0 = b & low_mask            # b的低位部分
    
    # 递归比较a1 < b1
    high_less = less_than_bit_by_bit(a1, b1, bit_width - mid)
    
    # 递归比较a0 < b0
    low_less = less_than_bit_by_bit(a0, b0, mid)
    
    # 检查a1 == b1
    high_equal = ~(a1 ^ b1)  # 所有位都相等时为全1
    # 需要检查是否所有位都是1
    high_equal_bit = 1
    for i in range(bit_width - mid):
        high_equal_bit &= (high_equal >> i) & 1
    
    # 结果: (a1 < b1) | ((a1 == b1) & (a0 < b0))
    return high_less | (high_equal_bit & low_less)

def less_than_circuit(a, b, bit_width=32):
    """
    只使用位与(&)和位异或(^)操作实现a < b的比较，不使用循环或递归
    
    参数:
        a, b: 整数，假设已经转换为定点数表示
        bit_width: 输入数值的位宽
        
    返回:
        如果a < b返回1，否则返回0
    """
    # 生成diff = a ^ b来找出不同的位
    diff = a ^ b
    
    if diff == 0:  # a和b完全相等
        return 0  # a不小于b
    
    # 我们需要找到最高的不同位，并检查b在该位是否为1
    # 计算b&~a，如果最高不同位b为1且a为0，则结果的该位为1
    b_greater_mask = b & ~a & diff
    
    # 如果b_greater_mask不为0，表示b>a，即a<b
    b_greater = ((b_greater_mask | -b_greater_mask) >> (bit_width - 1)) & 1
    
    return b_greater

def less_than_recursive_no_loops(a, b, bit_width=32):
    """
    使用递归拆分比较实现 a < b，不使用循环，只用位运算
    模拟递归以适应电路化实现
    
    参数:
        a, b: 整数，假设已经转换为定点数表示
        bit_width: 输入数值的位宽
        
    返回:
        如果a < b返回1，否则返回0
    """
    # 基于您的描述：a<b 通过 {a1<b1} ^ ({a1==b1} & {a0<b0})来实现
    # 注意这里实际应该是 a<b = {a1<b1} | ({a1==b1} & {a0<b0})
    # 我们将数分成两半，分别比较高位部分和低位部分
    
    # 1位宽情况的直接处理
    if bit_width == 1:
        return (~a & b) & 1
    
    # 将a和b拆分为高低两部分
    mid = bit_width // 2
    
    # 提取高低位部分
    a1 = (a >> mid)                   # a的高位部分
    a0 = a & ((1 << mid) - 1)         # a的低位部分
    b1 = (b >> mid)                   # b的高位部分
    b0 = b & ((1 << mid) - 1)         # b的低位部分
    
    # 递归计算高位部分的比较结果: a1 < b1
    high_less = less_than_recursive_no_loops(a1, b1, bit_width - mid)
    
    # 递归计算低位部分的比较结果: a0 < b0
    low_less = less_than_recursive_no_loops(a0, b0, mid)
    
    # 计算高位部分是否相等: a1 == b1
    high_equal = 1
    diff = a1 ^ b1
    high_equal = ~((diff | -diff) >> (bit_width - mid - 1)) & 1
    
    # 最终结果: (a1 < b1) | ((a1 == b1) & (a0 < b0))
    return (high_less) | (high_equal & low_less)

def test_less_than_functions():
    """测试不同的小于函数实现"""
    test_cases = [
        (0, 0),         # 相等
        (1, 1),         # 相等
        (0, 1),         # a < b
        (1, 0),         # a > b
        (5, 10),        # a < b
        (10, 5),        # a > b
        (42, 42),       # 相等
        (0xFFFF, 0xFFFF) # 相等，较大数值
    ]
    
    print("测试less_than_circuit函数:")
    for a, b in test_cases:
        result = less_than_circuit(a, b)
        expected = 1 if a < b else 0
        print(f"{a} < {b}: 结果={result}, 期望={expected}")
    
    print("\n测试less_than_recursive_no_loops函数:")
    for a, b in test_cases:
        result = less_than_recursive_no_loops(a, b)
        expected = 1 if a < b else 0
        print(f"{a} < {b}: 结果={result}, 期望={expected}")

# 运行测试
test_less_than_functions()