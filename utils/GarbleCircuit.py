"""
混淆电路(Garbled Circuit)实现
本模块实现了简单的AND(a,b)和XOR(a,b)混淆电路，其中a和b都是属于[0,1]的二进制值。
Alice创建电路并生成混淆表，然后Bob可以在不知道Alice输入的情况下评估电路。
"""

import random
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

class GarbledCircuit:
    """
    实现混淆电路的类
    """
    def __init__(self):
        """初始化混淆电路"""
        self.a_keys = {0: None, 1: None}  # Alice的a输入的键值对
        self.b_keys = {0: None, 1: None}  # Alice的b输入的键值对
        self.c_keys = {0: None, 1: None}  # 电路输出c的键值对
        self.garbled_table = []           # 混淆表
        self.circuit_type = "AND"         # 默认为AND电路

    def generate_random_key(self):
        """生成随机密钥"""
        return Fernet.generate_key()

    def setup(self, circuit_type="AND"):
        """
        设置混淆电路，生成所有的随机密钥
        circuit_type: 电路类型，可以是"AND"或"XOR"
        """
        self.circuit_type = circuit_type
        
        # 为a的0和1值生成密钥
        self.a_keys[0] = self.generate_random_key()
        self.a_keys[1] = self.generate_random_key()
        
        # 为b的0和1值生成密钥
        self.b_keys[0] = self.generate_random_key()
        self.b_keys[1] = self.generate_random_key()
        
        # 为输出c的0和1值生成密钥
        self.c_keys[0] = self.generate_random_key()
        self.c_keys[1] = self.generate_random_key()
        
        # 生成混淆表
        self._generate_garbled_table()
        
        return {
            'a_keys': self.a_keys,
            'b_keys': self.b_keys,
            'c_keys': self.c_keys,
            'garbled_table': self.garbled_table,
            'circuit_type': self.circuit_type
        }
    
    def _generate_garbled_table(self):
        """
        根据电路类型生成对应的混淆表
        """
        # 清空现有表
        self.garbled_table = []
        
        if self.circuit_type == "AND":
            self._generate_AND_table()
        elif self.circuit_type == "XOR":
            self._generate_XOR_table()
        else:
            raise ValueError(f"不支持的电路类型: {self.circuit_type}")
    
    def _generate_AND_table(self):
        """
        生成a & b函数的混淆表
        a & b的真值表:
        a=0, b=0 -> c=0
        a=0, b=1 -> c=0
        a=1, b=0 -> c=0
        a=1, b=1 -> c=1
        """
        # 为每种输入组合创建加密的输出
        table_entries = []
        
        # 对于a=0, b=0 -> c=0
        entry_00 = self._encrypt_output(0, 0, 0)
        table_entries.append(entry_00)
        
        # 对于a=0, b=1 -> c=0
        entry_01 = self._encrypt_output(0, 1, 0)
        table_entries.append(entry_01)
        
        # 对于a=1, b=0 -> c=0
        entry_10 = self._encrypt_output(1, 0, 0)
        table_entries.append(entry_10)
        
        # 对于a=1, b=1 -> c=1
        entry_11 = self._encrypt_output(1, 1, 1)
        table_entries.append(entry_11)
        
        # 随机混淆表的顺序
        random.shuffle(table_entries)
        self.garbled_table = table_entries
    
    def _generate_XOR_table(self):
        """
        生成a ^ b函数的混淆表
        a ^ b的真值表:
        a=0, b=0 -> c=0
        a=0, b=1 -> c=1
        a=1, b=0 -> c=1
        a=1, b=1 -> c=0
        """
        # 为每种输入组合创建加密的输出
        table_entries = []
        
        # 对于a=0, b=0 -> c=0
        entry_00 = self._encrypt_output(0, 0, 0)
        table_entries.append(entry_00)
        
        # 对于a=0, b=1 -> c=1
        entry_01 = self._encrypt_output(0, 1, 1)
        table_entries.append(entry_01)
        
        # 对于a=1, b=0 -> c=1
        entry_10 = self._encrypt_output(1, 0, 1)
        table_entries.append(entry_10)
        
        # 对于a=1, b=1 -> c=0
        entry_11 = self._encrypt_output(1, 1, 0)
        table_entries.append(entry_11)
        
        # 随机混淆表的顺序
        random.shuffle(table_entries)
        self.garbled_table = table_entries
    
    def _encrypt_output(self, a_val, b_val, c_val):
        """
        使用a和b的密钥加密c的值
        a_val: a的值 (0或1)
        b_val: b的值 (0或1)
        c_val: c的值 (0或1)
        返回加密的输出
        """
        # 获取对应的密钥
        a_key = self.a_keys[a_val]
        b_key = self.b_keys[b_val]
        c_key = self.c_keys[c_val]
        
        # 使用a的密钥加密
        fernet_a = Fernet(a_key)
        intermediate = fernet_a.encrypt(c_key)
        
        # 使用b的密钥进一步加密
        fernet_b = Fernet(b_key)
        ciphertext = fernet_b.encrypt(intermediate)
        
        return {
            'a_val': a_val,
            'b_val': b_val,
            'ciphertext': ciphertext
        }
    
    def evaluate(self, a_key, b_key, garbled_table):
        """
        评估混淆电路
        a_key: 对应a输入的密钥
        b_key: 对应b输入的密钥
        garbled_table: 混淆表
        返回解密后的输出值
        """
        # 尝试每个表项
        for entry in garbled_table:
            try:
                # 使用b的密钥解密
                fernet_b = Fernet(b_key)
                intermediate = fernet_b.decrypt(entry['ciphertext'])
                
                # 使用a的密钥解密
                fernet_a = Fernet(a_key)
                c_key = fernet_a.decrypt(intermediate)
                
                # 检查解密的结果是否是有效的输出密钥
                if c_key == self.c_keys[0]:
                    return 0
                elif c_key == self.c_keys[1]:
                    return 1
            except Exception:
                # 如果解密失败，尝试下一个表项
                continue
        
        # 如果所有表项都解密失败，返回None
        return None

class Alice:
    """
    Alice方的实现，负责创建混淆电路
    """
    def __init__(self):
        self.circuit = GarbledCircuit()
        self.setup_data = None
    
    def setup_circuit(self, circuit_type="AND"):
        """设置混淆电路"""
        self.setup_data = self.circuit.setup(circuit_type)
        return self.setup_data
    
    def get_input_key(self, input_name, value):
        """获取特定输入的密钥"""
        if input_name == 'a':
            return self.setup_data['a_keys'][value]
        elif input_name == 'b':
            return self.setup_data['b_keys'][value]
        else:
            raise ValueError(f"Unknown input name: {input_name}")
    
    def get_garbled_table(self):
        """获取混淆表"""
        return self.setup_data['garbled_table']
    
    def send_to_bob(self, bob_input_name, bob_input_value):
        """向Bob发送他需要的信息"""
        # 在实际应用中，这将通过网络通信实现
        # 这里简化为直接返回
        return {
            'garbled_table': self.get_garbled_table(),
            'bob_key': self.get_input_key(bob_input_name, bob_input_value),
            'c_keys': self.setup_data['c_keys']  # 在实际应用中，只发送输出映射，而不是实际密钥
        }

class Bob:
    """
    Bob方的实现，负责评估混淆电路
    """
    def __init__(self):
        self.circuit = None
        self.a_key = None
        self.b_key = None
        self.garbled_table = None
        self.c_keys = None
    
    def receive_from_alice(self, data_from_alice, my_key):
        """从Alice接收信息"""
        self.garbled_table = data_from_alice['garbled_table']
        self.my_key = my_key
        self.c_keys = data_from_alice['c_keys']
        
    def evaluate_circuit(self, alice_key):
        """评估电路并获取结果"""
        try:
            # 尝试每个表项
            for entry in self.garbled_table:
                try:
                    # 使用Bob的密钥解密
                    fernet_bob = Fernet(self.my_key)
                    intermediate = fernet_bob.decrypt(entry['ciphertext'])
                    
                    # 使用Alice的密钥解密
                    fernet_alice = Fernet(alice_key)
                    c_key = fernet_alice.decrypt(intermediate)
                    
                    # 检查解密的结果是否是有效的输出密钥
                    if c_key == self.c_keys[0]:
                        return 0
                    elif c_key == self.c_keys[1]:
                        return 1
                except Exception:
                    # 如果解密失败，尝试下一个表项
                    continue
            
            # 如果所有表项都解密失败，返回None
            return None
        except Exception as e:
            print(f"评估电路时出错: {e}")
            return None

def demonstrate_garbled_circuit():
    """演示混淆电路的使用"""
    print("=== 混淆电路演示: AND(a,b) ===")
    
    # 初始化Alice和Bob
    alice = Alice()
    bob = Bob()
    
    # Alice设置电路
    alice.setup_circuit()
    print("Alice已设置电路")
    
    # 假设Alice的输入是a=0，Bob的输入是b=1
    alice_input_name = 'a'
    alice_input_value = 1
    bob_input_name = 'b'
    bob_input_value = 1
    
    print(f"Alice的输入: {alice_input_name}={alice_input_value}")
    print(f"Bob的输入: {bob_input_name}={bob_input_value}")
    
    # Alice获取她的输入密钥
    alice_key = alice.get_input_key(alice_input_name, alice_input_value)
    
    # Alice将混淆表和Bob的输入密钥发送给Bob
    data_for_bob = alice.send_to_bob(bob_input_name, bob_input_value)
    
    # Bob接收来自Alice的数据
    bob.receive_from_alice(data_for_bob, data_for_bob['bob_key'])
    
    # Bob评估电路
    result = bob.evaluate_circuit(alice_key)
    print(f"电路评估结果: AND({alice_input_value}, {bob_input_value}) = {result}")
    
    # 验证结果
    expected = alice_input_value & bob_input_value
    if result == expected:
        print("结果正确！")
    else:
        print(f"结果错误！预期是 {expected}")

if __name__ == "__main__":
    demonstrate_garbled_circuit()