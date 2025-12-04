import json
import requests

# 测试数据生成函数
def generate_test_request():
    """生成符合 Request 协议的测试数据"""
    request_data = {
        "userId": "user_12345",
        "features": '{"u_r_click":{"type":5,"value":["click-0","click-1","click-2","click-3","click-4","click-5","click-6","click-7","click-8","click-9","click-10","click-11","click-12","click-13","click-14","click-15","click-16","click-17","click-18","click-19","click-20","click-21","click-22","click-23","click-24","click-25","click-26","click-27","click-28","click-29","click-30","click-31","click-32","click-33","click-34","click-35","click-36","click-37","click-38","click-39","click-40","click-41","click-42","click-43","click-44","click-45","click-46","click-47","click-48","click-49","click-50","click-51","click-52","click-53","click-54","click-55","click-56","click-57","click-58","click-59","click-60","click-61","click-62","click-63","click-64","click-65","click-66","click-67","click-68","click-69","click-70","click-71","click-72","click-73","click-74","click-75","click-76","click-77","click-78","click-79","click-80","click-81","click-82","click-83","click-84","click-85","click-86","click-87","click-88","click-89","click-90","click-91","click-92","click-93","click-94","click-95","click-96","click-97","click-98","click-99"]},"embedding":{"type":4,"value":[0.1,0.2,0.3,0.4,0.5]}}',
        "entries": [
            {
                "id": "item1",
            },
            {
                "id": "item1",
            },{
                "id": "item1",
            },
            {
                "id": "item1",
            }
        ],
    }
    return request_data



# 测试用例类
class RankServiceTest:
    def __init__(self, base_url="http://localhost:9528"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/v1/rank"
    
    def test_rank_request(self):
        """测试 Rank 服务请求"""
        print("=" * 60)
        print("测试 Rank 服务请求")
        print("=" * 60)
        
        # 生成测试请求数据
        request_data = generate_test_request()
        
        # 打印请求 JSON
        print("\n📤 请求 JSON:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))
        
        try:
            # 发送 POST 请求
            response = requests.post(
                self.endpoint,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # 打印响应
            print(f"\n📥 响应状态码: {response.status_code}")
            print("\n响应 JSON:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ 请求失败: {e}")
            return None


# 主测试函数
def main():
    print("\n🚀 Rank Service JSON 测试工具")
    print("=" * 60)
    
    # 创建测试实例
    tester = RankServiceTest()

    tester.test_rank_request()
    
   


if __name__ == "__main__":
    main()
