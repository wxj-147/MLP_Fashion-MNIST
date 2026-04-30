# 环境依赖
python版本为3.12.8  
第三方库包括numpy、matplotlib  
另外，还包括python内置标准库：os、gzip、urllib.request、pickle、json、time、typing（无需额外安装）

### 可使用虚拟环境
python -m venv venv  
source venv/bin/activate # Linux/Mac  
venv\Scripts\activate # Windows  
pip install numpy matplotlib # 安装依赖

# 脚本运行
train.py中在最后注释了一个训练函数的测试代码，用于检测训练代码是否正常运行  
完整的训练主函数在random_search.py中，若要重复实验只需运行该脚本
