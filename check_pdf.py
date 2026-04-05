from src.data_loader import DataLoader

loader = DataLoader()
doc = loader.load_document('data/raw/python-doc-27-34.pdf')

# 找到源文件编码相关段落
idx = doc.content.find('源程序的编码')
if idx == -1:
    idx = doc.content.find('源文件')
print("找到位置:", idx)
print(doc.content[idx:idx+500])
