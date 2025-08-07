# CZCE AI 应用

### 项目结构说明

|-- app                             #应用代码,比如 pipeline, api 接口等
|-- czce_ai                         #核心库,包含各种工具和库
    |-- s3                          #和对象存储交互,增删改查
    |-- document                    #各种文件读取,负责读取和解析文件(比如OCR等)
        |-- reader                  #从文件中读取到原始文本之后对文本进行分块操作
        |-- chunking                #分块策略
    |-- embedder                    # embedding 相关
    |-- models                      #模型相关
        |-- ds                      #类 deepseek 大语言模型调用
    |-- tokenizer                   # 分词相关
        |-- jieba.py                #jieba分词
    |-- utils                       #工具类
    |-- vectordb                    #向量数据库相关
        |-- milvus.py               # milvus 向量数据库适配
|-- notebooks                       # jupyter notebook 文件
|-- resources                       #资源文件,比如词典等
|-- test_data                       # 测试数据
    |-- pdf                         #测试 pdf 文件

### 开发

#### 1.安装依赖

更新 UV 至0.7.13 版本
```bash
pip install uv==0.1.13 --find-links /root/projects/ai-platform/packages/
```

同步开发环境

```bash
uv sync
```

安装新依赖
``
uv add package_name
```
