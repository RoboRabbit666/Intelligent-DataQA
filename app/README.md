# AI 中台服务程序

## 项目目录结构

app/
|-- api
|   |-- main.py           # fastapi 程序入口
|   |-- routers           # 路由模块
|       |-- knoq.py         # 知识问答入口
|-- config
|   |-- config.py         # 配置文件管理模块,通过env配置
|-- core
|   |-- components.py     # 核心模块,业务逻辑
|   |-- rag               # 组件模块,共用组件,包括milvus,llm等
|       |-- prompt.py     # rag 相关逻辑
|       |-- workflow.py   # rag workflow
|-- scripts
|   |-- service_launcher.py # 服务启动脚本
|-- utils
    |-- log.py

## 开发

### 启动服务

```bash
cd app
python scripts/service_launcher.py