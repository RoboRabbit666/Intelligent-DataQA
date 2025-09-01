#!/usr/bin/env python
# coding: utf-8

import os
import sys

# Add project directory to sys.path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_dir)

import czce_ai.document.reader.excel_reader as er
from app.core.rag.workflow import RagWorkflow
from czce_ai.llm.message import Message
import requests
import json


def test_post():
    """测试rag_workflow_api的功能
    说明:
        运行该函数之前先运行app/rag_workflow/下面的service_lancher.py来启动服务
    """
    url = "http://localhost:7996/knoq/chat/completions"
    url_health = "http://localhost:7996/knoq/health"
    headers = {"Content-Type": "application/json"}
    response_health = requests.get(url_health, headers=headers)
    print(response_health.json())
    out_stream = True
    data = {
        "messages": [
            {"role": "system", "content": "你是一个制度专家,回答制度相关问题"},
            {"role": "user", "content": "美国夏威夷的出差标准是什么?"},
        ],
        "knowledge_base_ids": ["3cc33ed2-21fb-4452-9e10-528867bd5f98"],
        "stream": out_stream,
        "model": "ragmodel",
    }

    if not out_stream:
        response = requests.post(url, headers=headers, json=data)
        print(response.json())
    else:
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            print(response)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data:"):
                        content = decoded_line[6:]
                        if content == "[DONE]":
                            print("\n流式输出完成")
                        else:
                            chunk = json.loads(content)
                            # print(chunk)
                            if chunk["object"] == "chat.completion.chunk":
                                if len(chunk["choices"]) == 0:
                                    continue
                                if (
                                    chunk["choices"][0]["delta"].get(
                                        "reasoning_content"
                                    )
                                ) is not None:
                                    print(
                                        chunk["choices"][0]["delta"].get(
                                            "reasoning_content"
                                        ),
                                        end="",
                                        flush=True,
                                    )
                                if (
                                    chunk["choices"][0]["delta"].get("content")
                                ) is not None:
                                    print(
                                        chunk["choices"][0]["delta"].get("content"),
                                        end="",
                                        flush=True,
                                    )


if __name__ == "__main__":
    # call_excel_reader()
    # call_rag_workflow()
    test_post()