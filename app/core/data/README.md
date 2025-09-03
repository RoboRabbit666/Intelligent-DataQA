## DataQA Pipeline

### Overview
1. 问题改写/追问处理  
2. 实体识别增强  
3. 业务知识搜索（提供背景理解）  
4. 并行快速路径检查：  
   - 4a. FAQ匹配（阈值 ≥ 0.9）  
   - 4b. API定位（阈值 ≥ 0.8，**API优先**）  
5. 表格定位（仅当 4a 与 4b 均不满足时执行）  
6. SQL生成  
   - 集成业务知识上下文  
   - 集成表格 schema  
   - FAQ 分数在 0.7–0.9：仅作参考示例  
   - FAQ < 0.7：不使用 FAQ

### Fast-path decision table
| API ≥ 0.8 | FAQ ≥ 0.9 | 结果 |
|---|---|---|
| 是 | 是 | 返回 API（**API 优先**） |
| 是 | 否 | 返回 API |
| 否 | 是 | 返回 FAQ |
| 否 | 否 | 进入步骤 5（表格定位） |

### End-to-end flow (Mermaid)
```mermaid
flowchart TB
    A[1. 问题改写/追问处理] --> B[2. 实体识别增强]
    B --> C[3. 业务知识搜索（背景理解）]

    %% 并行快速路径（API 优先）
    C --> E{4b. API 定位<br/>分数 ≥ 0.8 ?}
    C --> D{4a. FAQ 匹配<br/>分数 ≥ 0.9 ?}

    E -->|是| K{FAQ ≥ 0.9 ?}
    K -->|是| F[返回 API 调用<br/>（流程结束，API 优先）]
    K -->|否| F[返回 API 调用<br/>（流程结束）]

    E -->|否| D
    D -->|是| G[返回 FAQ 答案/SQL<br/>（流程结束）]
    D -->|否| H[5. 表格定位]

    H --> I[6. SQL 生成<br/>• 业务知识上下文<br/>• 表格 schema<br/>• FAQ 0.7–0.9 仅作示例<br/>• FAQ < 0.7 不使用 FAQ]