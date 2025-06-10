## Day1

### Python 核心内容

#### 环境搭建
- 查看 Python 版本：`python --version`
- 创建虚拟环境：`venv` 模块
- 安装库：`pip install`

#### 数据类型
- 基本数据类型：`int`、`float`、`str`、`bool`
- 容器类型：`list`、`tuple`、`dict`、`set`
- 注意作用域与类型转换

#### 运算符
- 算术运算符：`+`、`-`、`*`、`/` 等
- 比较运算符：`==`、`>` 等
- 逻辑运算符：`and`、`or` 等

#### 控制流
- 分支结构：`if`
- 循环结构：`for`、`while`
- 异常处理：`try-except`

#### 函数
- 定义函数：`def`（支持默认参数、`*args`）
- 匿名函数：`lambda`
- 高阶函数

#### 模块包
- 导入模块：`import`
- `.py` 文件为模块，包含 `__init__.py` 的文件夹为包

#### OOP（面向对象编程）
- 定义类：`class`（`__init__` 初始化方法）
- 继承与方法重写

#### 装饰器
- 使用 `@` 语法糖
- 高阶函数修改函数行为

#### 文件操作
- 读写文本文件：`with open`
- 处理 CSV/JSON 数据

### Git 关键命令

#### 初始化
- `git init`

#### 提交流程
1. `git add .`
2. `git commit -m "msg"`

#### 远程操作
- 添加远程仓库：`git remote add origin url`
- 拉取/推送代码：`git pull/push origin main`

#### 配置
- 设置全局用户名和邮箱：`git config --global user.name/email`