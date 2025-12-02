# Git 仓库设置说明

由于终端环境问题，请按照以下步骤手动执行命令来连接远程仓库并上传代码：

## 步骤 1: 初始化 Git 仓库

在项目根目录下打开命令行（PowerShell 或 CMD），执行：

```bash
git init
```

## 步骤 2: 添加远程仓库

```bash
git remote add origin https://github.com/YZA114514/Grounding-Dino-Light.git
```

如果远程仓库已存在，可以使用以下命令更新：

```bash
git remote set-url origin https://github.com/YZA114514/Grounding-Dino-Light.git
```

## 步骤 3: 添加所有文件

```bash
git add .
```

## 步骤 4: 提交文件

```bash
git commit -m "Initial commit: Add GroundingDINO project files"
```

## 步骤 5: 设置主分支并推送

```bash
git branch -M main
git push -u origin main
```

## 注意事项

1. 如果远程仓库不为空，可能需要先拉取：
   ```bash
   git pull origin main --allow-unrelated-histories
   ```

2. 如果遇到认证问题，可能需要配置 GitHub 的访问令牌（Personal Access Token）

3. 如果推送失败，可以尝试强制推送（谨慎使用）：
   ```bash
   git push -u origin main --force
   ```

## 或者使用批处理脚本

我已经创建了 `setup_git.bat` 文件，您可以直接双击运行它来自动执行上述所有步骤。

