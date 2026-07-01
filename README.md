这是 GitHub Pages 静态站点仓库。根目录的 `index.html` 是“我的网页索引”，用于手工展示各个论文解读页面。

## 新增一篇论文解读时需要改什么

假设新论文解读的目录名是 `new-paper/`，要让它出现在“我的网页索引”里，需要完成下面几步。

### 1. 新建论文页面目录

在仓库根目录下新建一个独立文件夹，例如：

```text
new-paper/
  index.html
  paper.pdf
  images/
```

要求：

- `index.html` 是这个论文解读页面的入口。
- 图片、PDF 等资源放在同目录或子目录里，并用相对路径引用。
- 如果有 Markdown 源文档，可以保留在该目录下，但正式浏览页不应依赖运行时动态加载 Markdown；推荐把阅读笔记静态写入 `index.html`。

### 2. 更新根目录的网页索引

打开根目录的 `index.html`，找到：

```html
<section class="grid" aria-label="网页列表">
```

在现有论文卡片后面、`<div class="empty">` 占位卡片前面，新增一个卡片：

```html
<a class="card" href="/new-paper/">
  <span class="tag">论文笔记</span>
  <h2>新论文标题阅读笔记</h2>
  <span class="meta">
    <span>一句话描述这篇论文的主题</span>
    <span>打开页面</span>
  </span>
</a>
```

需要替换的内容：

- `href="/new-paper/"`：改成新论文文件夹路径，必须和目录名一致。
- `<h2>`：改成索引页上显示的标题。
- `.meta` 第一个 `<span>`：改成简短摘要或关键词。

注意：只新建 `new-paper/` 文件夹不会自动出现在索引页；必须手动在根 `index.html` 里加这段卡片。

### 3. 本地验证

推荐在仓库根目录启动本地静态服务：

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

然后访问：

```text
http://127.0.0.1:8765/
```

检查：

- “我的网页索引”里出现了新增卡片。
- 点击卡片能打开 `/new-paper/`。
- 新页面里的图片、PDF、CSS、JS 路径都能正常加载。

### 4. 提交和发布

确认无误后提交并推送：

```bash
git add index.html new-paper/
git commit -m "Add new paper reading note"
git push origin main
```

推送到 `main` 后，GitHub Pages 会按仓库配置自动发布。
