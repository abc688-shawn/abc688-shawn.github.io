const articleEl = document.querySelector("#article");
const tocEl = document.querySelector("#toc");
const titleEl = document.querySelector("#paper-title");
const summaryEl = document.querySelector("#hero-summary");
const metaEl = document.querySelector("#meta-strip");
const progressEl = document.querySelector(".read-progress");

const state = {
  headings: []
};

fetch("./README.md")
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Markdown request failed: ${response.status}`);
    }
    return response.text();
  })
  .then((markdown) => {
    const doc = prepareDocument(markdown);
    renderHero(doc);
    articleEl.innerHTML = renderMarkdown(doc.body);
    renderToc();
    bindScrollState();
    if (window.MathJax) {
      window.MathJax.typesetPromise([articleEl]);
    }
  })
  .catch((error) => {
    articleEl.innerHTML = `
      <blockquote>
        <p>Markdown 没有载入成功。请通过本地服务器打开这个页面，而不是直接用 file:// 打开。</p>
        <p><code>${escapeHtml(error.message)}</code></p>
      </blockquote>
    `;
  });

function prepareDocument(markdown) {
  const lines = markdown.split(/\r?\n/);
  const metaLines = [];
  let index = 0;

  while (index < lines.length && lines[index].startsWith(">")) {
    metaLines.push(lines[index]);
    index += 1;
  }

  while (index < lines.length && lines[index].trim() === "") {
    index += 1;
  }

  const meta = {};
  metaLines.forEach((line) => {
    const match = line.match(/^>\s*\*\*([^*]+)\*\*[：:]\s*(.+?)\s*$/);
    if (match) {
      meta[match[1].trim()] = match[2].trim();
    }
  });

  let body = lines.slice(index).join("\n").trimStart();
  const tldrMatch = body.match(/## TL;DR\s*\n+([\s\S]*?)(?=\n##\s+)/);
  const tldr = tldrMatch ? tldrMatch[1].trim().replace(/\n+/g, " ") : "";
  body = body.replace(/## TL;DR\s*\n+[\s\S]*?(?=\n##\s+)/, "").trimStart();

  return { meta, tldr, body };
}

function renderHero(doc) {
  const paperTitle = doc.meta["论文"] || "GLM-5: from Vibe Coding to Agentic Engineering";
  titleEl.textContent = paperTitle;
  summaryEl.innerHTML = renderInline(doc.tldr || "从 vibe coding 到 agentic engineering 的模型系统解剖。");

  const keys = ["作者", "版本", "阅读时长", "难度"];
  metaEl.innerHTML = keys
    .filter((key) => doc.meta[key])
    .map((key) => `
      <div>
        <dt>${escapeHtml(key)}</dt>
        <dd>${renderInline(doc.meta[key])}</dd>
      </div>
    `)
    .join("");
}

function renderMarkdown(markdown) {
  state.headings = [];
  const lines = markdown.split(/\r?\n/);
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i += 1;
      continue;
    }

    const fence = line.match(/^```(\w+)?\s*$/);
    if (fence) {
      const lang = fence[1] || "";
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(`<pre><code class="language-${escapeAttr(lang)}">${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    if (line.trim() === "$$") {
      const math = [];
      i += 1;
      while (i < lines.length && lines[i].trim() !== "$$") {
        math.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(`<div class="math-block">$$\n${escapeHtml(math.join("\n"))}\n$$</div>`);
      continue;
    }

    const heading = line.match(/^(#{2,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const title = stripInline(heading[2]);
      const id = uniqueSlug(title);
      state.headings.push({ id, title, level });
      blocks.push(`<h${level} id="${id}">${renderInline(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    const image = line.match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
    if (image) {
      blocks.push(`
        <figure>
          <img src="${escapeAttr(image[2])}" alt="${escapeAttr(image[1])}">
          ${image[1] ? `<figcaption>${escapeHtml(image[1])}</figcaption>` : ""}
        </figure>
      `);
      i += 1;
      continue;
    }

    if (line.startsWith(">")) {
      const quote = [];
      while (i < lines.length && lines[i].startsWith(">")) {
        quote.push(lines[i].replace(/^>\s?/, ""));
        i += 1;
      }
      blocks.push(`<blockquote>${renderParagraphs(quote.join("\n"))}</blockquote>`);
      continue;
    }

    if (isTableStart(lines, i)) {
      const tableLines = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        tableLines.push(lines[i]);
        i += 1;
      }
      blocks.push(renderTable(tableLines));
      continue;
    }

    if (/^\s*-\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*-\s+/, ""));
        i += 1;
      }
      blocks.push(`<ul>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push(`<ol>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ol>`);
      continue;
    }

    const paragraph = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !isSpecialStart(lines, i)
    ) {
      paragraph.push(lines[i]);
      i += 1;
    }
    blocks.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
  }

  return blocks.join("\n");
}

function isSpecialStart(lines, index) {
  const line = lines[index] || "";
  return (
    /^```/.test(line) ||
    /^#{2,4}\s+/.test(line) ||
    /^!\[/.test(line) ||
    /^>/.test(line) ||
    /^\s*-\s+/.test(line) ||
    /^\s*\d+\.\s+/.test(line) ||
    line.trim() === "$$" ||
    isTableStart(lines, index)
  );
}

function isTableStart(lines, index) {
  return (
    /^\s*\|.*\|\s*$/.test(lines[index] || "") &&
    /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1] || "")
  );
}

function renderTable(lines) {
  const rows = lines
    .filter((line, idx) => idx !== 1)
    .map((line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim()));
  const [head, ...body] = rows;
  return `
    <div class="table-wrap">
      <table>
        <thead><tr>${head.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead>
        <tbody>
          ${body.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderParagraphs(text) {
  return text
    .split(/\n{2,}/)
    .map((part) => `<p>${renderInline(part.replace(/\n/g, " "))}</p>`)
    .join("");
}

function renderInline(input) {
  const codeParts = [];
  let text = String(input).replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE_${codeParts.length}@@`;
    codeParts.push(`<code>${escapeHtml(code)}</code>`);
    return token;
  });

  text = escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
      return `<a href="${escapeAttr(href)}">${label}</a>`;
    });

  codeParts.forEach((code, index) => {
    text = text.replace(`@@CODE_${index}@@`, code);
  });

  text = text.replace(/\$(?=\d)/g, "\\$");

  return text;
}

function stripInline(text) {
  return String(text)
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*`_]/g, "")
    .trim();
}

const slugCounts = new Map();

function uniqueSlug(title) {
  const base = title
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "section";
  const count = slugCounts.get(base) || 0;
  slugCounts.set(base, count + 1);
  return count ? `${base}-${count + 1}` : base;
}

function renderToc() {
  tocEl.innerHTML = state.headings
    .filter((heading) => heading.level <= 3)
    .filter((heading) => !(heading.level === 3 && /^Q\d+:/i.test(heading.title)))
    .map((heading) => `
      <a href="#${heading.id}" class="depth-${heading.level}" data-target="${heading.id}">
        ${escapeHtml(heading.title)}
      </a>
    `)
    .join("");
}

function bindScrollState() {
  const tocLinks = Array.from(document.querySelectorAll("[data-target]"));
  const headings = tocLinks
    .map((link) => document.getElementById(link.dataset.target))
    .filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          tocLinks.forEach((link) => link.classList.toggle("active", link.dataset.target === entry.target.id));
        }
      });
    },
    { rootMargin: "-24% 0px -66% 0px", threshold: 0.01 }
  );

  headings.forEach((heading) => observer.observe(heading));
  window.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();
}

function updateProgress() {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = max > 0 ? window.scrollY / max : 0;
  progressEl.style.width = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/'/g, "&#39;");
}
