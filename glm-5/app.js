const articleEl = document.querySelector("#article");
const tocEl = document.querySelector("#toc");
const progressEl = document.querySelector(".read-progress");

const headings = Array.from(articleEl.querySelectorAll("h2, h3, h4"))
  .map((heading) => ({
    id: heading.id,
    title: heading.textContent.trim(),
    level: Number(heading.tagName.slice(1))
  }))
  .filter((heading) => heading.id);

renderToc();
bindScrollState();

if (window.MathJax) {
  window.MathJax.typesetPromise([articleEl]);
}

function renderToc() {
  tocEl.innerHTML = headings
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
  const observedHeadings = tocLinks
    .map((link) => document.getElementById(link.dataset.target))
    .filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          tocLinks.forEach((link) => {
            link.classList.toggle("active", link.dataset.target === entry.target.id);
          });
        }
      });
    },
    { rootMargin: "-24% 0px -66% 0px", threshold: 0.01 }
  );

  observedHeadings.forEach((heading) => observer.observe(heading));
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
