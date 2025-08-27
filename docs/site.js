// SPA-like nav to keep background canvas alive
const menuBtn = document.querySelector('.menu-button');
const dropdown = document.querySelector('.dropdown-menu');
const header = document.querySelector('.site-nav');

function closeMenu() {
  dropdown?.classList.remove('show');
  if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
  if (dropdown) dropdown.setAttribute('aria-hidden', 'true');
}

menuBtn?.addEventListener('click', (e) => {
  e.stopPropagation();
  const willShow = !dropdown.classList.contains('show');
  dropdown.classList.toggle('show', willShow);
  menuBtn.setAttribute('aria-expanded', String(willShow));
  dropdown.setAttribute('aria-hidden', String(!willShow));
});

document.addEventListener('click', (e) => {
  if (!dropdown?.classList.contains('show')) return;
  if (!dropdown.contains(e.target) && e.target !== menuBtn) {
    closeMenu();
  }
});

function setActivePath(pathname) {
  document.querySelectorAll('.nav-links a, .dropdown-menu a').forEach(a => a.classList.remove('active'));
  const match = document.querySelectorAll(`.nav-links a[href="${pathname}"] , .dropdown-menu a[href="${pathname}"]`);
  if (match.length) match.forEach(a => a.classList.add('active'));
}

async function fetchPage(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error('Failed to fetch');
  const html = await res.text();
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const newMain = doc.querySelector('main.content');
  const title = doc.querySelector('title')?.textContent || document.title;
  return { newMain, title };
}

async function navigate(url, push = true) {
  try {
    closeMenu();
    const { newMain, title } = await fetchPage(url);
    if (!newMain) throw new Error('No main.content in target');
    const curMain = document.querySelector('main.content');
    curMain.replaceWith(newMain);
    document.title = title;
    const u = new URL(url, location.href);
    if (push) history.pushState({ url: u.pathname }, '', u.pathname);
    setActivePath(u.pathname);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (window.hydrateSiteData) window.hydrateSiteData();
  } catch (err) {
    location.assign(url);
  }
}

function isInternalNavLink(a) {
  if (!a || a.target === '_blank' || a.hasAttribute('download')) return false;
  const href = a.getAttribute('href');
  if (!href) return false;
  return [
    'index.html', '/index.html',
    'projects.html', '/projects.html',
    'hackathons.html', '/hackathons.html',
    'cv.html', '/cv.html'
  ].includes(href);
}

header?.addEventListener('click', (e) => {
  const a = e.target.closest('a');
  if (!a) return;
  if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  if (!isInternalNavLink(a)) return;
  e.preventDefault();
  navigate(a.getAttribute('href'));
});

document.addEventListener('click', (e) => {
  const a = e.target.closest('a');
  if (!a) return;
  if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  if (!isInternalNavLink(a)) return;
  e.preventDefault();
  navigate(a.getAttribute('href'));
});

window.addEventListener('popstate', (e) => {
  const path = location.pathname.endsWith('/') ? '/index.html' : location.pathname;
  navigate(path, false);
});

setActivePath(location.pathname.endsWith('/') ? '/index.html' : location.pathname); 
