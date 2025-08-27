async function loadData() {
	const res = await fetch('site-data.json', { cache: 'no-cache' });
	if (!res.ok) throw new Error('Failed to load site-data.json');
	return res.json();
  }
  
  function fmtDuration(startISO, endISO) {
	const start = new Date(startISO);
	const end = endISO ? new Date(endISO) : new Date();
	let years = end.getFullYear() - start.getFullYear();
	let months = end.getMonth() - start.getMonth();
	if (months < 0) { years -= 1; months += 12; }
	const parts = [];
	if (years > 0) parts.push(`${years} year${years !== 1 ? 's' : ''}`);
	if (months > 0) parts.push(`${months} month${months !== 1 ? 's' : ''}`);
	if (parts.length === 0) parts.push('0 months');
	return parts.join(', ');
  }
  
  function fmtMonthYear(d) {
	const dt = (d instanceof Date) ? d : new Date(d);
	const mon = dt.toLocaleString(undefined, { month: 'short' });
	const yr = String(dt.getFullYear());
	return `${mon} ${yr}`;
  }
  
  function sameMonthYear(aISO, bISO) {
	if (!aISO || !bISO) return false;
	const a = new Date(aISO);
	const b = new Date(bISO);
	return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth();
  }
  
  function el(html) {
	const t = document.createElement('template');
	t.innerHTML = html.trim();
	return t.content.firstElementChild;
  }
  
  function iconFor(type) {
	switch (type) {
	  case 'page':
		return 'https://img.icons8.com/?size=100&id=69143&format=png&color=000000';
	  case 'arxiv':
		return 'https://cdn.simpleicons.org/arxiv/000000';
	  case 'github':
		return 'https://cdn.simpleicons.org/github/000000';
	  case 'devpost':
		return 'https://cdn.simpleicons.org/devpost/000000';
	  default:
		return 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/link-45deg.svg';
	}
  }
  
  function renderProject(p) {
	const venue = p.venue || {};
	const badge = venue.url && !venue.muted
	  ? `<a class="badge-venue" href="${venue.url}" target="_blank" rel="noreferrer">${venue.text}</a>`
	  : `<div class="badge-venue ${venue.muted ? 'badge-venue--muted' : ''}">${venue.text || ''}</div>`;
	const authors = (p.authors || []).map(a => {
	  const cls = a.me ? 'author-me' : '';
	  const href = a.url || '#';
	  return `<a class="${cls}" href="${href}">${a.name}</a>`;
	}).join(', ');
	const links = (p.links || []).map(l => `
	  <a class="proj-link" href="${l.url}" target="_blank" rel="noreferrer">
		<img width="16" height="16" src="${iconFor(l.type)}" alt="${l.text}" />
		<span>${l.text}</span>
	  </a>`).join('');
	return el(`
	  <article class="project-card">
		<div class="proj-right">
		  <img class="proj-media" src="${p.media}" alt="Project preview" />
		</div>
		<div class="proj-left">
		  <h3 class="proj-title">${p.title}</h3>
		  <div class="proj-authors">${authors}</div>
		  ${badge}
		  <div class="proj-links">${links}</div>
		</div>
	  </article>
	`);
  }
  
  function renderExperience(e) {
	const endText = e.end ? fmtMonthYear(e.end) : 'Present';
	const startText = fmtMonthYear(e.start);
	const dateText = (e.end && sameMonthYear(e.start, e.end)) ? startText : `${startText} – ${endText}`;
	const duration = fmtDuration(e.start, e.end);
	return el(`
	  <article class="experience-item">
		<div class="exp-left">
		  <img class="exp-logo" src="${e.logo || 'assets/company.png'}" alt="Company logo" />
		</div>
		<div class="exp-right">
		  <h3 class="exp-title">${e.title}</h3>
		  <div><span class="exp-company">${e.company}</span> · <span class="exp-type">${e.type || ''}</span></div>
		  <div class="exp-meta">${dateText} · ${duration}</div>
		  <div class="exp-location">${e.location || ''}</div>
		  <p class="exp-desc">${e.description || ''}</p>
		</div>
	  </article>
	`);
  }
  
  function renderEducation(ed) {
	const endText = ed.end ? fmtMonthYear(ed.end) : 'Present';
	const startText = fmtMonthYear(ed.start);
	const dateText = (ed.end && sameMonthYear(ed.start, ed.end)) ? startText : `${startText} – ${endText}`;
	return el(`
	  <article class="experience-item">
		<div class="exp-left">
		  <img class="exp-logo" src="${ed.logo || 'assets/school.png'}" alt="Institution logo" />
		</div>
		<div class="exp-right">
		  <h3 class="exp-title">${ed.title}</h3>
		  <div><span class="exp-company">${ed.company}</span> · <span class="exp-type">${ed.type || ''}</span></div>
		  <div class="exp-meta">${dateText}</div>
		  <div class="exp-location">${ed.location || ''}</div>
		  <p class="exp-desc">${ed.description || ''}</p>
		</div>
	  </article>
	`);
  }
  
  function renderHonor(h) {
	const dateText = fmtMonthYear(h.date);
	return el(`
	  <li class="honor-item">
		<div><span class="honor-title">${h.title}</span> — <span class="honor-place">${h.place}</span></div>
		<div class="honor-date">${dateText}</div>
		<div class="honor-desc">${h.description || ''}</div>
	  </li>
	`);
  }
  
  function renderHackathon(h) {
	const dateText = fmtMonthYear(h.date);
	const imgs = (h.images || []).slice(0, 2);
	const mediaClass = imgs.length === 1 ? 'hack-media single' : 'hack-media';
	const media = imgs.map(src => `<img src="${src}" alt="Hackathon media" />`).join('');
	const links = (h.links || []).map(l => `
	  <a class="hack-link" href="${l.url}" target="_blank" rel="noreferrer">
		<img width="16" height="16" src="${iconFor(l.type)}" alt="${l.text}" />
		<span>${l.text}</span>
	  </a>`).join('');
	const extra = [h.duration ? `Duration: ${h.duration}` : '', h.challenger ? `Challenger: ${h.challenger}` : '']
	  .filter(Boolean).join(' · ');
	return el(`
	  <article class="hack-card">
		<div class="hack-right">
		  <div class="${mediaClass}">${media}</div>
		</div>
		<div class="hack-left">
		  <h3 class="hack-title">${h.title}</h3>
		  <div class="hack-meta">${h.place} · ${dateText}</div>
		  ${extra ? `<div class="hack-extra">${extra}</div>` : ''}
		  <p class="hack-desc">${h.desc || ''}</p>
		  <div class="hack-links">${links}</div>
		</div>
	  </article>
	`);
  }
  
  async function fetchLastCommitDate() {
	try {
	  const res = await fetch('https://api.github.com/repos/Pikurrot/Pikurrot.github.io/commits?per_page=1', { headers: { 'Accept': 'application/vnd.github+json' } });
	  if (!res.ok) throw new Error('GitHub API error');
	  const data = await res.json();
	  const date = new Date(data[0]?.commit?.committer?.date || Date.now());
	  return fmtMonthYear(date);
	} catch (e) {
	  return fmtMonthYear(new Date());
	}
  }
  
  async function renderFooter() {
	const el = document.getElementById('site-footer');
	if (!el) return;
	const lastUpdated = await fetchLastCommitDate();
	el.innerHTML = `
	  <div>© ${new Date().getFullYear()} Eric López — All rights reserved.</div>
	  <small>Last updated: ${lastUpdated}</small>
	  <small>Template created by myself.</small>
	`;
  }
  
  // expose hydrate for SPA
  window.hydrateSiteData = async function hydrateSiteData() {
	try {
	  const data = await loadData();
	  const featured = document.getElementById('featured-projects');
	  if (featured) {
		featured.innerHTML = '';
		data.projects.slice(0, 2).forEach(p => featured.appendChild(renderProject(p)));
	  }
	  const projPage = document.getElementById('projects-page-list');
	  if (projPage) {
		projPage.innerHTML = '';
		data.projects.forEach(p => projPage.appendChild(renderProject(p)));
	  }
	  const hackList = document.getElementById('hackathons-list');
	  if (hackList) {
		hackList.innerHTML = '';
		data.hackathons.forEach(h => hackList.appendChild(renderHackathon(h)));
	  }
	  const expList = document.getElementById('experience-list');
	  if (expList) {
		expList.innerHTML = '';
		data.experience.forEach(e => expList.appendChild(renderExperience(e)));
	  }
	  const eduList = document.getElementById('education-list');
	  if (eduList) {
		eduList.innerHTML = '';
		data.education.forEach(ed => eduList.appendChild(renderEducation(ed)));
	  }
	  const honorsList = document.getElementById('honors-list');
	  if (honorsList) {
		honorsList.innerHTML = '';
		data.honors.forEach(h => honorsList.appendChild(renderHonor(h)));
	  }
	  await renderFooter();
	} catch (e) {
	  console.error(e);
	}
  };
  
  // initial hydrate
window.hydrateSiteData(); 
