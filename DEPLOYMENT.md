# GitHub Pages Deployment Guide: Workflow Benchmark Suite Web Dashboard

This guide explains how to publish and maintain your professional dashboard using GitHub Pages. Choose **automatic** (recommended) or **manual** deployment, follow best practices, and troubleshoot common problems.

---

## Prerequisites
- A GitHub account and a repository created for this project
- [Git](https://git-scm.com/) installed locally
- Python 3.10+ installed
- [Poetry](https://python-poetry.org/) installed (for dependency management)

---

## Option 1: Automatic Deployment via GitHub Actions (Recommended)

1. Ensure `.github/workflows/deploy.yml` exists in your repository (see source for configuration).
2. Push your code to the `main` branch:
   ```bash
   git add .
   git commit -m "Update and deploy web interface"
   git push origin main
   ```
3. On GitHub, go to your repository → **Actions** tab. The "Deploy Web Dashboard" workflow will run automatically.
4. Wait for the workflow to complete (3–5 minutes typical). View detailed logs by clicking the workflow run.
5. When successful, go to **Settings → Pages** in your repository.
6. Ensure the source is set to the `gh-pages` branch and root folder.
7. Note the published site URL (e.g. `https://yourusername.github.io/repo-name`).
8. Visit the URL in your browser. The production version should now be live!

**Tips:**
- The deployment runs every time you push to main or trigger manually from the Actions tab (via "Run workflow").
- If the workflow fails, check the logs for Python/Poetry/file path issues, and ensure your results files are generated.

---

## Option 2: Manual Deployment

_Manual deployment is helpful if you want to control exactly what gets published or GitHub Actions is misconfigured._

1. Build project locally:
   ```bash
   python main.py
   bash scripts/prepare_web_data.sh
   ```
2. Verify `web/data/` and `web/images/` have valid files (e.g., web_results.json, PNG charts).
3. Create orphan branch for a clean deploy:
   ```bash
   git checkout --orphan gh-pages
   git rm -rf .
   cp -R web/* .
   git add .
   git commit -m "Deploy to GitHub Pages"
   git push -f origin gh-pages
   git checkout main
   ```
4. Enable GitHub Pages in repository settings
   - Source: `gh-pages` branch
   - Folder: `/` (root)
   - Save settings and note the public URL

**Warning:** Manual deploy will overwrite the gh-pages branch. Do NOT use on repos with other content on this branch.

---

## Option 3: Deploy From Local Build (Advanced)
1. Build the dashboard as above.
2. Use [gh-pages npm package](https://www.npmjs.com/package/gh-pages) or equivalent tool to publish `web/`.
3. Follow your tool's CLI instructions for branch and directory configuration.

---

## GitHub Pages Configuration
- Go to **Settings → Pages** in your GitHub repository
- Source: "Deploy from a branch"
- Branch: `gh-pages`, Folder: `/` (root)
- [Optional] **Custom Domain**: add domain, set CNAME, and DNS as detailed below
- Always **enforce HTTPS** for security

---

## Custom Domain Setup (Optional)
- In `web/`, create a file named `CNAME` containing your domain (e.g., `benchmark.yourdomain.com`)
- Add `CNAME` to your repo before deployment
- At your domain registrar, add a CNAME DNS record: 
    - host: your subdomain (e.g. `benchmark`)
    - value: `yourusername.github.io.` (trailing dot optional)
- In repository **Settings → Pages**, set and save your custom domain
- Wait for DNS propagation (can take hours)
- Enable HTTPS once GitHub shows it's available

---

## Updating the Site
- **Auto**: Any new push to the `main` branch will trigger a build and redeploy (via Actions)
- **Manual**: Repeat Option 2 steps. Always confirm the live site updates.
- **Recommended**: Check `web/data/` and `web/images/` before each deployment.

---

## Troubleshooting
- **404 error**: Check `gh-pages` branch exists and contains files; repository settings must point to this branch.
- **Workflow fails**: Confirm the correct Python version, Poetry is installed, all files exist; view logs in Actions tab.
- **Old content showing**: Browser caching can delay updates; force refresh or clear cache. Sometimes GitHub delays CDN updates by minutes/hours.
- **CSS/JS not loading**: Ensure file paths in HTML are relative (not absolute or starting with `/`).
- **Images/PNGs missing**: Confirm PNG files exist in `web/images/` and are referenced correctly in HTML.
- **JSON/data fetch errors**: Confirm `web/data/web_results.json` exists and is valid. Check for local CORS issues when testing, but GitHub Pages will allow JS fetches of colocated files.
- **Build logs**: View and re-run failed builds in Actions → Deploy Web Dashboard run.
- **Rollback**: Make an older commit on main or force push a previous build to `gh-pages`.

---

## Testing Before Deployment
1. Serve locally (`python web/serve.py`)
2. Test page navigation, links, and section anchors
3. View on desktop and mobile (browser dev tools)
4. Try Chrome, Firefox, Safari at least
5. Confirm all charts, data, and summary load
6. Check the browser console for errors
7. Validate accessibility: tab through nav, check skip links, screen reader basics

---

## See Also
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [GitHub Actions Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

