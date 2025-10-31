# Deployment Guide
## How to Deploy and Share Your Workflow Optimization Benchmark Suite

This guide provides step-by-step instructions for deploying your project so others can view, test, and interact with it.

---

## Table of Contents

1. [Quick Start - Local Sharing](#quick-start---local-sharing)
2. [Cloud Deployment Options](#cloud-deployment-options)
3. [GitHub Pages Deployment](#github-pages-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Platform Deployments](#cloud-platform-deployments)
6. [Sharing Pre-Generated Results](#sharing-pre-generated-results)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start - Local Sharing

### Option 1: Simple Local Network Sharing (Easiest)

**Use Case**: Share with people on the same network (e.g., office, home WiFi)

#### Step 1: Generate Benchmark Results

```bash
# Run benchmarks to generate data
./run_and_serve.sh

# Or manually:
poetry run python main.py
bash scripts/prepare_web_data.sh
```

#### Step 2: Find Your IP Address

**macOS/Linux**:
```bash
# Find your local IP
ifconfig | grep "inet " | grep -v 127.0.0.1

# Or use:
hostname -I
```

**Windows**:
```bash
ipconfig
# Look for IPv4 Address under your network adapter
```

#### Step 3: Start Server on Network Interface

```bash
cd web
python serve.py --port 8000
# Or use your IP explicitly:
python -c "import http.server; import socketserver; socketserver.TCPServer(('0.0.0.0', 8000), http.server.SimpleHTTPRequestHandler).serve_forever()"
```

#### Step 4: Share the URL

Share this URL with others on your network:
```
http://YOUR_IP_ADDRESS:8000
```

**Example**: `http://192.168.1.100:8000`

**Important**: 
- Make sure firewall allows connections on port 8000
- Others must be on the same network
- Server stops when you close the terminal

---

### Option 2: Using ngrok (Best for Remote Access)

**Use Case**: Share with anyone, anywhere (even remote users)

#### Step 1: Install ngrok

```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download
```

#### Step 2: Generate Results

```bash
./run_and_serve.sh
# Keep the server running, or in a separate terminal:
cd web && python serve.py --port 8000
```

#### Step 3: Create ngrok Tunnel

```bash
# In a new terminal
ngrok http 8000
```

You'll see output like:
```
Forwarding  https://abc123.ngrok.io -> http://localhost:8000
```

#### Step 4: Share the ngrok URL

Share the `https://abc123.ngrok.io` URL with anyone.

**Benefits**:
- ✅ Works from anywhere (not just local network)
- ✅ HTTPS automatically enabled
- ✅ Free tier available
- ✅ Easy to set up

**Limitations**:
- Free tier: URL changes each time you restart ngrok
- Free tier: Limited connections per minute

---

## Cloud Deployment Options

### Option 1: GitHub Pages (Free, Static Hosting)

**Best For**: Public sharing, documentation, demo purposes

#### Step 1: Prepare Web Assets

```bash
# 1. Generate benchmark results
poetry run python main.py
bash scripts/prepare_web_data.sh

# 2. Ensure web/data/web_results.json exists
ls -la web/data/web_results.json
```

#### Step 2: Create GitHub Pages Branch

```bash
# Create a gh-pages branch
git checkout -b gh-pages

# Copy web files to root (for GitHub Pages)
cp -r web/* .

# Commit and push
git add .
git commit -m "Deploy web dashboard to GitHub Pages"
git push origin gh-pages
```

#### Step 3: Configure GitHub Pages

1. Go to your GitHub repository
2. Settings → Pages
3. Source: Select `gh-pages` branch
4. Save

Your site will be available at:
```
https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/
```

#### Step 4: Update Web Results Automatically

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install Poetry
        run: pip install poetry
      
      - name: Install dependencies
        run: poetry install
      
      - name: Run benchmarks
        run: poetry run python main.py
      
      - name: Prepare web data
        run: bash scripts/prepare_web_data.sh
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./web
```

---

### Option 2: Netlify (Free, Easy)

**Best For**: Quick deployment with automatic updates

#### Step 1: Prepare Project

```bash
# Create netlify.toml in project root
cat > netlify.toml << EOF
[build]
  publish = "web"
  command = "poetry run python main.py && bash scripts/prepare_web_data.sh"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
EOF
```

#### Step 2: Deploy via Netlify

1. Go to [netlify.com](https://netlify.com)
2. Sign up/login
3. Click "Add new site" → "Import an existing project"
4. Connect your GitHub repository
5. Build settings:
   - Build command: `poetry install && poetry run python main.py && bash scripts/prepare_web_data.sh`
   - Publish directory: `web`
6. Deploy!

Your site will be available at:
```
https://YOUR_SITE_NAME.netlify.app
```

#### Step 3: Add Build Script (Optional)

Create `netlify_build.sh`:

```bash
#!/bin/bash
set -e
poetry install
poetry run python main.py
bash scripts/prepare_web_data.sh
```

---

### Option 3: Vercel (Free, Fast)

**Best For**: Modern web apps, automatic deployments

#### Step 1: Install Vercel CLI

```bash
npm i -g vercel
```

#### Step 2: Configure Project

Create `vercel.json`:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "web/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/web/$1"
    }
  ]
}
```

#### Step 3: Deploy

```bash
# Generate results first
poetry run python main.py
bash scripts/prepare_web_data.sh

# Deploy
vercel --prod
```

Or connect via GitHub for automatic deployments.

---

### Option 4: Render (Free Tier Available)

**Best For**: Full-stack apps with backend needs

#### Step 1: Create render.yaml

```yaml
services:
  - type: web
    name: workflow-benchmark
    env: python
    buildCommand: poetry install && poetry run python main.py && bash scripts/prepare_web_data.sh
    staticPublishPath: web
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

#### Step 2: Deploy

1. Go to [render.com](https://render.com)
2. Create new Static Site
3. Connect GitHub repository
4. Configure build settings
5. Deploy!

---

## Docker Deployment

### Option 1: Docker Container for Web Server

Create `Dockerfile.web`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy web files
COPY web/ ./web/
COPY scripts/ ./scripts/

# Generate results (optional - can be done externally)
RUN poetry run python main.py && bash scripts/prepare_web_data.sh || true

WORKDIR /app/web

EXPOSE 8000

CMD ["python", "serve.py", "--port", "8000"]
```

#### Build and Run

```bash
# Build
docker build -f Dockerfile.web -t workflow-benchmark-web .

# Run locally
docker run -p 8000:8000 workflow-benchmark-web

# Run on network (accessible from other devices)
docker run -p 0.0.0.0:8000:8000 workflow-benchmark-web
```

#### Deploy to Docker Hub

```bash
# Tag image
docker tag workflow-benchmark-web YOUR_USERNAME/workflow-benchmark-web

# Push to Docker Hub
docker push YOUR_USERNAME/workflow-benchmark-web

# Others can run with:
docker run -p 8000:8000 YOUR_USERNAME/workflow-benchmark-web
```

---

### Option 2: Docker Compose (Full Stack)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.web
    ports:
      - "8000:8000"
    volumes:
      - ./web:/app/web
      - ./results:/app/results
    environment:
      - PYTHONUNBUFFERED=1
```

Run with:
```bash
docker-compose up
```

---

## Cloud Platform Deployments

### AWS Deployment

#### Option 1: AWS S3 + CloudFront (Static)

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Build and prepare
poetry run python main.py
bash scripts/prepare_web_data.sh

# Upload to S3
aws s3 sync web/ s3://YOUR_BUCKET_NAME --delete

# Enable static website hosting in S3 console
# Or use CloudFront for CDN
```

#### Option 2: AWS EC2 (Full Server)

```bash
# On EC2 instance:
git clone YOUR_REPO
cd workflow-optimization-benchmark
poetry install
./run_and_serve.sh

# Configure security group to allow port 8000
# Access via: http://EC2_PUBLIC_IP:8000
```

---

### Google Cloud Platform

#### Option 1: Google Cloud Storage (Static)

```bash
# Install gcloud CLI
# Configure project
gcloud config set project YOUR_PROJECT_ID

# Build and prepare
poetry run python main.py
bash scripts/prepare_web_data.sh

# Upload to GCS
gsutil -m rsync -r web/ gs://YOUR_BUCKET_NAME/

# Enable static website hosting
gsutil web set -m index.html -e 404.html gs://YOUR_BUCKET_NAME
```

#### Option 2: Google App Engine

Create `app.yaml`:

```yaml
runtime: python310

handlers:
  - url: /
    static_files: web/index.html
    upload: web/index.html

  - url: /(.*)
    static_files: web/\1
    upload: web/(.*)
```

Deploy:
```bash
gcloud app deploy
```

---

### Azure Deployment

#### Azure Static Web Apps

1. Create Azure Static Web App resource
2. Connect GitHub repository
3. Configure build:
   - App location: `/`
   - Output location: `web`
   - Build command: `poetry install && poetry run python main.py && bash scripts/prepare_web_data.sh`

---

## Sharing Pre-Generated Results

### Option 1: Share Results Folder

If you just want to share results without running benchmarks:

```bash
# 1. Generate comprehensive results
poetry run python -m src.cli run --trials 10

# 2. Prepare web data
bash scripts/prepare_web_data.sh

# 3. Create a zip file
zip -r benchmark_results.zip web/ results/

# 4. Share the zip file
# Recipients can extract and run: cd web && python serve.py
```

### Option 2: Include Results in Repository

```bash
# Commit results (if repository is public)
git add web/data/web_results.json
git add results/
git commit -m "Add benchmark results"
git push

# Others can clone and run:
git clone YOUR_REPO
cd workflow-optimization-benchmark
cd web && python serve.py
```

---

## Complete Deployment Scripts

### Automated Deployment Script

Create `deploy.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Deploying Workflow Benchmark Suite..."

# Check if results exist
if [ ! -f "web/data/web_results.json" ]; then
    echo "📊 Generating benchmark results..."
    poetry run python main.py
    bash scripts/prepare_web_data.sh
fi

# Choose deployment method
echo "Select deployment method:"
echo "1) Local network (0.0.0.0)"
echo "2) ngrok tunnel"
echo "3) GitHub Pages"
echo "4) Netlify"
read -p "Choice [1-4]: " choice

case $choice in
    1)
        echo "🌐 Starting local network server..."
        cd web
        python serve.py --port 8000
        ;;
    2)
        echo "🔗 Starting ngrok tunnel..."
        cd web
        python serve.py --port 8000 &
        sleep 2
        ngrok http 8000
        ;;
    3)
        echo "📄 Deploying to GitHub Pages..."
        git checkout -b gh-pages
        cp -r web/* .
        git add .
        git commit -m "Deploy to GitHub Pages"
        git push origin gh-pages
        echo "✅ Deployed! Check GitHub Pages settings."
        ;;
    4)
        echo "☁️  Deploying to Netlify..."
        netlify deploy --prod --dir=web
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
```

Make executable:
```bash
chmod +x deploy.sh
```

---

## Quick Reference: Deployment Options Comparison

| Method | Cost | Setup Time | Public Access | Best For |
|--------|------|------------|---------------|----------|
| **Local Network** | Free | 1 min | Same network only | Office demos |
| **ngrok** | Free (limited) | 2 min | Anyone | Quick demos |
| **GitHub Pages** | Free | 5 min | Anyone | Public demos |
| **Netlify** | Free | 5 min | Anyone | Easy deployment |
| **Vercel** | Free | 5 min | Anyone | Modern apps |
| **Docker** | Free | 10 min | Configurable | Containerized |
| **AWS/GCP/Azure** | Varies | 15+ min | Anyone | Production |

---

## Troubleshooting

### Issue: Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
python web/serve.py --port 8080
```

### Issue: Firewall Blocking

**macOS**:
```bash
# Allow Python through firewall
System Preferences → Security & Privacy → Firewall → Firewall Options
```

**Linux**:
```bash
# Allow port 8000
sudo ufw allow 8000/tcp
```

**Windows**:
- Windows Defender Firewall → Allow an app → Python

### Issue: Web Dashboard Shows "No Data"

```bash
# Regenerate web data
bash scripts/prepare_web_data.sh

# Check if file exists
ls -la web/data/web_results.json

# Verify it has content
cat web/data/web_results.json | head -20
```

### Issue: ngrok Not Working

```bash
# Check if ngrok is running
ps aux | grep ngrok

# Restart ngrok
pkill ngrok
ngrok http 8000
```

### Issue: GitHub Pages Not Updating

1. Check branch: Must be `gh-pages` or configured branch
2. Check build logs in GitHub Actions
3. Wait 1-2 minutes for propagation
4. Hard refresh browser (Ctrl+Shift+R)

---

## Security Considerations

### For Public Deployments

1. **Don't commit sensitive data**:
   - Remove any API keys or secrets
   - Use environment variables

2. **Limit CORS** (if needed):
   ```python
   # In web/serve.py, modify CORS header:
   self.send_header('Access-Control-Allow-Origin', 'https://yourdomain.com')
   ```

3. **Use HTTPS**:
   - GitHub Pages, Netlify, Vercel provide HTTPS automatically
   - For custom servers, use Let's Encrypt

4. **Rate Limiting**:
   - Consider adding rate limiting for public APIs
   - Use services like Cloudflare for DDoS protection

---

## Recommendations by Use Case

### 🎯 **Quick Demo for Colleagues**
→ Use **ngrok** or **local network sharing**

### 🎯 **Public Demo/Portfolio**
→ Use **GitHub Pages** or **Netlify**

### 🎯 **Production Deployment**
→ Use **AWS S3 + CloudFront**, **Google Cloud**, or **Azure**

### 🎯 **Containerized Deployment**
→ Use **Docker** + **Docker Hub** or **Kubernetes**

### 🎯 **Continuous Updates**
→ Use **GitHub Actions** + **GitHub Pages** or **Netlify** with GitHub integration

---

## Next Steps

1. **Choose your deployment method** based on your needs
2. **Test locally** first: `./run_and_serve.sh`
3. **Generate results**: Ensure `web/data/web_results.json` exists
4. **Deploy** using one of the methods above
5. **Share the URL** with your audience
6. **Monitor** usage and update results as needed

---

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review `README.md` for setup instructions
3. Check logs in the terminal/console
4. Verify all dependencies are installed: `poetry install`

---

**Last Updated**: October 2025  
**For questions or issues**: Check the main README.md or project documentation

