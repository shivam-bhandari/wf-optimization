#!/bin/bash
# Quick deployment script for sharing the workflow benchmark suite

set -e

echo "=========================================="
echo "🚀 Workflow Benchmark Deployment Helper"
echo "=========================================="
echo ""

# Check if results exist
if [ ! -f "web/data/web_results.json" ]; then
    echo "⚠️  No web results found. Generating now..."
    echo ""
    
    if command -v poetry &> /dev/null; then
        poetry run python main.py
    else
        python main.py
    fi
    
    bash scripts/prepare_web_data.sh
    echo ""
fi

echo "Select deployment method:"
echo ""
echo "1) 🌐 Local Network (same WiFi/LAN)"
echo "   → Share with people on your network"
echo ""
echo "2) 🔗 ngrok Tunnel (anyone, anywhere)"
echo "   → Share with anyone via ngrok URL"
echo ""
echo "3) 📄 GitHub Pages (public hosting)"
echo "   → Free, permanent URL via GitHub"
echo ""
echo "4) ☁️  Netlify (quick cloud deployment)"
echo "   → Fast, free cloud hosting"
echo ""
echo "5) 📦 Docker Container"
echo "   → Containerized deployment"
echo ""
echo "6) 📋 Just show me the commands"
echo "   → Print deployment commands"
echo ""

read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "🌐 Starting local network server..."
        echo ""
        
        # Get local IP
        if [[ "$OSTYPE" == "darwin"* ]]; then
            IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
        else
            IP=$(hostname -I | awk '{print $1}')
        fi
        
        if [ -z "$IP" ]; then
            IP="localhost"
        fi
        
        echo "📍 Your local IP: $IP"
        echo "🔗 Share this URL: http://$IP:8000"
        echo ""
        echo "Press Ctrl+C to stop the server"
        echo ""
        
        cd web
        python serve.py --port 8000
        ;;
        
    2)
        echo ""
        echo "🔗 Starting ngrok tunnel..."
        echo ""
        
        if ! command -v ngrok &> /dev/null; then
            echo "❌ ngrok not found!"
            echo ""
            echo "Install ngrok:"
            echo "  macOS: brew install ngrok"
            echo "  Or download from: https://ngrok.com/download"
            exit 1
        fi
        
        # Start server in background
        cd web
        python serve.py --port 8000 --no-open &
        SERVER_PID=$!
        cd ..
        
        sleep 2
        
        echo "✅ Server started"
        echo "🔗 Starting ngrok tunnel..."
        echo ""
        echo "📋 Share the ngrok URL shown below:"
        echo ""
        
        # Run ngrok
        ngrok http 8000
        
        # Cleanup
        kill $SERVER_PID 2>/dev/null || true
        ;;
        
    3)
        echo ""
        echo "📄 Deploying to GitHub Pages..."
        echo ""
        
        # Check if git repo
        if ! git rev-parse --git-dir > /dev/null 2>&1; then
            echo "❌ Not a git repository!"
            echo "Initialize git first: git init"
            exit 1
        fi
        
        # Check if gh-pages branch exists
        if git show-ref --verify --quiet refs/heads/gh-pages; then
            echo "⚠️  gh-pages branch already exists"
            read -p "Overwrite? [y/N]: " overwrite
            if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
                echo "Cancelled."
                exit 0
            fi
            git branch -D gh-pages
        fi
        
        # Create gh-pages branch
        git checkout -b gh-pages
        
        # Copy web files to root
        cp -r web/* .
        
        # Commit
        git add .
        git commit -m "Deploy to GitHub Pages" || echo "No changes to commit"
        
        echo ""
        echo "✅ Files ready for GitHub Pages"
        echo ""
        echo "Next steps:"
        echo "1. Push to GitHub: git push origin gh-pages"
        echo "2. Go to GitHub repo → Settings → Pages"
        echo "3. Select 'gh-pages' branch as source"
        echo "4. Your site will be at: https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/"
        echo ""
        
        read -p "Push to GitHub now? [y/N]: " push
        if [[ "$push" =~ ^[Yy]$ ]]; then
            git push origin gh-pages
            echo ""
            echo "✅ Pushed! Check GitHub Pages settings."
        else
            echo "Files are ready. Push manually when ready."
        fi
        
        # Return to main branch
        git checkout main 2>/dev/null || git checkout master 2>/dev/null || echo "Note: Switch back to your main branch manually"
        ;;
        
    4)
        echo ""
        echo "☁️  Deploying to Netlify..."
        echo ""
        
        if ! command -v netlify &> /dev/null; then
            echo "❌ Netlify CLI not found!"
            echo ""
            echo "Install Netlify CLI:"
            echo "  npm install -g netlify-cli"
            echo ""
            echo "Or deploy manually at: https://app.netlify.com"
            exit 1
        fi
        
        echo "Building..."
        if command -v poetry &> /dev/null; then
            poetry run python main.py
        else
            python main.py
        fi
        bash scripts/prepare_web_data.sh
        
        echo ""
        echo "Deploying to Netlify..."
        netlify deploy --prod --dir=web
        
        echo ""
        echo "✅ Deployed! Check your Netlify dashboard."
        ;;
        
    5)
        echo ""
        echo "📦 Docker Container Deployment..."
        echo ""
        
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker not found!"
            echo "Install Docker from: https://docker.com"
            exit 1
        fi
        
        echo "Building Docker image..."
        docker build -f Dockerfile.web -t workflow-benchmark-web . 2>/dev/null || {
            echo ""
            echo "⚠️  Dockerfile.web not found. Creating one..."
            cat > Dockerfile.web << 'EOF'
FROM python:3.10-slim
WORKDIR /app
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-dev
COPY web/ ./web/
COPY scripts/ ./scripts/
WORKDIR /app/web
EXPOSE 8000
CMD ["python", "serve.py", "--port", "8000"]
EOF
            echo "✅ Created Dockerfile.web"
            echo "Building..."
            docker build -f Dockerfile.web -t workflow-benchmark-web .
        }
        
        echo ""
        echo "✅ Image built!"
        echo ""
        echo "Run with:"
        echo "  docker run -p 8000:8000 workflow-benchmark-web"
        echo ""
        read -p "Start container now? [y/N]: " start
        if [[ "$start" =~ ^[Yy]$ ]]; then
            docker run -p 8000:8000 workflow-benchmark-web
        fi
        ;;
        
    6)
        echo ""
        echo "📋 Deployment Commands Reference:"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "1. LOCAL NETWORK:"
        echo "   cd web && python serve.py --port 8000"
        echo "   Share: http://YOUR_IP:8000"
        echo ""
        echo "2. NGROK:"
        echo "   cd web && python serve.py --port 8000 &"
        echo "   ngrok http 8000"
        echo "   Share the ngrok URL"
        echo ""
        echo "3. GITHUB PAGES:"
        echo "   git checkout -b gh-pages"
        echo "   cp -r web/* ."
        echo "   git add . && git commit -m 'Deploy'"
        echo "   git push origin gh-pages"
        echo ""
        echo "4. NETLIFY:"
        echo "   netlify deploy --prod --dir=web"
        echo ""
        echo "5. DOCKER:"
        echo "   docker build -f Dockerfile.web -t workflow-benchmark-web ."
        echo "   docker run -p 8000:8000 workflow-benchmark-web"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "For detailed instructions, see: DEPLOYMENT_GUIDE.md"
        ;;
        
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

