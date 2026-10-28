# Deployment Architecture & Production Guide

## 1. Hosting Architecture Overview

Deep Learning inference using TensorFlow involves substantial compiled C++ binaries (`tensorflow-intel` / CUDA) and memory footprints exceeding 300MB–1GB. 

### Why Serverless (e.g. AWS Lambda / Vercel Serverless Functions) Has Constraints
- **Vercel Serverless Function Limit:** Vercel functions have an unzipped package size limit of 50MB (or 250MB for enterprise) and a default memory ceiling of 1024MB. TensorFlow and its dependencies exceed these unzipped limits.
- **Cold Starts:** Loading a 300MB TensorFlow wheel and reconstructing weights on every cold invocation can introduce 8–15 second latencies.

### Recommended Dual-Tier Deployment Architecture
1. **Frontend / API Gateway (Vercel):**
   - Deploys static HTML5/CSS/JS frontend assets and lightweight serverless proxy routing.
   - Handles SSL, CDN distribution, domain mapping, and client asset caching.
2. **Inference Backend Service (Containerized / Dedicated Host):**
   - Deploys the Flask deep learning inference engine (`app.py`) inside a Docker container.
   - Recommended free/academic hosting targets:
     - **Render.com / Railway.app:** Native Docker support with 512MB–2GB RAM.
     - **Hugging Face Spaces (Docker / Gradio / Flask):** Free CPU/T4 GPU allocation tailored for ML models.
     - **AWS EC2 / DigitalOcean Droplet:** Persistent virtual machines.

---

## 2. Docker Deployment (Backend Container)

A production-grade `Dockerfile` is provided in the repository root:

```bash
# 1. Build the Docker container
docker build -t mobile-classifier:latest .

# 2. Run the container on port 5000
docker run -p 5000:5000 --name mobile-ai mobile-classifier:latest
```

The container starts Gunicorn/Flask with an in-memory loaded model, providing &lt; 50ms inference times.

---

## 3. Vercel Deployment Guide

To deploy the frontend to Vercel:

1. Configure `vercel.json` in the root repository (provided in project).
2. Set Environment Variable in Vercel Project Settings:
   - `INFERENCE_API_URL`: Points to your deployed containerized inference URL (e.g., `https://mobile-classifier.onrender.com`).
3. Deploy via Vercel CLI:
   ```bash
   npm install -g vercel
   vercel --prod
   ```

---

## 4. Health Check & Diagnostic Verification

Verify that the deployed service is active:

```bash
# Health Check Endpoint
curl -X GET https://your-deployment-url/api/health

# Expected JSON response:
# {
#   "status": "healthy",
#   "service": "Mobile Phone Model Classification API",
#   "engine_ready": true,
#   "classes_count": 5,
#   "version": "1.0.0"
# }
```

