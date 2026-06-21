# Deployment Guide — AWS (backend + inference) + Vercel (frontend)

This is the step-by-step runbook for getting Gridlock 2.0 live: the API + inference container on
an **AWS EC2 GPU box**, and the Next.js dashboard on **Vercel**.

You'll need accounts on both AWS and Vercel. **Never paste AWS secret keys or Vercel tokens into
chat** — run the auth steps below in your own terminal (`! <command>` runs it in this session if
you want me to see the output, but for login/credential prompts, run them in a normal terminal
window instead so the secret never appears in this conversation).

---

## Part A — AWS (backend + inference)

### A0. One-time AWS CLI setup (you do this, not me)

```bash
aws configure
```
This prompts for **Access Key ID**, **Secret Access Key**, default region (pick one close to you,
e.g. `ap-south-1` for India or `us-east-1`), and output format (`json`). Run this in your own
terminal — once configured, I can run `aws ec2 ...` commands on your behalf using your local
profile without ever seeing the keys.

**Don't have an IAM user yet?** AWS Console → IAM → Users → Create user → attach
`AmazonEC2FullAccess` (or narrower, EC2-only permissions) → Security credentials tab → Create
access key → "Command Line Interface (CLI)".

### A1. Launch the EC2 instance

Recommended: AWS's **Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 24.04)** — ships the NVIDIA
driver + Docker + `nvidia-container-toolkit` pre-installed, so you skip manual driver setup.

**Via Console (simplest for a first deploy):**
1. EC2 → Launch instance
2. AMI: search "Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 24.04)" → select the AWS Marketplace one (free, you only pay EC2 compute)
3. Instance type: **g4dn.xlarge** (1× T4 GPU, 16GB VRAM, 4 vCPU, 16GB RAM) — ~$0.526/hr on-demand (us-east-1; check your region)
4. Key pair: create new (download the `.pem`, keep it safe — needed for SSH)
5. Network settings → Edit → Security group rules:
   - SSH (22) — source: **My IP** (not 0.0.0.0/0)
   - Custom TCP (8000) — source: **Anywhere (0.0.0.0/0)** (the API; tighten later behind HTTPS/a domain if you want)
6. Storage: bump to **60 GB** (the CUDA base image + two Python environments need real room — 8GB default will fill up)
7. Launch

**Via CLI** (if you'd rather I drive it — tell me to go ahead and I'll run these with your configured profile):
```bash
aws ec2 run-instances \
  --image-id <ami-id-from-AWS-Marketplace-DLAMI> \
  --instance-type g4dn.xlarge \
  --key-name <your-key-pair-name> \
  --security-group-ids <sg-id> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":60}}]' \
  --count 1
```

### A2. Allocate + associate an Elastic IP

Keeps your public IP stable across stop/start (free while the instance is running; small hourly
charge if you allocate one but leave it unattached — always associate it).
```bash
aws ec2 allocate-address --domain vpc
aws ec2 associate-address --instance-id <instance-id> --allocation-id <allocation-id>
```
Or: EC2 Console → Elastic IPs → Allocate → Associate with your instance.

### A3. SSH in and sanity-check the GPU stack

```bash
ssh -i your-key.pem ubuntu@<elastic-ip>
nvidia-smi                          # confirms the driver
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi   # confirms nvidia-container-toolkit
```
Both should print GPU info. If the second one fails, the DLAMI's toolkit needs a restart:
`sudo systemctl restart docker`.

### A4. Get the code onto the instance

```bash
git clone <your-repo-url>.git gridlock
cd gridlock
```
(If the repo is private, set up a deploy key or use `gh auth login` / a PAT.)

### A5. Upload your `.env` (secrets — never commit, never put in the repo)

From your **local machine**:
```bash
scp -i your-key.pem .env ubuntu@<elastic-ip>:~/gridlock/.env
```
Then on the instance, edit it for production:
```bash
nano .env
```
Set:
```
INFERENCE_MODE=real
ALLOWED_ORIGINS=https://<your-vercel-domain>.vercel.app,http://localhost:3000
DETECTION_VARIANT=large
PIPELINE_USE_SAM3=true
PIPELINE_USE_VLM=true
```
(Add the Vercel domain once Part B gives you one — you can come back and edit this after.)

### A6. Build and run the container

```bash
cd ~/gridlock
docker build -t gridlock-api .          # first build: ~15-25 min (CUDA base + 2x torch+cuda envs)
docker run -d --gpus all \
  --env-file .env \
  -p 8000:8000 \
  --restart unless-stopped \
  --name gridlock-api \
  gridlock-api
```

Check it's alive:
```bash
curl http://localhost:8000/api/health
# {"status":"ok","inference_mode":"real","supabase_configured":true}
docker logs -f gridlock-api             # watch model loading / requests
```

From your own machine: `curl http://<elastic-ip>:8000/api/health` — if that works, the public API
link is `http://<elastic-ip>:8000`.

### A7. (Recommended before sharing widely) Put HTTPS in front of it

Browsers will block a Vercel (`https://`) frontend from calling a plain `http://` backend in some
contexts, and a bare IP:port looks unfinished. If you have a domain, point an `A` record at the
Elastic IP, then run **Caddy** as a reverse proxy (automatic free HTTPS via Let's Encrypt):

```bash
sudo docker run -d --name caddy --restart unless-stopped \
  -p 80:80 -p 443:443 \
  -v caddy_data:/data \
  caddy:2 caddy reverse-proxy --from your-domain.com --to localhost:8000
```
Now the backend is `https://your-domain.com` — use that as `NEXT_PUBLIC_BACKEND_URL` in Vercel,
and update `ALLOWED_ORIGINS` accordingly.

### A8. Cost control

- **Stop the instance** (`aws ec2 stop-instances --instance-ids <id>` or Console) when not
  actively demoing — you stop paying for compute (EBS storage still bills, ~$5/month for 60GB).
  The Elastic IP stays associated.
- **Start it again** (`aws ec2 start-instances ...`) takes ~1 minute; the container auto-starts
  (`--restart unless-stopped`) once Docker comes up.
- g4dn.xlarge on-demand: ~$0.526/hr (~$380/mo if left running 24/7) — stop-when-idle is the easy
  lever if you're watching cost.

---

## Part B — Vercel (frontend)

### B0. Push the repo to GitHub (Vercel deploys best from a git repo)

If not already on GitHub, push it there now — Vercel's GitHub integration gives you automatic
deploys on every push, which you'll want.

### B1. Import the project

**Via Vercel dashboard (recommended for the first deploy):**
1. https://vercel.com/new → Import your GitHub repo
2. **Root Directory**: set to `frontend` (the Next.js app lives there, not the repo root)
3. Framework Preset: Next.js (auto-detected)
4. Environment Variables — add:
   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_SUPABASE_URL` | `https://orhsqnowvfbgnuosraxs.supabase.co` |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | (the anon key — safe to expose, it's public-by-design) |
   | `NEXT_PUBLIC_BACKEND_URL` | `http://<elastic-ip>:8000` (or your Caddy HTTPS domain from A7) |
5. Deploy

**Via CLI** (if you'd rather I drive it from here — run `vercel login` yourself first since it
opens a browser for auth, then tell me to continue):
```bash
cd frontend
npx vercel login        # YOU run this — opens browser, do not share the resulting token
npx vercel link         # links this folder to a Vercel project
npx vercel env add NEXT_PUBLIC_SUPABASE_URL production
npx vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY production
npx vercel env add NEXT_PUBLIC_BACKEND_URL production
npx vercel --prod       # deploys; prints the public URL
```

### B2. Close the loop — update the backend's CORS allow-list

Once Vercel gives you a URL (e.g. `https://gridlock-2-0.vercel.app`), SSH back into the EC2 box:
```bash
nano .env   # update ALLOWED_ORIGINS to include the vercel URL
docker restart gridlock-api
```

### B3. Smoke test the full deployed stack

Open the Vercel URL in a browser:
- Dashboard loads, shows the deployed-models strip (reads Supabase directly — works even if the
  backend is down)
- Upload a photo → should hit the EC2 backend → real pipeline runs → violation appears in the
  realtime feed within a second or two

---

## Quick reference

| Component | Where | URL pattern |
|---|---|---|
| Frontend | Vercel | `https://<project>.vercel.app` |
| Backend + inference | AWS EC2 (g4dn.xlarge) | `http://<elastic-ip>:8000` or `https://<your-domain>` |
| Database + storage | Supabase | `https://orhsqnowvfbgnuosraxs.supabase.co` |

## Updating after a code change

```bash
# on the EC2 box
cd ~/gridlock && git pull
docker build -t gridlock-api .
docker stop gridlock-api && docker rm gridlock-api
docker run -d --gpus all --env-file .env -p 8000:8000 --restart unless-stopped --name gridlock-api gridlock-api
```
Vercel redeploys automatically on every `git push` to the connected branch — no manual step needed.
