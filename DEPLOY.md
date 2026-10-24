# Deploying EpochZero

Two parts, two free hosts:

| Part | Host | What runs there |
|---|---|---|
| Backend (`backend/`) | Render, free web service | FastAPI, the agent, the PyTorch sandbox; calls Groq |
| Frontend (`frontend/`) | Vercel, free hobby plan | The Next.js app in the browser; calls the backend |

The Groq key only ever lives on Render. The frontend only knows the backend's URL.

Order matters: deploy the backend first (the frontend needs its URL), then the frontend, then tell the backend the
frontend's URL (CORS).

---

## 0. Push the code to GitHub

Both hosts deploy from the GitHub repository (`aaishii07/EpochZero`). From the project folder:

```powershell
git add -A
git status            # check that no .env file is listed
git commit -m "Prepare deployment: Render blueprint, Vercel config, health check, rate limit"
git push
```

`backend/.env` and `frontend/.env*` are git-ignored, so the Groq key is not pushed. If `git status` ever lists a
`.env` file, stop and remove it from the commit.

---

## 1. Backend on Render

1. Sign in at render.com with GitHub.
2. New, then Blueprint. Pick the `EpochZero` repository. Render finds `render.yaml` at the root and shows one
   service, `epochzero-api`, on the free plan.
3. It asks for the two values marked secret:
   - `GROQ_API_KEY`: your Groq key.
   - `EPOCHZERO_CORS_ORIGINS`: leave empty for now; it is filled in step 3.
4. Apply. The first build takes several minutes (it installs PyTorch, about 200 MB).
5. When it is live, open `https://<your-service>.onrender.com/health`. Expect:
   `{"status":"ok","llm":"live", ... ,"sandbox":"ready"}`.
   - `"llm":"mock"` means the Groq key is missing: set it under Environment and redeploy.
   - `"sandbox":"starting"` means PyTorch is still loading; wait a minute and reload.

Copy the service URL, for example `https://epochzero-api.onrender.com`.

What `render.yaml` sets, and why:

| Variable | Value | Why |
|---|---|---|
| `PYTHON_VERSION` | 3.12.7 | The version PyTorch wheels and the tests are built for |
| `GROQ_API_KEY` | secret | The LLM provider key |
| `EPOCHZERO_CORS_ORIGINS` | your Vercel URL | Only your frontend may call the API from a browser |
| `EPOCHZERO_RATE_LIMIT` | 20 | Checks per visitor per 10 minutes, so strangers cannot drain your Groq quota |
| `EPOCHZERO_DRY_RUN_MAX_PARAMS` | 5000000 | The free plan has 512 MB of memory; bigger models still get the shape probe |
| `EPOCHZERO_SANDBOX_TIMEOUT` | 45 | The free plan's CPU is slow; checks need more time |
| `EPOCHZERO_SANDBOX_STARTUP` | 240 | Loading PyTorch on that CPU can take minutes |

---

## 2. Frontend on Vercel

1. Sign in at vercel.com with GitHub.
2. Add New, then Project. Import the `EpochZero` repository.
3. Root Directory: `frontend` (important; the repository root is not the Next.js app). Framework: Next.js
   (detected).
4. Environment Variables: `NEXT_PUBLIC_API_URL` = the Render URL from step 1, without a trailing slash.
5. Deploy. Copy the production URL, for example `https://epochzero.vercel.app`.

`NEXT_PUBLIC_API_URL` is baked in at build time. If you change it later, redeploy the frontend.

---

## 3. Connect them (CORS)

Back on Render: the service, Environment, set `EPOCHZERO_CORS_ORIGINS` to the Vercel URL, for example
`https://epochzero.vercel.app`. Save; Render redeploys.

To also allow a custom domain or a preview URL, separate them with commas.

---

## 4. Check it end to end

1. Open the Vercel URL. A banner says the server is waking or loading PyTorch; wait until it disappears.
2. Load example, Shape mismatch. The first check starts by itself.
3. Agent trace shows the agent's steps; Findings shows the Agent report and a verified fix.
4. Apply fix, then Check: the History tab shows the change.

If something fails:

| Symptom | Cause | Fix |
|---|---|---|
| Banner says the server is not reachable | Render service down, or wrong `NEXT_PUBLIC_API_URL` | Open `/health` on Render; fix the URL on Vercel and redeploy |
| Browser console shows a CORS error | `EPOCHZERO_CORS_ORIGINS` missing or different from the Vercel URL | Set it exactly (https, no trailing slash) |
| Report says "Built-in summary (offline mode)" | Groq key missing on Render | Set `GROQ_API_KEY`, redeploy |
| Report says the agent model was unavailable | Groq rate limit or daily quota reached | Wait, or use another key (see below) |
| "Too many checks from this address" | The per-visitor limit | Wait 10 minutes, or raise `EPOCHZERO_RATE_LIMIT` |
| Render logs show the service restarted while checking | Out of memory on the free plan | Lower `EPOCHZERO_DRY_RUN_MAX_PARAMS`, or move to a bigger plan or Hugging Face Spaces |

---

## 5. Free plan limits to know before a demo

- Render free services sleep after about 15 minutes without traffic. The next visit wakes it, which takes about a
  minute plus PyTorch loading. Open the site a few minutes before presenting, and run one example.
- Groq's free tier has per-minute and per-day token limits. Running many checks just before a demo can use up the
  day's quota; the app then falls back to its built-in plan and says so. Keep a second key from a teammate's own
  Groq account ready; changing `GROQ_API_KEY` on Render and redeploying switches to it.
- Logs and history are stored in each visitor's browser (localStorage). They are not shared between computers.

## 6. Security notes

- The backend runs parts of the submitted script (class definitions) inside its sandbox process to build the model.
  The sandbox blocks weight downloads and file loading during that step and has time limits, but it is not a
  security boundary. For the hackathon this is acceptable; do not deploy it with secrets beyond the Groq key.
- Keep the rate limit on while the URL is public.
- Never put the Groq key in the frontend or in `NEXT_PUBLIC_` variables: anything there is visible to every visitor.

## Updating after deployment

Push to GitHub. Render and Vercel both redeploy the new commit automatically.
