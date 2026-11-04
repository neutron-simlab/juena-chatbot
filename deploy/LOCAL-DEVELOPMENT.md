# Local development

Use this setup when you need to develop or test JüNA on your own computer.
Normal use is through the deployed server; see [SERVER.md](SERVER.md) for
installation and operation. Local settings use `.env` and the development
authentication bypass, and must not be applied to the shared server.

Follow these steps to run JüNA on your own computer. The setup below uses
**macOS with Docker Desktop** and a **Blablador API key**. It enables chat,
Python plots, and repository search, starting with jscatter.

On Windows, start with the [WSL 2 instructions](#windows-wsl-2).

## 1. Install the tools

Install and open **Docker Desktop**. Wait until it says Docker is running.

With **Homebrew** installed, run this in Terminal:

```bash
brew install git uv node@24 podman
export PATH="$(brew --prefix node@24)/bin:$PATH"
podman machine init --memory 8192
```

Run `podman machine init` only once. If you already have a Podman machine,
skip that command.

## 2. Download the project

Run these commands in the folder where you keep your projects:

```bash
git clone https://github.com/neutron-simlab/juena-chatbot.git
cd juena-chatbot
```

Only the chatbot checkout is required. `pyproject.toml` and `uv.lock` pin
published core and RAG commits. Docker builds their code and core's Files panel
automatically; native `uv sync` also requires Node.js 24 and npm on PATH.

## 3. Add your settings

Create your local settings file:

```bash
./juena env
```

Set `RAG_BLABLADOR_API_KEY` in `.env` to the service key for indexing and search
embeddings. Generate `JUENA_SECRET_KEY` once using the command in `env.example`
and keep it stable across restarts. Sign in to JüNA, open **My profile**, and add
your personal Blablador key before chatting. Tavily and Context7 keys are optional.
See [personal API keys](PERSONAL-API-KEYS.md) for setup and upgrade instructions.

For an existing local `.env` from before personal API keys, run
`./juena profile-setup` once. It moves the old shared Blablador key to the RAG
setting, creates the encryption key if missing, and keeps a protected backup.
It preserves an existing encryption key and dedicated RAG key.

Leave the other settings as supplied. They already set up Blablador, the
sandbox, disabled tracing and `AUTH_DEV_BYPASS=true`, which lets you sign in locally without
setting up FZJ login. Use that only on your own development computer.

## 4. Prepare repository search once

From the chatbot folder, run:

```bash
./juena repos-ingest jscatter
```

This builds the shared image, starts the bundled pgvector database, initializes
an empty schema, and runs ingestion in a container. Repeating initialization
preserves the existing corpus; incompatible schemas fail without changing data.

Wait for this command to finish. Downloading repositories alone does not make
them searchable. To add more software later, see [Adding repositories](../HOW-TO-ADD-REPO.md).

## 5. Start JüNA

From `juena-chatbot`, run:

```bash
./juena up
```

The first start downloads software and builds the app, so it can take a while.
Wait until the command prints the UI address. Then check:

```bash
./juena ps
./juena health
```

The app and database should be running, with `sandbox-worker` and `juena-rag`
also reported as running. The health command should return a successful response.

## 6. Open the chat

Open **<http://localhost:9501>** and click **Sign in (development)**.

Try asking: **How do I fit a sphere model to scattering data with jscatter?**

To try code execution, select **Plot with Python**. Review and approve the
proposed command when JüNA asks, then wait for the plot.

## Everyday commands

Run these from `juena-chatbot`:

| What you want to do | Command |
| --- | --- |
| Start JüNA | `./juena up` |
| Stop JüNA and keep your chats and search data | `./juena down` |
| Restart after changing settings | `./juena down`, then `./juena up` |
| Check what is running | `./juena ps` |
| Watch startup or app errors | `./juena logs` |
| Watch search errors | `./juena rag-logs` |

Press **Ctrl+C** to stop watching logs; JüNA keeps running.

## If something goes wrong

| Problem | What to check |
| --- | --- |
| Docker cannot connect | Open Docker Desktop and wait for it to finish starting. |
| Git dependency build fails | Check GitHub/npm access; native source installs need Node.js 24 and npm on PATH. |
| Startup fails or the page does not open | Run `./juena logs` and check the error. |
| The page asks for FZJ login | Set `AUTH_DEV_BYPASS=true` in the chatbot's `.env`, then run `./juena up` again. |
| Repository search is unavailable or empty | Finish step 4, then run `./juena rag-up`. Check `./juena rag-logs` if it fails. |
| The model reports an API key error | Check the key in the chatbot `.env`, then run `./juena down` followed by `./juena up`. |

## Windows (WSL 2)

1. Open **PowerShell as Administrator** and install Ubuntu:

   ```powershell
   wsl --install -d Ubuntu
   ```

   Restart Windows if prompted, then open **Ubuntu** and create your Linux
   username and password. If Ubuntu is already installed, run `wsl --update`
   and check that `wsl --list --verbose` shows version **2**.
   See [Microsoft's WSL setup](https://learn.microsoft.com/en-us/windows/wsl/install).

2. Install and open **Docker Desktop for Windows**. Enable **Use WSL 2 based
   engine**, then enable **Ubuntu** under **Settings → Resources → WSL
   Integration** and apply the changes.
   See [Docker's WSL setup](https://docs.docker.com/desktop/features/wsl/).

3. Run the remaining commands in the **Ubuntu terminal**. Install the tools:

   ```bash
   sudo apt update
   sudo apt install -y git curl podman uidmap slirp4netns fuse-overlayfs
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"
   systemctl --user enable --now podman.socket
   docker compose version
   podman info
   ```

   Native source installs also require Node.js 24 and npm on PATH. The production
   sandbox setup reuses the application image and does not need them on the host.
   If the `systemctl` command fails, follow
   [Microsoft's systemd setup](https://learn.microsoft.com/en-us/windows/wsl/systemd)
   and reopen Ubuntu. If Docker is unavailable, check the Ubuntu integration
   setting from the previous step.

4. Create a project folder inside Ubuntu:

   ```bash
   mkdir -p ~/projects
   cd ~/projects
   ```

   Follow **steps 2–4 above** to download the chatbot repository and prepare
   the settings and search database. Keep the checkout under `~/projects`
   so Docker and Podman can share the same files.

5. From `~/projects/juena-chatbot`, run `./juena up`, then `./juena ps`
   and `./juena health`. Open **<http://localhost:9501>** in your Windows
   browser and click **Sign in (development)**. Stop with `./juena down`.

## Other setups

- **Linux:** install Docker with Compose, Git, uv, and rootless Podman. Enable the
  Podman socket with `systemctl --user enable --now podman.socket`. Follow steps
  2–6. Native installs need Node.js 24 and npm on PATH; the production worker
  reuses the application's built dependencies.
- **OpenAI for chat:** use the OpenAI settings in [env.example](../env.example).
  Repository search still needs the Blablador key in the chatbot `.env`.
- **FZJ login or a shared server:** use the [login setup guide](SAML-IFFLOGIN.md)
  and [sandbox deployment guide](SANDBOX.md).
- **Tracing every chat:** set `LANGFUSE_TRACING=true`, then run `./juena langfuse-up` for a Langfuse server on
  <http://localhost:3000>; see the [Langfuse guide](LANGFUSE.md).
- **More settings:** see [env.example](../env.example).
- **Development checks:** run `./juena test`. Run `./juena test-integration`
  for the database checks.
