# JüNA Chatbot

<div align="center">
  <img src="app/assets/logo.png" alt="JüNA logo" width="180"/>
</div>

JüNA helps you explore neutron science, understand scientific software, and
create Python plots through chat.

Follow these steps to run JüNA on your own computer. The setup below uses
**macOS with Docker Desktop** and a **Blablador API key**. It enables chat,
Python plots, and repository search, starting with jscatter.

On Windows, start with the [WSL 2 instructions](#windows-wsl-2).

## 1. Install the tools

Install and open **Docker Desktop**. Wait until it says Docker is running.

With **Homebrew** installed, run this in Terminal:

```bash
brew install git uv podman
podman machine init --memory 8192
```

Run `podman machine init` only once. If you already have a Podman machine,
skip that command.

## 2. Download the project

Run these commands in the folder where you keep your projects:

```bash
git clone --branch juena-core-cutover-runtime https://github.com/neutron-simlab/juena-chatbot.git
git clone https://github.com/neutron-simlab/juena-core.git
git clone https://github.com/neutron-simlab/juena-rag.git
cd juena-chatbot
```

Keep the three folders beside each other. If you already have them, skip the
clone commands and open Terminal in `juena-chatbot`.

## 3. Add your settings

Create your local settings file:

```bash
./juena env
```

Open `.env` in your editor and put your Blablador API key after
`BLABLADOR_API_KEY=`:

```dotenv
BLABLADOR_API_KEY=your-blablador-api-key
```

Leave the other settings as supplied. They already set up Blablador, the
sandbox, Langfuse tracing (its UI starts with `./juena langfuse-up`; chats work
without it) and `AUTH_DEV_BYPASS=true`, which lets you sign in locally without
setting up FZJ login. Use that only on your own development computer.

## 4. Prepare repository search once

Open the search folder and copy its settings template. If `.env` already
exists, skip only the `cp` command:

```bash
cd ../juena-rag
cp .env.example .env
```

Open **this folder's `.env`** and fill in `BLABLADOR_API_KEY` with the same key.
Keep its other settings unchanged.

Install the search tools and start its database:

```bash
uv sync
docker compose up -d --wait pgvector
```

On your first setup only, copy and run this block to prepare the database.
Skip it if you already prepared it:

```bash
docker compose exec -T pgvector createdb -U postgres juena_rag_corpus
uv run python - <<'PYTHON'
import os
import psycopg
from dotenv import load_dotenv
from juena_rag.store import apply_ddl

load_dotenv('.env')
with psycopg.connect(os.environ['JUENA_RAG_DSN']) as connection:
    apply_ddl(connection, dimension=4096, create_extension=True)
PYTHON
```

Add jscatter to the search library, then return to the chatbot folder:

```bash
uv run --group use-cases python use-cases/code-repository/ingest.py jscatter
cd ../juena-chatbot
```

Wait for this command to finish. Downloading repositories alone does not make
them searchable. To add more software later, see [Adding repositories](HOW-TO-ADD-REPO.md).

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
| Build cannot find `juena-core` | Put `juena-core` beside `juena-chatbot`, as in step 2. |
| Startup fails or the page does not open | Run `./juena logs` and check the error. |
| The page asks for FZJ login | Set `AUTH_DEV_BYPASS=true` in the chatbot's `.env`, then run `./juena up` again. |
| Repository search is unavailable or empty | Finish step 4, then run `./juena rag-up`. Check `./juena rag-logs` if it fails. |
| The model reports an API key error | Check the key in both `.env` files, then run `./juena down` followed by `./juena up`. |

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

   If the `systemctl` command fails, follow
   [Microsoft's systemd setup](https://learn.microsoft.com/en-us/windows/wsl/systemd)
   and reopen Ubuntu. If Docker is unavailable, check the Ubuntu integration
   setting from the previous step.

4. Create a project folder inside Ubuntu:

   ```bash
   mkdir -p ~/projects
   cd ~/projects
   ```

   Follow **steps 2–4 above** to download the three repositories and prepare
   the settings and search database. Keep the repositories under `~/projects`
   so Docker and Podman can share the same files.

5. From `~/projects/juena-chatbot`, start JüNA with Ubuntu's current address
   for repository search:

   ```bash
   JUENA_RAG_URL="http://$(hostname -I | awk '{print $1}'):8100" \
     JUENA_RAG_BIND=0.0.0.0 ./juena up
   ./juena ps
   ./juena health
   ```

   Use this startup command each time; WSL's address can change after a restart.
   Open **<http://localhost:9501>** in your Windows browser and click
   **Sign in (development)**. Stop the app from Ubuntu with `./juena down`.

## Other setups

- **Linux:** install Docker with Compose, Git, uv, and rootless Podman. Enable the
  Podman socket with `systemctl --user enable --now podman.socket`. Follow steps
  2–6, using `JUENA_RAG_BIND=0.0.0.0 ./juena up` in step 5 so the app can reach
  search. Keep the search port accessible only to your local containers.
- **OpenAI for chat:** use the OpenAI settings in [env.example](env.example).
  Repository search still needs the Blablador key in `juena-rag/.env`.
- **FZJ login or a shared server:** use the [login setup guide](SAML-IFFLOGIN.md)
  and [sandbox deployment guide](deploy/SANDBOX.md).
- **Tracing every chat:** run `./juena langfuse-up` for a Langfuse server on
  <http://localhost:3000>; see the [Langfuse guide](deploy/LANGFUSE.md).
- **More settings:** see [env.example](env.example).
- **Development checks:** run `./juena test`. Run `./juena test-integration`
  for the database checks.

## License

MIT
