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

Open `.env` in your editor and put your Blablador API key after
`BLABLADOR_API_KEY=`:

```dotenv
BLABLADOR_API_KEY=your-blablador-api-key
```

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

## Keeping repository search up to date (maintainer)

Repository search does not update itself. JüNA searches what juena-rag ingested,
and nothing ingests new commits automatically. That is on purpose: indexing at
every start used to make startup take minutes and up to 12 GB. The search data
lives in juena-rag's database volume, so starting, stopping or rebuilding JüNA
never re-indexes it and never loses it.

**The maintainer re-ingests every two weeks, and at least once a month**, or sooner
after a release people will ask about. On 2026-10-22, six of the eight repositories
had new commits since their last ingest two months earlier.

From `juena-chatbot`:

```bash
./juena repos-ingest
```

It ingests every repository in `config/repositories.yaml`, the one list of what
JüNA covers, or only the ids you name (`./juena repos-ingest jscatter silx`). A
repository with no new commits prints `already at …; nothing to do` and costs
nothing. A changed one is re-embedded in full, a few minutes each. Search keeps
answering from the old data until each repository's new data is complete.
`--dry-run` only counts.

At the end it moves JüNA's copies of the repositories to the commits just
ingested, so the lines the model quotes are the lines search found. If JüNA is not
running, its next `./juena up` does that.

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

   Install Node.js 24 and npm on PATH as well for the host worker source build.
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
  2–6. Native installs and the host worker also need Node.js 24 and npm on PATH.
- **OpenAI for chat:** use the OpenAI settings in [env.example](env.example).
  Repository search still needs the Blablador key in the chatbot `.env`.
- **FZJ login or a shared server:** use the [login setup guide](SAML-IFFLOGIN.md)
  and [sandbox deployment guide](deploy/SANDBOX.md).
- **Tracing every chat:** set `LANGFUSE_TRACING=true`, then run `./juena langfuse-up` for a Langfuse server on
  <http://localhost:3000>; see the [Langfuse guide](deploy/LANGFUSE.md).
- **More settings:** see [env.example](env.example).
- **Development checks:** run `./juena test`. Run `./juena test-integration`
  for the database checks.

## License

MIT

## Linux development server

Use [deploy/SERVER.md](deploy/SERVER.md) for the single-Compose server setup.
The gateway publishes HTTPS; chatbot and RAG communicate on a private network.
The sandbox worker runs under a host systemd user service using rootless Podman.
Core and RAG source checkouts are unnecessary. Tracing is opt-in.

The administrator and IFFLogin confirmation checklist is maintained locally at
`DOCS/DEV-SERVER-CHECKLIST.md`, ignored under this repository's DOCS policy.
SAML acceptance requires a real browser login, authenticated session, logout,
and repeat login against IFFLogin.
