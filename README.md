# JüNA Chatbot

<div align="center">
  <img src="app/assets/logo.png" alt="JüNA logo" width="180"/>
</div>

JüNA helps researchers explore neutron science, search scientific software
repositories, and create Python plots through chat. It uses IFFLogin for server
sign-in, juena-rag for repository search, and an isolated worker for approved
code execution.

The primary deployment is a Linux server. Both a shared nginx gateway and a
dedicated server are supported. Only this checkout is needed: `uv.lock` pins
the core and RAG dependencies, and Docker builds the application.

## Server operation

Start with [the server guide](deploy/SERVER.md) for installation. On an already
configured server, run from the deployed checkout:

```bash
sudo ./juena prod up
sudo ./juena prod health
sudo ./juena prod doctor
```

Server commands read `/etc/juena/juena.env`. The checkout's `.env` is for local
development. See the guides below for indexing, worker setup, updates and shutdown.

## Documentation

| Guide | Contents |
| --- | --- |
| [Server deployment](deploy/SERVER.md) | Installation, both gateway modes, updates, capacity and backups |
| [Troubleshooting](deploy/TROUBLESHOOTING.md) | Login, repository discovery, indexing and sandbox fixes |
| [Shared gateway](deploy/SHARED-PROXY.md) | JüNA alongside eLabFTW; dedicated servers use the server guide |
| [Sandbox](deploy/SANDBOX.md) | Execution worker, workspace permissions and acceptance checks |
| [IFFLogin](SAML-IFFLOGIN.md) | SAML configuration and browser acceptance |
| [Repository maintenance](HOW-TO-ADD-REPO.md) | Add repositories and keep their indexes current |
| [Deployment record](deploy/BUILD-RECORD.md) | Build evidence, reported checks and outstanding acceptance |
| [Tracing](deploy/LANGFUSE.md) | Optional Langfuse integration |
| [Local development](deploy/LOCAL-DEVELOPMENT.md) | Optional macOS, Linux and WSL setup for development/testing |

## License

MIT
