# JüNA Chatbot

<div align="center">
  <img src="app/assets/logo.png" alt="JüNA logo" width="180"/>
</div>

JüNA is a chat assistant for research work. It helps people find explanations
in scientific software and documentation, explore neutron-science questions,
and work with data and Python plots in the same conversation.

This repository is the complete application: the chat pages, research tools,
sign-in, and instructions for running it. Its current focus is neutron science,
but collaborators can adapt its instructions, tools, and knowledge sources for another
research project.

## How the JüNA projects fit together

| Project | What it provides | Start here when you want to… |
| --- | --- | --- |
| **juena-chatbot** | A complete research assistant | Run JüNA or adapt its existing chat experience |
| [juena-core](https://github.com/neutron-simlab/juena-core) | Shared building blocks for conversations, agents, files, and optional code execution | Build an application with its own workflow and interface |
| [juena-rag](https://github.com/neutron-simlab/juena-rag) | Searchable knowledge with references to the source material | Add document or code search to any application |

The chatbot uses both packages. You only need to clone **this repository** to
run it: its dependency files select the core and RAG versions, and its Docker
setup includes the search service and databases.

## What you can do

- Ask research questions and follow the sources behind an answer.
- Search configured scientific software repositories and inspect relevant code.
- Upload text, DAT, and TIFF files and work with them in a conversation.
- Review and approve Python or shell commands before they run in an isolated
  environment, then collect the resulting plots and files.
- Keep conversations and shared working files, and send longer research tasks
  to the background while continuing to chat.

The assistant combines a coordinating agent with software and neutron-research
specialists. Available tools depend on the deployment and your personal API keys.

## Start using JüNA

**If your group already runs JüNA**, ask its administrator for the website
address. Sign in with **iffLogin**, open **My profile**, and add your personal
Blablador API key. Tavily for web search and Context7 for software documentation
are optional. See [personal API keys](deploy/PERSONAL-API-KEYS.md) for the steps.

Try a question such as “How do I fit a sphere model with jscatter?” or upload
a data file and describe the plot you need. Repository questions require the
administrator to have indexed the relevant repositories; code execution requires
the optional worker to be enabled.

**To run a development copy**, follow [Local development](deploy/LOCAL-DEVELOPMENT.md).
It covers macOS, Linux, and Windows through WSL, including the required tools,
settings, repository indexing, and first sign-in. After completing that setup:

```bash
./juena up       # start the application
./juena health   # check that it is responding
./juena down     # stop it, keeping saved data
```

**To host JüNA for a team**, follow [Server deployment](deploy/SERVER.md).
It covers a dedicated Linux server or a server shared with another application.
On an already configured server:

```bash
sudo ./juena prod up
sudo ./juena prod health
sudo ./juena prod doctor
```

Server commands read `/etc/juena/juena.env`. Local development uses the checkout's
`.env` and a development sign-in option. Use the server guide when other people
will access the application.

## Adapt it for your project

Start with one task your collaborators need to complete, such as finding an
instrument procedure or analyzing a particular measurement. Run that task in
the existing application, then change the parts that depend on your subject.

| What you want to change | Where to start |
| --- | --- |
| The repositories the assistant can search | [Repository configuration](config/repositories.yaml) and [adding a repository](HOW-TO-ADD-REPO.md) |
| How the assistant coordinates its work | [Supervisor instructions](src/juena/agents/SUPERVISOR.md) |
| The specialists' knowledge and responsibilities | [Software instructions](src/juena/agents/specialists/software/AGENT.md) and [research instructions](src/juena/agents/specialists/neutron_research/AGENT.md) |
| Tools or a new specialist | [Agent assembly](src/juena/agents/juena_agent.py), [specialists](src/juena/agents/specialists/), and [tools](src/juena/tools/) |
| Pages, branding, and suggested questions | [The app folder](app/) |
| Sign-in and service settings | [Authentication](src/juena/server/auth/), [local settings](env.example), and [server settings](deploy/server.env.example) |

Adding a searchable repository requires configuration and indexing; it does not
require a new agent. A new specialist is useful when the project needs a distinct
set of tools or a different way of working. Keep these project choices in your
application, so improvements to the shared core and search packages remain reusable.

If you need a substantially different interface or workflow, the
[small juena-core example](https://github.com/neutron-simlab/juena-core/tree/main/examples/simple_chat)
is a simpler starting point. You can still use juena-rag for the project's knowledge.

For changes to this application, run `./juena test`. Database checks are available
through `./juena test-integration`; the [development guide](deploy/LOCAL-DEVELOPMENT.md)
covers the environment they need. Also try a representative research task before
sharing a change with your collaborators.

## Further reading

| Guide | What it helps with |
| --- | --- |
| [Local development](deploy/LOCAL-DEVELOPMENT.md) | Run and develop JüNA on your computer |
| [Server deployment](deploy/SERVER.md) | Install, update, and back up a team installation |
| [Troubleshooting](deploy/TROUBLESHOOTING.md) | Resolve common sign-in, search, and execution problems |
| [Personal API keys](deploy/PERSONAL-API-KEYS.md) | Set up the keys used for chat and optional tools |
| [Repository maintenance](HOW-TO-ADD-REPO.md) | Add knowledge sources and keep them searchable |
| [Detector files](deploy/DETECTOR-FILES.md) | Work with TIFF and DAT files |
| [Steering and Stop](deploy/STEERING-AND-STOP.md) | Adjust or stop an answer while it is running |
| [Sandbox](deploy/SANDBOX.md) | Set up the optional code-execution worker |
| [iffLogin setup](deploy/SAML-IFFLOGIN.md) | Configure institutional sign-in |
| [Shared gateway](deploy/SHARED-PROXY.md) | Host JüNA alongside eLabFTW |
| [Tracing](deploy/LANGFUSE.md) | Inspect runs during development |
| [Deployment record](deploy/BUILD-RECORD.md) | Record what was installed and verified |

## License

MIT
