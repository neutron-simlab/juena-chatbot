# Supervisor and First-Class Specialists

This document describes JueNA's implemented prompt, specialist, and skill
architecture. The previous monolithic prompt extraction and global skill design
have been retired.

## Runtime shape

```text
src/juena/agents/
├── juena_agent.py
├── SUPERVISOR.md
├── backends.py
├── specialist_runtime.py
└── specialists/
    ├── software/
    │   ├── agent.py
    │   ├── tools.py
    │   └── AGENT.md
    └── neutron_research/
        ├── agent.py
        ├── tools.py
        └── AGENT.md
```

`juena_agent.py` is assembly only. It initializes process-wide resources, builds
both specialists explicitly, registers their `CompiledSubAgent` entries with
Deep Agents' `SubAgentMiddleware`, and compiles the supervisor with the
top-level checkpointer and Store.

There is no specialist registry, capability framework, filesystem discovery, or
import discovery. Adding a specialist requires a package, a builder import, and
one explicit entry in the supervisor's specialist list.

## Ownership

The supervisor owns:

- greetings and clarification;
- explicit remember and forget requests;
- specialist selection and task formulation;
- reconciliation, evidence preservation, and the only user-facing answer;
- the runtime-selected user model and top-level conversation checkpoint.

The supervisor receives no repository, Context7, Tavily, execution, or global
skill tools. Its filesystem tools expose only per-user `/memories/AGENTS.md`.
Staged files remain in graph state so Deep Agents can pass them to a specialist,
but supervisor filesystem calls cannot read them.

The software specialist owns:

- MiniMax M2.7;
- indexed repository search and read-only `/repos` access when healthy;
- read-only staged `/inputs`;
- Context7 tools when configured;
- software-specific optional skills.

The neutron-research specialist owns:

- Blablador GPT-OSS 120B;
- neutron-science and broader factual research in JueNA's domain;
- read-only staged `/inputs`;
- Tavily when configured;
- neutron-research-specific optional skills.

Both specialists remain registered when optional capabilities are unavailable.
Their runtime prompt suffix states exactly what is available and forbids invented
repository evidence, documentation checks, or citations.

## Delegation

`SUPERVISOR.md` contains the behavioral policy. Every substantive domain request
is delegated. One specialist is used by default; multiple task calls are
reserved for explicit comparisons or genuinely independent aspects requiring
different expertise.

Saved memory is untrusted preference data. The supervisor may use a relevant
preference to choose a specialist or phrase a task, but it never forwards the
complete memory file and never lets memory override the current request or
verified evidence.

Each specialist returns a private Markdown report:

```markdown
## Finding
## Evidence
## Actions
## Artifacts
## Limitations
```

Artifact rows use `artifact ID | filename | MIME type | caption`, or `None` when
the specialist produced no artifact. The supervisor removes duplication,
resolves conflicts where evidence permits, preserves limitations, and writes the
final answer.

## Skills

Skills are optional and specialist-owned:

```text
specialists/<specialist>/skills/<skill-name>/SKILL.md
```

The specialist backend mounts authored skills read-only at `/skills/`.
`SkillsMiddleware` is installed only when at least one valid `SKILL.md` exists.
There are no placeholder skills, global skill directory, discovery endpoint, or
UI skill chips.

## Artifacts and execution

This refactor preserves read-only repository and staged-input behavior. It does
not add code execution. The future sandbox design remains in
[`SANDBOX-PLAN.md`](SANDBOX-PLAN.md).

When execution is added, the executing specialist creates plots and persists
validated artifacts itself. Artifact events travel directly from that specialist
execution path and are queued for the turn. The supervisor receives artifact
references in the report, never image bytes and never plotting code to rerun.
The UI renders queued artifacts beneath the completed supervisor answer.

## Compatibility

The public agent id remains `juena`; `code_chat_agent` and `react_agent` remain
aliases. Request and response schemas, SSE event types, client APIs, chat
history, attachments, starter prompts, process-wide Context7 lifecycle, fixed
GPT-OSS summarization, retries, model and tool limits, and runtime supervisor
model selection are unchanged.

Packaged prompt and future specialist skill Markdown must be included in wheels
and container images. Tests cover prompt loading, explicit registration, tool
ownership, truthful degradation, skills isolation, routing, memory filtering,
staged inputs, and suppression of specialist reasoning from the user stream.
