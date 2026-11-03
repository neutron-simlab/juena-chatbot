#!/usr/bin/env bash
# Sourced by ./juena. No host changes occur merely by loading these functions.

production_worker_user() {
    local worker_uid
    worker_uid="$(id -u juena-sandbox)"
    # The administrator's checkout may be under a private home directory.
    # Podman re-enters its working directory after dropping privileges, so it
    # must not inherit that path. A subshell preserves the launcher's cwd.
    (
        cd /
        runuser -u juena-sandbox -- env \
            XDG_RUNTIME_DIR="/run/user/$worker_uid" \
            DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$worker_uid/bus" "$@"
    )
}

production_worker_installed() {
    [ -f /var/lib/juena-sandbox/home/.config/systemd/user/juena-sandbox-worker.service ]
}

production_worker_stop() {
    if production_worker_installed; then
        production_worker_user systemctl --user disable --now juena-sandbox-worker.service
    fi
}

production_worker_image() {
    local identifier app_image imported
    identifier="$("${COMPOSE_PROD[@]}" ps -q juena-chatbot)"
    [ -n "$identifier" ] || die "start the application with './juena prod up' first"
    app_image="$(docker inspect --format '{{.Image}}' "$identifier")"
    imported="$(production_worker_user podman image inspect --format '{{.Id}}' localhost/juena-worker:current 2>/dev/null || true)"
    if [ "${imported#sha256:}" != "${app_image#sha256:}" ]; then
        # Reuse the running app's exact dependency versions. No second checkout,
        # host virtualenv, Node.js installation, or registry push is needed.
        docker save "$app_image" | production_worker_user podman load
        production_worker_user podman tag "$app_image" localhost/juena-worker:current
    fi
}

production_worker_start() {
    local attempt
    production_worker_installed || die "sandbox worker is not installed; run './juena prod sandbox-setup'"
    production_worker_image
    production_worker_preflight
    production_worker_user systemctl --user enable --now podman.socket
    # A running API process may have discovered runtimes before crun was
    # installed. Refresh only this dedicated account's API before the worker
    # connects and requests execution containers with runtime=crun.
    production_worker_user systemctl --user restart podman.service
    production_worker_user systemctl --user enable --now juena-sandbox-worker.service
    # The advisory lock is acquired before startup cleanup. Wait for readiness
    # before submitting a test job that cleanup could otherwise mark failed.
    for attempt in {1..30}; do
        if production_worker_user podman logs juena-sandbox-worker 2>&1 \
                | grep -q 'Sandbox worker ready with'; then return 0; fi
        sleep 1
    done
    production_worker_stop
    die "sandbox worker did not become ready; inspect './juena prod sandbox-logs'"
}

production_worker_preflight() {
    local worker_uid worker_gid workspace
    worker_uid="$(id -u juena-sandbox)"
    worker_gid="$(id -g juena-sandbox)"
    workspace="$(production_setting SANDBOX_WORKSPACE_HOST_ROOT)"
    # Exercise the real keep-id mount setup before waiting for service logs.
    # This short-lived check gets no credentials or runtime socket and mounts
    # the workspace read-only. The queued PNG test checks execution separately.
    production_worker_user podman run --runtime=crun --rm --pull=never \
        --userns=keep-id --user="$worker_uid:$worker_gid" --network=none --read-only \
        --cap-drop=ALL --security-opt=no-new-privileges --memory=512m --pids-limit=256 \
        --volume="$workspace:$workspace:ro" \
        --env=PYTHONDONTWRITEBYTECODE=1 \
        --entrypoint=/app/.venv/bin/python localhost/juena-worker:current \
        -c 'import os, sys; assert os.getuid() != 0; assert os.access(sys.argv[1], os.R_OK | os.X_OK); print("Sandbox container startup passed.")' \
        "$workspace" \
        || die "sandbox container startup check failed; worker not started"
}

production_sandbox_check() {
    "${COMPOSE_PROD[@]}" exec -T juena-chatbot python - "$@" \
        < "$REPO_ROOT/src/juena/deployment/sandbox_check.py"
}

write_production_worker_dsn() {
    "${COMPOSE_PROD[@]}" exec -T juena-chatbot python - "$(production_setting POSTGRES_PORT 5432)" <<'PY'
import os
import sys
from urllib.parse import urlsplit, urlunsplit

p = urlsplit(os.environ["DATABASE_URL"])
if p.hostname != "postgres":
    raise SystemExit("Worker setup expects the bundled Postgres service")
port = int(sys.argv[1])
if not 0 < port < 65536:
    raise SystemExit("Invalid POSTGRES_PORT")
dsn = urlunsplit(p._replace(netloc=p.netloc.rsplit("@", 1)[0] + "@127.0.0.1:" + str(port)))
# This file is read by both systemd and Podman, whose handling of quoted env
# values differs. URLs must encode whitespace/quotes rather than use shell quoting.
if any(c.isspace() or c in "\"'\\" for c in dsn):
    raise SystemExit("Database URL must percent-encode whitespace, quotes and backslashes")
print("DATABASE_URL=" + dsn)
PY
}

enable_production_sandbox() {
    local backup updated
    backup="$(mktemp "$PROD_ENV_FILE.before-sandbox.XXXXXX")"
    cp -p "$PROD_ENV_FILE" "$backup"
    updated="$(mktemp "$PROD_ENV_FILE.sandbox.XXXXXX")"
    awk '/^[[:space:]]*(export[[:space:]]+)?SANDBOX_ENABLED[[:space:]]*=/ { if (!done++) print "SANDBOX_ENABLED=true"; next } { print } END { if (!done) print "SANDBOX_ENABLED=true" }' \
        "$PROD_ENV_FILE" > "$updated"
    chmod 0600 "$updated"
    mv "$updated" "$PROD_ENV_FILE"
    # Run deployment in a child shell so failures/exit cannot skip restoration.
    if env SANDBOX_ENABLED=true PROD_ENV_FILE="$PROD_ENV_FILE" PROD_PROXY_MODE="$PROD_PROXY_MODE" \
            PROD_EDGE_DIR="$PROD_EDGE_DIR" "$REPO_ROOT/juena" prod up \
            && production_sandbox_check --require-enabled; then
        echo "Sandbox execution enabled; previous environment retained at $backup"
    else
        cp -p "$backup" "$PROD_ENV_FILE"
        production_worker_stop
        configure_production_compose
        "${COMPOSE_PROD[@]}" up -d --no-deps --wait --wait-timeout 900 juena-chatbot \
            || echo "Restoring the application environment failed; inspect './juena prod logs'." >&2
        die "sandbox activation failed; restored the previous environment from $backup"
    fi
}

activate_prepared_sandbox() {
    production_sandbox_check --initialize
    production_worker_start
    if ! production_sandbox_check; then
        production_worker_stop
        die "sandbox smoke test failed; app execution setting unchanged. Inspect './juena prod sandbox-logs'"
    fi
    enable_production_sandbox
}

ensure_production_sandbox_dependencies() {
    if command -v podman >/dev/null 2>&1 \
            && command -v newuidmap >/dev/null 2>&1 \
            && command -v newgidmap >/dev/null 2>&1 \
            && command -v crun >/dev/null 2>&1; then
        return 0
    fi
    command -v apt-get >/dev/null 2>&1 || die "install Podman, crun, uidmap and a user D-Bus session first"
    apt-get update
    apt-get install -y podman crun uidmap dbus-user-session slirp4netns fuse-overlayfs
}

write_production_worker_limits() {
    local key value
    # runc 1.2 checks cwd before adopting the job UID/GID and cannot enter an
    # app-owned 2770 workspace. crun retries after adopting the execution user.
    # This selects the runtime per job without changing host engine settings.
    printf 'SANDBOX_OCI_RUNTIME=crun\n'
    for key in SANDBOX_CONCURRENCY SANDBOX_CPU_LIMIT SANDBOX_MEMORY_LIMIT \
               SANDBOX_WORKSPACE_LIMIT_BYTES SANDBOX_IDLE_TTL_SECONDS; do
        value="$(production_setting "$key")"
        [ -n "$value" ] || die "$key must be explicit in the production environment"
        case "$value" in *[!a-zA-Z0-9.]* ) die "invalid worker limit: $key" ;; esac
        printf '%s=%s\n' "$key" "$value"
    done
    printf 'SANDBOX_PID_LIMIT=256\nSANDBOX_TMPFS_SIZE=1g\n'
}

setup_production_sandbox() {
    [ "$(id -u)" -eq 0 ] || die "run sandbox-setup with sudo (installs the dedicated worker account and service)"
    [ -d /run/systemd/system ] && [ -f /sys/fs/cgroup/cgroup.controllers ] \
        || die "sandbox-setup requires Linux with systemd and cgroup v2"
    [ "$(production_setting ENVIRONMENT)" = production ] \
        && [ "$(production_setting AUTH_DEV_BYPASS false)" = false ] \
        || die "sandbox-setup requires production with AUTH_DEV_BYPASS=false"
    local workspace worker_home worker_env key updated execution_image
    workspace="$(production_setting SANDBOX_WORKSPACE_HOST_ROOT)"
    # Permission changes are restricted to the documented host tree.
    [ "$workspace" = /var/lib/juena-sandbox/workspaces ] \
        || die "sandbox-setup expects SANDBOX_WORKSPACE_HOST_ROOT=/var/lib/juena-sandbox/workspaces"
    [ ! -L "$workspace" ] && [ ! -L /var/lib/juena-sandbox ] && [ ! -L "$PROD_ENV_FILE" ] \
        || die "sandbox host directories and production environment must not be symlinks"
    "${COMPOSE_PROD[@]}" exec -T juena-chatbot true \
        || die "start the application with './juena prod up' first"
    worker_home=/var/lib/juena-sandbox/home
    if id juena-sandbox >/dev/null 2>&1; then
        [ "$(getent passwd juena-sandbox | cut -d: -f6)" = "$worker_home" ] \
            || die "existing juena-sandbox account has another home; review its installation first"
    fi
    ensure_production_sandbox_dependencies
    if ! id juena-sandbox >/dev/null 2>&1; then
        useradd --create-home --home-dir "$worker_home" --shell /bin/bash juena-sandbox
    fi
    for key in /etc/subuid /etc/subgid; do
        awk -F: '$1 == "juena-sandbox" && $3 >= 65536 { found=1 } END { exit !found }' "$key" \
            || die "juena-sandbox needs a non-overlapping subordinate ID range in $key"
    done
    chgrp juena-sandbox /var/lib/juena-sandbox
    # keep-id's container root maps to a subordinate host UID, not host root.
    # It cannot bypass a root-owned 0710 ancestor while preparing its rootfs.
    # Permit traversal only; home/workspace contents retain their own modes.
    chmod 0711 /var/lib/juena-sandbox
    install -d -o juena-sandbox -g juena-sandbox -m 2770 "$workspace"
    # Preserve all files and the identity secret; do not follow symlinks.
    find "$workspace" -xdev ! -type l -exec chgrp -h juena-sandbox {} +
    find "$workspace" -xdev -type d -exec chmod g+rws {} +
    find "$workspace" -xdev -type f -exec chmod g+rw {} +
    loginctl enable-linger juena-sandbox
    systemctl start "user@$(id -u juena-sandbox).service"
    production_worker_user systemctl --user enable --now podman.socket
    production_worker_user podman info --format '{{.Host.Security.Rootless}}' | grep -qx true \
        || die "Podman must be rootless for the dedicated worker"
    # Use Docker to build from this checkout; import into the worker's rootless
    # storage. The worker never needs access to the administrator's home.
    docker build -f "$REPO_ROOT/deploy/sandbox/Dockerfile" -t juena-sandbox:latest "$REPO_ROOT"
    execution_image="$(docker image inspect --format '{{.Id}}' juena-sandbox:latest)"
    docker save "$execution_image" | production_worker_user podman load
    production_worker_user podman tag "$execution_image" localhost/juena-sandbox:latest
    production_worker_stop
    production_worker_image
    worker_env=/etc/juena/sandbox-worker.env
    if [ -f "$worker_env" ]; then
        updated="$(mktemp "$worker_env.before-setup.XXXXXX")"
        cp -p "$worker_env" "$updated"
    fi
    updated="$(mktemp /etc/juena/sandbox-worker.env.XXXXXX)"
    chmod 0600 "$updated"
    # Resolve the live DSN without displaying it. Only its database host/port
    # change. Unquoted URL values work in both systemd and Podman env files.
    write_production_worker_dsn > "$updated"
    {
        printf 'SANDBOX_IMAGE=localhost/juena-sandbox:latest\nSANDBOX_WORKSPACE_ROOT=%s\n' "$workspace"
        write_production_worker_limits
    } >> "$updated"
    chown root:juena-sandbox "$updated"
    chmod 0640 "$updated"
    mv "$updated" "$worker_env"
    chgrp juena-sandbox /etc/juena
    chmod 0710 /etc/juena
    for key in "$worker_home" "$worker_home/.config" "$worker_home/.config/systemd" \
               "$worker_home/.config/systemd/user"; do
        [ ! -L "$key" ] || die "worker configuration directories must not be symlinks"
    done
    install -d -o juena-sandbox -g juena-sandbox -m 0700 \
        "$worker_home/.config" "$worker_home/.config/systemd" "$worker_home/.config/systemd/user"
    install -o juena-sandbox -g juena-sandbox -m 0644 \
        "$REPO_ROOT/deploy/juena-sandbox-worker.service" "$worker_home/.config/systemd/user/"
    production_worker_user systemctl --user daemon-reload
    activate_prepared_sandbox
}
