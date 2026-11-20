# Lease-owned remote commands

`enginesmith remote-exec` supervises one foreground command over SSH. The remote worker owns its process group
and stops it if the control connection closes or its heartbeat lease expires. This is a building block for
multi-node service experiments, not a scheduler or a complete P/D deployment manager.

Install the same EngineSmith revision and Python 3.12+ on the controller and remote host. Configure an SSH alias
with verified host keys, the desired user, port and identity in your local SSH configuration. The command uses
batch mode; it does not change host-key checking, authorize keys, copy code or install anything remotely.

```toml
version = 1
host = "worker-a"
python = "python3"
cwd = "."
command = ["python3", "-c", "print('remote command completed')"]
inherit_env = ["PATH", "HOME"]
lease_s = 15
timeout_s = 60
```

```sh
uv run enginesmith remote-exec remote.toml --out results/remote-01
```

Remote cwd/interpreter paths are resolved on the remote host. A relative cwd is relative to the SSH session's
initial working directory. `env` supplies explicit string values; `inherit_env` names variables inherited from
the remote worker's environment, not from the local client. Use SSH configuration aliases in examples and keep
real infrastructure identifiers and credentials out of shared manifests. Commands and explicit environment are
recorded in local results; `remote.log` contains remote stdout/stderr and SSH diagnostics. Review them before sharing.

## Ownership and failure behavior

The controller creates a fresh random ownership token per connection. Only the worker for that connection owns
the command group. Heartbeats renew a monotonic-clock lease; absent, invalid or mismatched controls stop the group.
The worker sends a terminal event after cleanup. `remote_cleanup_confirmed` is true only after that event reports
the command exited. If the transport disappears first, the controller reports cleanup as unconfirmed even though
the remote lease should expire. It never kills a PID from a prior run or searches process names. Normal command
exit, nonzero exit, SIGINT/SIGTERM, lease expiry and EOF all enter the cleanup path.

`lease_s` is 2-300 seconds. Heartbeats are sent every third of the lease. Cleanup allows up to ten additional
seconds for TERM/KILL using the existing process helper. A short lease can expire during network or host stalls;
choose a budget suitable for your environment. Command timeout includes SSH startup. The remote worker also
limits how long it waits for its initial start message. No ownership tokens or inherited environment values are
written into the controller's result.

The inference command is an argv array sent as JSON over stdin and runs without a shell. SSH itself starts a
fixed worker module using a quoted interpreter argument. All model commands must remain in the foreground;
daemonized children or children escaping the owned process group are outside this guarantee. A hard kill of
the remote worker, kernel/host failure or a blocked kernel process can prevent cleanup. Full process-tree/resource
isolation requires an external cgroup/container manager and is not provided here.

## Composition and current limits

Use one `remote-exec` controller per role under `service-check`; readiness must probe the actual service endpoint,
not merely the existence of an SSH process. The local cleanup grace must allow the remote terminal event to
arrive; if it does not, treat cleanup as unconfirmed and inspect the remote host before starting another run.
This PR does not allocate GPUs/NICs, reserve ports, transfer model/KV state, arbitrate multiple controllers,
collect files from remote disks, or implement automatic restarts. Remote command logs stream back over SSH.

CPU tests exercise the worker with actual local processes and a local fake SSH transport. These prove the lease
protocol and cleanup behavior in those tests, not network-partition behavior on production hosts or GPU performance.
Two-host verification and a full P/D recipe remain separate evidence requirements under
[#28](https://github.com/EngineSmith/enginesmith/issues/28).
