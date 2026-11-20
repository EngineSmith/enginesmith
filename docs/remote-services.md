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
startup_timeout_s = 30
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
The controller requires a protocol-1 `started` event with its connection token and a positive process ID. It then
accepts exactly one `stopped` event with the same protocol and token, a known reason, and a typed exit-code/signal
record. Duplicate events, unexpected fields, unknown events and out-of-order messages fail the run and invalidate
cleanup confirmation. A `start_failed` event before the handshake reports launch failure, never cleanup success.

The worker sends a terminal event after its process-group cleanup helper returns. `remote_cleanup_confirmed` is
true only after that validated event reports a non-null integer exit code for the owned command. This is the
worker's report that the command leader exited after group signaling; it does not attest that every descendant
has exited. The controller also drains the event stream and requires a successful SSH exit before reporting run
success. A nonzero command exit retains its exit code and cleanup evidence while failing the run. If the transport
disappears before a terminal event, the controller reports cleanup as unconfirmed even though
the remote lease should expire. It never kills a PID from a prior run or searches process names. Normal command
exit, nonzero exit, SIGINT/SIGTERM, lease expiry and EOF all enter the cleanup path.

`lease_s` is 2-300 seconds. Heartbeats are sent every third of the lease. Cleanup allows up to ten additional
seconds for TERM/KILL using the existing process helper. A short lease can expire during network or host stalls;
choose a budget suitable for your environment. `startup_timeout_s` defaults to 30 seconds and bounds the ownership
handshake independently of `timeout_s`; both clocks start before launching SSH, and the earlier deadline wins.
The start-message pipe write is nonblocking and uses that same deadline. Heartbeats begin only after the handshake.
The remote worker independently waits at most 15 seconds for its initial start message. It stops and joins its
control reader during cleanup, including when the controller keeps stdin open after command completion.

Command timeout includes SSH startup. On failure, the controller stops heartbeats, attempts a stop message for at
most one second, and closes stdin. For an acknowledged, valid peer it allows `lease_s + 11` seconds for a cleanup
event. An unacknowledged or invalid peer cannot earn this long extension (at most one second for pending events).
Local SSH teardown can add up to eleven seconds, plus a one-second reader join; a terminal event allows up to two
seconds for transport exit within the command deadline. Failure deadlines bound protocol work, not kernel or host
stalls. No ownership tokens or inherited environment values are
written into the controller's result.

Both directions use newline-terminated UTF-8 JSON objects, limited to 1 MiB per message including the newline.
Reads are bounded before parsing, including when no newline arrives. Duplicate JSON keys, non-finite literals and
malformed UTF-8/JSON are rejected; reader queues hold at most four messages, with backpressure. Controller writes
also have deadlines, so a worker that never reads its pipe cannot make a long manifest bypass startup timeout.

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

CPU tests exercise the worker with actual local processes and a local fake SSH transport: startup timeout,
blocked pipe writes, nonzero exit, failed launch, command-timeout cleanup, malformed/oversized streams, event
ordering, token/schema validation, lease expiry, heartbeats and preservation of unrelated processes. These prove the lease
protocol and cleanup behavior in those tests, not network-partition behavior on production hosts or GPU performance.
Two-host verification and a full P/D recipe remain separate evidence requirements under
[#28](https://github.com/EngineSmith/enginesmith/issues/28).
