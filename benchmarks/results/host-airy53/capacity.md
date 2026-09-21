# Capacity reassessment: issue 53 versus issue 52

Same cold full-store protocol, historical fixtures, two component streams and
64 MB bound as the [issue 52 matrix](../planning52/capacity.md). The columns below
compare the two adjacent revisions in the stack. These are not repeated warm
speed measurements.

| Machine / fixture | Single visibility | Issue 52 | Issue 53 |
| --- | ---: | --- | --- |
| mini / `aa4-out-of-core` | 7.98 GiB | PASS, 168.5 s, 2.54 GiB RSS | PASS, 166.2 s, 2.50 GiB RSS |
| mini / `aa1-host-stress` | 1.88 GiB | PASS, 64.8 s, 2.69 GiB RSS | PASS, 65.6 s, 2.76 GiB RSS |
| mini / `aa4-host-out-of-core` | 31.94 GiB | Disk preflight SKIP | Disk preflight SKIP |
| Daint / `aa4-out-of-core` | 7.98 GiB | PASS, 185.2 s, 4.72 GiB RSS | PASS, 178.6 s, 4.56 GiB RSS |
| Daint / `aa1-host-stress` | 1.88 GiB | PASS, 97.9 s, 5.17 GiB RSS | PASS, 97.1 s, 5.54 GiB RSS |
| Daint / `aa4-host-out-of-core` | 31.94 GiB | Allocation-limited TIMEOUT at 357 s, 4.58 GiB RSS | TIMEOUT at 400 s, 4.84 GiB RSS |

Times are whole-process wall times including startup, validation and cleanup;
RSS is externally sampled process-tree memory. Exact results, cold simulation
times and allocation statistics are in [capacity.json](capacity.json). Neither
minor cold-run timing differences nor reserved JAX pool sizes establish a speed
or active-device-memory advantage.

The largest mini case needs 208.04 GB projected disk space versus 94.86 GB usable
scratch after reserve. No guard was bypassed. Daint uses GH200 and approximately
856 GiB host RAM; none of these visibility arrays exceeds that machine's physical
memory. The successful 7.98 GiB mini case is also smaller than its 16 GiB RAM.
Consequently this round makes no completed beyond-physical-memory claim.

Five-repeat promotion of these cold passes requires a repeated staged full-store
protocol: the current capacity plugin intentionally rejects warm mode. It also
needs sufficient scratch and allocation time; the largest case has explicit disk
and runtime blockers. The five-repeat 512-RFI workload in the main report is the
qualified timing comparison for this PR.

The final issue 53 trial used a separate debug allocation to give it a full
400-second allowance after the first allocation expired. Components completed
by 198 s, `vis_obs` completed before 367 s, and the deadline interrupted
`vis_calibrated`. Host RSS stayed below the 32 GiB guard. This is a timeout,
not a completed store or an observed OOM. The run used another allocated node;
partial phase progress is not a controlled speed comparison against issue 52's
shortened 357-second trial.
