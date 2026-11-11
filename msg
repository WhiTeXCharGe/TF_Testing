Tools 2-6 and 3-1 now read one shared config.txt and run with no options. The online mode is written but untested against real Azure. I only tried it against a fake server. The 2-6 suite (42 tests) and the 3-1 suite (25 tests) pass.

What changed
config.txt is in weekly_planning/. It uses the same key=value style as the SchedulerWeb config.txt, and the tools run from any folder. It holds:
the connection: mode = local or online, azure_url, and local_url (default http://localhost:3001);
for 2-6: schedule, env, out_dir, run_id (the job id), params_file, stage1_minutes, stage2_enabled, stage2_minutes, stage1_unimproved_seconds, stage2_unimproved_seconds, random_seed and allow_cut_fixed_trip;
for 3-1: run_info, result_dir, watch and download (ask, yes or no).
Priority is command line, then config.txt, then the built-in default. Relative paths are relative to the folder of config.txt. An empty value counts as not set, and an unknown key gives a warning, so a typo is not silently ignored.
mode = local works as before.
mode = online sends to azure_url, using the same four requests and no sign-in, like SchedulerWeb.
2-6 calls /health first, which also wakes an idle Container App.
Status, download and health requests are retried on 502/503/504 and connection failures (after 3, 8 and 20 seconds).
A bad address, the placeholder URL or an unreachable server gives a message pointing at azure_url and the network.
A run_id the controller would silently replace is refused before sending.
3-1 takes the server from config.txt. It uses the server in the run record only when neither the command line nor the config says which. It prints a note if the record says the run went somewhere else, for example you submitted locally and then switched to online.
I removed --mode real and the “not available yet” stop. The two modes are now local and online.
The READMEs, the weekly_planning README and the Phase 9 Azure doc describe config.txt, and the doc has a short “after this phase” test. I added a tests file for the config loader.
Notes for tomorrow
Set mode = online and azure_url to the Container App address, with https:// and no trailing path. Then run 2-6 with stage1_minutes = 2 and stage2_enabled = false as a short test.
Leave run_id empty. A run id can be used only once on Azure, and the Batch task would collide on a repeat.
The PC running the tools needs a network path to the Container App. SchedulerWeb has the same requirement.
The new params field and fixed file names only work once the updated controller is deployed. The old controller answers “Unexpected field”.
If a full-size run fails with exit 137, add JVM_MAX_HEAP=12g to the Batch task options in the controller. It is not done yet and is listed in the Phase 9 doc.
config.txt is committed with a placeholder URL, like the SchedulerWeb one. If you put the real Azure address in it, check before you commit it.

I did not re-run the Docker solve through MockServer after this change. The tests cover the request format, and the endpoints are the same as before.
