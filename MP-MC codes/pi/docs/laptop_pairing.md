# Authorizing a desktop without editing configuration files

The Pi supports an authenticated local pairing extension for the Windows app.
These source changes have isolated tests; they have not been deployed to a Pi.

1. Complete ADAM's network setup in mobile, then connect desktop to the Pi's
   hostname/IP using its data connection key (`SYNC_TOKEN`).
2. The desktop offers **Allow ADAM to control this PC**. Only after that button
   is confirmed, it sends `POST /api/laptops/pair` with
   `{host:<PC's private IP>,port:<agent port>,token:<PC-generated key>}` and the
   Pi's `X-ADAM-Token` header.
3. The Pi validates the private IP, port and key, then calls
   `GET http://<PC>:<port>/pair/verify` with `X-ADAM-Token:<PC key>`.
   The response must be exactly identifiable as the desktop service:
   `{status:"ok",app:"ADAM Companion",kind:"laptop",api:1}`. No laptop action
   runs during verification. Redirects and environment HTTP proxies are disabled.
4. Only after verification and an atomic private file save does the Pi return
   `{ok:true,api:1,paired:true,host,port}`. Keys never appear in acknowledgements,
   status responses, telemetry or logs.

`GET /api/ping` advertises `capabilities.laptop_pairing:true` when the service is
initialized. The existing sync authorization guards pair and unpair. Missing or
incorrect keys return403; invalid/unreachable laptop targets return400; storage
failure returns500 and keeps the previously active pairing.

`POST /api/laptops/unpair` with the same Pi authorization returns
`{ok:true,api:1,paired:false}` after saving a revocation marker. That marker disables
the old environment/mDNS fallback, including after a restart. A future explicit
pair replaces it. Devices that have never used this pairing flow retain their
existing environment/mDNS setup.

The file `laptop_pairing.json` lives beside the Pi runtime and uses mode0600.
It contains the PC control key; exclude it from source control, shared diagnostics
and distribution bundles. It is distinct from `SYNC_TOKEN` and Firebase account
credentials. No `.env` values are rewritten.

`laptop_agent_client.py` snapshots host, port and key together for each request.
An explicit pairing remains pinned to that endpoint; discovering another laptop
cannot silently send the key to it. Existing `/actions`, `/control`, action aliases
and value types remain unchanged. A changed DHCP address requires the desktop to
authorize the new address again. Currently one active laptop endpoint is supported;
this extension does not claim multi-laptop selection.

All verification and file writes run outside the Pi event loop. A request already
sent may complete during revocation; queued/new work observes the revoked state.
The Windows app should additionally enforce its own Pause and action toggles.

Run `python -m unittest discover -s "MP-MC codes/pi/tests" -v` from the repository
root. Pairing tests use isolated loopback laptop and Pi HTTP servers, temporary
storage and fake configuration; they cover auth, identity validation, redirects,
disk errors, alias dispatch, restart persistence and revocation. Real Windows
Firewall behavior, physical network setup and hardware controls still need device
verification.
