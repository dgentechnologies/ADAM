# Voice controls and planner access

“ADAM connected” means the PC can read the Pi. Voice commands also need a connection in the other direction, from the Pi to the PC. A log saying `laptop_control → action=brightness_set value=0` followed by `Laptop agent not found` means the command was recognized but never reached Windows.

## Connect both directions

1. On the Pi, ensure `SYNC_TOKEN` in ADAM's `.env` is set to your private connection key. Reuse an existing key; do not replace it unnecessarily. If it is empty, configure a new private key locally and restart ADAM. An empty key intentionally keeps the Pi read-only.
2. In the PC app, open **My ADAM → Connection**. Enter the Pi's address and the same key in **ADAM connection key**, then choose **Connect ADAM**. In older builds, the key field is inside **Advanced connection options**.
3. Choose **Allow laptop control** and wait for ADAM to confirm it verified the computer. This saves the PC's current address, actual service port and laptop key on the Pi; it works without mDNS. The Pi connection key and the PC laptop key are separate credentials.
4. Keep the PC app running. Allow it through Windows Firewall on the trusted network used by ADAM. Both devices must be able to reach each other. Reauthorize after the PC's address or service port changes.
5. In **Controls → Permissions**, enable the desired controls and resume laptop control if paused. Test **Set brightness** locally at a visible level, then ask ADAM to change it by voice. Try an alarm or timer in **My ADAM → Planner** after the read-only notice has cleared.

If guided pairing is unavailable, update the Pi companion service. The existing manual fallback is `LAPTOP_AGENT_IP`, `LAPTOP_AGENT_PORT` and `LAPTOP_AGENT_TOKEN` in the Pi's `.env`, using the PC app's **Manual pairing → Show connection details**. Restart ADAM after changing these values. A previously revoked saved pairing deliberately overrides environment fallback; reauthorize through the app.

## Distinguish failures

- **Read-only / missing key:** enter the Pi's configured key in the desktop app and reconnect.
- **Pi is read-only despite a saved desktop key:** configure `SYNC_TOKEN` on the Pi and restart ADAM. Saving a key only on the PC cannot enable Pi writes.
- **Key rejected:** correct the key and reconnect. The app does not retry rejected writes or remove authentication.
- **Laptop not found:** finish **Allow laptop control**, then check the PC app is running and reachable through Windows Firewall. Changing brightness drivers cannot repair a discovery failure.
- **Brightness driver error after the command reaches Windows:** the display must support WMI or DDC/CI brightness control. For an external monitor, enable DDC/CI in its own menu. Mixed supported/unsupported monitor setups use the first successful brightness response. Returned brightness is the value acknowledged by the driver, which may differ from the requested value.

## Applying the source fixes

The discovery and audio-loop fixes live on the Pi (`laptop_agent_client.py`, `tool_handler.py`, `session.py`). Deploy those updated sources and restart its ADAM process. The planner UI and hardware-response fixes require an updated Windows companion build. Source changes do not alter an already-running or previously packaged `.exe`.

The checked-out project currently lacks `adam-desktop/src/account.py`. Restore that project module before building the full companion on Windows with its documented development dependencies and `python scripts/build.py`. Native packaging, DPAPI and physical Windows controls must be verified there; the cloud regressions use isolated services and simulated display responses.
