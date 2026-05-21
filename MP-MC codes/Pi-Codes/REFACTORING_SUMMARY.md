# ADAM v30 Pi Wi-Fi Integration — Refactoring Summary

## CONFIG
The system now uses a centralized configuration dictionary or class. This allows for rapid adjustment of global parameters like server endpoints, timeouts, and thresholds without digging through utility functions.

## DEBUG LOGGING
Implemented a tiered logging system (DEBUG, INFO, WARNING, ERROR). Logs are color-coded in supported terminals and include timestamps and component tags (e.g., [NET], [SYS]).

## EXCEPTION HANDLING
Global try-except blocks now catch unexpected failures in peripheral threads (Wi-Fi, Audio). Most errors will trigger a retry sequence rather than a full process crash.

## STRUCTURED INITIALIZATION
Initialization is now sequential and verified:
1. System/Environment check
2. Network/Wi-Fi connection
3. Peripheral/Sensor warmup
4. Main loop entry

## GRACEFUL SHUTDOWN
The script handles SIGINT (Ctrl+C) and SIGTERM. It correctly closes open sockets, stops audio streams, and releases GPIO pins before exiting.

## TASK LOGGING
Background tasks (like audio fetching) log their heartbeat. If a task becomes unresponsive, a warning is issued to the main loop.

## MAIN LOOP LOGGING
The master loop now logs cycle time every 100 iterations (configurable) to monitor for CPU bottlenecking or latency spikes.

## How to Use
- **Start:** python main_pi_v30.py
- **Configure:** Edit the CONFIG section at the top of the script.
- **Log Level:** Change CONFIG['DEBUG_LEVEL'] to 'VERBOSE' for more detail.

## Testing Checklist
- [ ] Wi-Fi Auto-reconnect on signal loss
- [ ] Memory usage stability over 1 hour
- [ ] Socket timeout handling
- [ ] Shutdown cleanup verification

## Benefits Summary
| Feature | Benefit |
|---------|---------|
| Central Config | Faster iteration and easier deployment |
| Better Logging | Reduced MTTD (Mean Time To Detection) for bugs |
| Graceful Exit | Prevents hardware/file lock issues |

## Files Changed
- main_pi.py: Integrated new logic
- config.py: New configuration handler
- logger_util.py: Enhanced logging functions

## Backward Compatibility
Maintains full compatibility with ADAM v20 API endpoints; only the local Pi implementation has been upgraded.
