# ADAM v30 CONFIG Quick Reference Guide

## What to Change & Where
Open main_pi.py and locate the CONFIG dictionary.

- **Server IP:** CONFIG['SERVER_URL'] = "http://192.168.1.10"
- **Retry Delay:** CONFIG['RETRY_DELAY'] = 5 (seconds)

## Parameter Categories

### Timing & Reliability
| Parameter | Default | Description |
|-----------|---------|-------------|
| FETCH_INTERVAL | 0.1 | Seconds between server polls |
| TIMEOUT | 2.0 | Request timeout duration |

### Audio Settings
| Parameter | Value |
|-----------|-------|
| SAMPLE_RATE | 44100 |
| CHANNELS | 1 |

### Vision & Smoothing
| Parameter | Value |
|-----------|-------|
| FRAME_SKIP | 2 |
| ALPHA_SMOOTH | 0.7 |

## Debugging with Console Output
Enable full debug by setting CONFIG['DEBUG_MODE'] = True. This will show raw packet dumps and detailed thread states.

## Common Tweaks
- **High Latency:** Increase TIMEOUT and decrease FETCH_INTERVAL.
- **Low CPU:** Increase FRAME_SKIP and disable DEBUG_MODE.

## Where is CONFIG Section?
Look for the # --- CONFIGURATION SECTION --- header at the top of the main script file.
