# Terminal Debug Output Reference

## Status Indicators
| Icon | Level | Description |
|------|-------|-------------|
| ✅ | SUCCESS | Component initialized or task completed |
| ❌ | ERROR | Critical failure requiring restart or attention |
| ⚠️ | WARNING | Non-critical issue; system will retry |
| 🛑 | STOP | System is shutting down |

## Component Status Indicators
- [NET] ✅ : Connected to Access Point
- [NET] ❌ : Connection Timeout
- [SYS] ⚠️ : High CPU Usage detected
- [AUD] ✅ : Audio stream buffer ready
- [VIS] 🛑 : Camera module detached

## Common Output Patterns
- [INFO] [NET] Sending Heartbeat...
- [DEBUG] [SYS] RAM Usage: 45%
- [ERROR] [AUD] Could not open Alsa device

## Debugging Flowchart
1. **Check ✅ icons:** Ensure all core systems initialized.
2. **Watch for ❌:** If found, check the error code following the icon.
3. **Analyze ⚠️:** If warnings repeat, check network signal or power supply.
