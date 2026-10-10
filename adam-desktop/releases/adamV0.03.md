# ADAM Desktop 0.03

Download `adamV0.03.exe` and its `.sha256` sidecar from this directory.

- Windows x64, 22,335,715 bytes; unsigned.
- Source: `e553ec2`.
- SHA-256: `4bd24040e9448b2f7408915d4ae727b1ebe32baa7ee24dad2c22a1f3306594da`.
- Built with Windows CPython 3.11.9 and PyInstaller 6.16.0 under Wine.
- PE architecture, packaged modules/current UI, native OAuth configuration,
  WebView2 loader, timezone data and checksum were inspected successfully.
- Requires Microsoft Edge WebView2 Runtime and supported Windows/.NET runtime.

Includes mandatory Firebase login, account-owned ADAM selection, physical-screen
pairing, protected per-device grants and canonical cloud/Pi record sync.

**Install the matching Pi and ESP32-S3 firmware changes and complete trusted
physical-device registration before pairing.** The executable alone does not
upgrade ADAM or deploy Firebase infrastructure. See
[implementation and rollout](../../docs/DESKTOP_V003_IMPLEMENTATION_AND_ROLLOUT.md)
for setup, validation and remaining production gates. Native Windows hardware
and live Firebase acceptance have not been performed in the Linux build host.
