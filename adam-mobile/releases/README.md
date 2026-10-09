# ADAM Android releases

Current artifact: [ADAM-0.2.2-release.apk](ADAM-0.2.2-release.apk), with its
[SHA-256 checksum](ADAM-0.2.2-release.apk.sha256). Package:
`com.dgentechnologies.adam`, version 0.2.2/code 10, Android 6+ with a current
System WebView. Update Windows to `adamV0.02.exe` for shared protocol version 2.

This update fixes startup recovery and authentication, uses the desktop logo,
restores device naming in the existing onboarding flow, and adds shared to-dos,
clocks and multiple named ADAM devices. BLE remains simulated.

Install the APK over the existing app to preserve local data when Android accepts
the signing identity. Keep the same release key for future updates. Earlier
versioned APKs remain here.

See [release verification](../docs/RELEASE_0.2.2.md) and the
[build/signing guide](../docs/ANDROID_RELEASE.md). Live Google login and real
cloud exchange require the Firebase registration and account checks documented
there; automated tests are not proof of successful live authentication.
