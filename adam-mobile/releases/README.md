# ADAM Companion App - Releases

This directory contains the standalone Android APK release build artifact for the ADAM mobile companion app.

## Current Release: ADAM 0.2.1

`ADAM-0.2.1-release.apk` is the signed production release build from **8 October 2026**, package `com.dgentechnologies.adam`, version code **8**, compatible with Android 6+ (API 23+) with a current System WebView.

### Artifact Specifications

- **File Name**: `ADAM-0.2.1-release.apk`
- **Application ID**: `com.dgentechnologies.adam`
- **Version Name**: `0.2.1`
- **Version Code**: `8`
- **Build Type**: Signed Release (`release` variant with `adam-release.jks`)
- **File Size**: `8,042,490 bytes` (7.67 MB)
- **SHA-256**:
  ```text
  cec742cb23b1e9233e6473be04b00cbe4ea1a3ac26dbb8d9e653f181654576d0
  ```

### Release Highlights & Architecture

1. **Google Play Protect Compliance**:
   - Clean manifest without invasive background notification listener (`BIND_NOTIFICATION_LISTENER_SERVICE` removed).
   - Release signed with production RSA-3072 keystore (`v1` + `v2` APK Signature Schemes).
   - Passes Google Play Protect on-device scanning.

2. **Complete Biometric Setup Flow**:
   - Restored sequential 13-step onboarding pipeline:
     `welcome` → `sign-in` → `discover` → `device-found` → `wifi-select` → `wifi-password` → `connecting` → `founder-reveal` → `ai-brain` → `byok` → `credits` → `camera-permission` → `face-capture` → `home`.
   - Dynamic 3-angle biometric face capture (Front, Left, Right) with head contour guide, circular progress indicator, and on-device photo persistence.
   - Zero biometric cloud storage: facial enrollment remains strictly on-device in LocalStorage / Preferences.

3. **Authentication & Identity**:
   - Official 4-color Google "G" logo branding for Google Sign-In.
   - Dual authentication options: Native Google Sign-In via Firebase Auth and Passwordless Email Sign-In.
   - Seamless skip option for quick local setup testing.

4. **Visual & UI Polish**:
   - OLED True Black `#000000` obsidian design system.
   - Dynamic interactive dot-matrix background with `CanvasRevealEffect` and `.digital-skin` texture.
   - Centered glowing ADAM eye animations with radial bloom.

### Installation Instructions

1. Copy `ADAM-0.2.1-release.apk` to your Android device via USB, local download, or file transfer.
2. Open the file on your device using any File Manager app.
3. Allow the *"Install unknown apps"* permission if prompted by Android.
4. Tap **Install** and launch **ADAM**.

### Checksum Verification

To verify the APK checksum before installing:

```bash
# Linux / macOS
sha256sum -c ADAM-0.2.1-release.apk.sha256

# Windows PowerShell
Get-FileHash ADAM-0.2.1-release.apk -Algorithm SHA256
```
