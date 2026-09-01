import type { CapacitorConfig } from '@capacitor/cli';

/**
 * Capacitor wraps @adam/web's static export directly from ../web/out, so the
 * build order is always: pnpm --filter @adam/web build → pnpm cap:sync.
 *
 * No native plugin is wired yet (BLE, camera, mDNS). Every native capability is
 * reached through apps/web/src/lib/native/*, which falls back to a browser stub
 * when Capacitor.isNativePlatform() is false — so the same bundle runs in a
 * plain browser without throwing.
 */
const config: CapacitorConfig = {
  appId: 'com.dgentechnologies.adam',
  appName: 'ADAM',
  webDir: '../web/out',
  /** True black, so no white flash appears before the first paint. */
  backgroundColor: '#000000',
  android: {
    backgroundColor: '#000000',
    allowMixedContent: false,
    /** Local HTTP to the Pi is explicitly permitted via network-security-config. */
    webContentsDebuggingEnabled: true,
  },
  ios: {
    backgroundColor: '#000000',
    contentInset: 'never',
  },
  plugins: {
    SplashScreen: {
      launchAutoHide: false,
      backgroundColor: '#000000',
      androidScaleType: 'CENTER_CROP',
      showSpinner: false,
      splashFullScreen: true,
      splashImmersive: true,
    },
    StatusBar: {
      style: 'DARK',
      backgroundColor: '#000000',
      overlaysWebView: true,
    },
    Keyboard: {
      resize: 'native',
      style: 'DARK',
      resizeOnFullScreen: true,
    },
  },
};

export default config;
