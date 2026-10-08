import type { CapacitorConfig } from '@capacitor/cli';

/**
 * Capacitor wraps @adam/web's static export directly from ../web/out, so the
 * build order is always: pnpm --filter @adam/web build → pnpm cap:sync.
 *
 * Native camera, sharing, preferences, account persistence and system settings
 * use Capacitor plugins. ADAM hardware transport is intentionally deferred.
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
    webContentsDebuggingEnabled: false,
  },
  ios: {
    backgroundColor: '#000000',
    contentInset: 'never',
  },
  plugins: {
    SplashScreen: {
      launchAutoHide: true,
      launchShowDuration: 1500,
      backgroundColor: '#000000',
      androidScaleType: 'CENTER_CROP',
      showSpinner: false,
      splashFullScreen: true,
      splashImmersive: true,
    },
    StatusBar: {
      style: 'LIGHT',
      backgroundColor: '#000000',
      overlaysWebView: false,
    },
    Keyboard: {
      resize: 'native',
      style: 'DARK',
      resizeOnFullScreen: true,
    },
    GoogleAuth: {
      scopes: ['profile', 'email'],
      serverClientId: '759320300226-spjdvgmqm9ccm622v6l80slbusokvcsp.apps.googleusercontent.com',
      clientId: '759320300226-spjdvgmqm9ccm622v6l80slbusokvcsp.apps.googleusercontent.com',
      androidClientId: '759320300226-spjdvgmqm9ccm622v6l80slbusokvcsp.apps.googleusercontent.com',
      forceCodeForRefreshToken: false,
    },
  },
};

export default config;
