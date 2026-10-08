package com.dgentechnologies.adam;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.provider.Settings;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

@CapacitorPlugin(name = "Companion")
public class CompanionPlugin extends Plugin {
    private boolean notificationAccess() {
        String listeners = Settings.Secure.getString(getContext().getContentResolver(), "enabled_notification_listeners");
        if (listeners == null) return false;
        android.content.ComponentName expected = new android.content.ComponentName(getContext(), AdamNotificationListener.class);
        for (String value : listeners.split(":")) if (expected.equals(android.content.ComponentName.unflattenFromString(value))) return true;
        return false;
    }
    @PluginMethod public void notificationStatus(PluginCall call) {
        JSObject result = new JSObject();
        result.put("accessGranted", notificationAccess());
        result.put("captureEnabled", NotificationStore.capturing(getContext()));
        result.put("connected", AdamNotificationListener.connected);
        result.put("error", AdamNotificationListener.lastError);
        call.resolve(result);
    }
    @PluginMethod public void getNotifications(PluginCall call) {
        try { JSObject result = new JSObject(); result.put("items", NotificationStore.read(getContext())); call.resolve(result); }
        catch (Exception ignored) { call.reject("Notification history could not be read. Clear the history and try again."); }
    }
    @PluginMethod public void setNotificationCapture(PluginCall call) {
        try {
            boolean enabled = Boolean.TRUE.equals(call.getBoolean("enabled"));
            if (enabled && !notificationAccess()) { call.reject("Enable notification access in Android first."); return; }
            NotificationStore.setCapturing(getContext(), enabled);
            if (enabled && !AdamNotificationListener.connected && android.os.Build.VERSION.SDK_INT >= 24) {
                try {
                    android.service.notification.NotificationListenerService.requestRebind(new android.content.ComponentName(getContext(), AdamNotificationListener.class));
                } catch (RuntimeException ignored) {
                    AdamNotificationListener.lastError = "Android has not connected notification access. Check access in Android settings.";
                }
            }
            call.resolve();
        } catch (Exception ignored) { call.reject("Could not update notification capture."); }
    }
    @PluginMethod public void clearNotifications(PluginCall call) {
        try { NotificationStore.clear(getContext(), Boolean.TRUE.equals(call.getBoolean("reset"))); AdamNotificationListener.lastError = ""; call.resolve(); }
        catch (Exception ignored) { call.reject("Could not clear notification history."); }
    }
    @PluginMethod public void openNotificationAccessSettings(PluginCall call) {
        try {
            Intent intent = new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS);
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                intent = new Intent(Settings.ACTION_NOTIFICATION_LISTENER_DETAIL_SETTINGS);
                intent.putExtra(Settings.EXTRA_NOTIFICATION_LISTENER_COMPONENT_NAME, new android.content.ComponentName(getContext(), AdamNotificationListener.class).flattenToString());
            }
            try { getActivity().startActivity(intent); }
            catch (android.content.ActivityNotFoundException ignored) { getActivity().startActivity(new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)); }
            call.resolve();
        } catch (Exception ignored) { call.reject("Open Android Settings and search for Notification access to enable ADAM."); }
    }
    private static final String ALIAS = "adam.companion.secrets.v1";
    private SharedPreferences prefs() { return getContext().getSharedPreferences("adam_secure", Context.MODE_PRIVATE); }
    @PluginMethod public void setAppearance(PluginCall call) {
        boolean dark = !"light".equals(call.getString("theme"));
        getActivity().runOnUiThread(() -> {
            int color = dark ? android.graphics.Color.BLACK : android.graphics.Color.WHITE;
            getActivity().getWindow().getDecorView().setBackgroundColor(color);
            getActivity().findViewById(android.R.id.content).setBackgroundColor(color);
            getActivity().getWindow().setNavigationBarColor(color);
            androidx.core.view.WindowCompat.getInsetsController(getActivity().getWindow(), getActivity().getWindow().getDecorView())
                .setAppearanceLightNavigationBars(!dark);
            call.resolve();
        });
    }
    private synchronized SecretKey key() throws Exception {
        KeyStore store = KeyStore.getInstance("AndroidKeyStore"); store.load(null);
        if (!store.containsAlias(ALIAS)) {
            KeyGenerator generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
            generator.init(new KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build());
            generator.generateKey();
        }
        return (SecretKey) store.getKey(ALIAS, null);
    }
    private String safeKey(PluginCall call) throws Exception {
        String name = call.getString("key");
        if (name == null || name.isEmpty() || name.length() > 300) throw new Exception("Invalid key");
        return name;
    }
    @PluginMethod public void setSecret(PluginCall call) {
        try {
            String name = safeKey(call), value = call.getString("value");
            if (value == null || value.length() > 100000) { call.reject("Invalid secret"); return; }
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding"); cipher.init(Cipher.ENCRYPT_MODE, key());
            String encrypted = Base64.encodeToString(cipher.getIV(), Base64.NO_WRAP) + ":" +
                Base64.encodeToString(cipher.doFinal(value.getBytes(StandardCharsets.UTF_8)), Base64.NO_WRAP);
            if (!prefs().edit().putString(name, encrypted).commit()) throw new Exception("Storage write failed");
            call.resolve();
        } catch (Exception e) { call.reject("Could not save securely on this device."); }
    }
    @PluginMethod public void getSecret(PluginCall call) {
        try {
            String stored = prefs().getString(safeKey(call), null);
            JSObject result = new JSObject();
            if (stored == null) { result.put("value", org.json.JSONObject.NULL); call.resolve(result); return; }
            String[] parts = stored.split(":", 2);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, key(), new GCMParameterSpec(128, Base64.decode(parts[0], Base64.NO_WRAP)));
            result.put("value", new String(cipher.doFinal(Base64.decode(parts[1], Base64.NO_WRAP)), StandardCharsets.UTF_8));
            call.resolve(result);
        } catch (Exception e) { call.reject("Saved credentials are unavailable. Please sign in again."); }
    }
    @PluginMethod public void removeSecret(PluginCall call) {
        try { if (!prefs().edit().remove(safeKey(call)).commit()) throw new Exception(); call.resolve(); }
        catch (Exception e) { call.reject("Could not remove the saved credential."); }
    }
    @PluginMethod public void clearSecrets(PluginCall call) {
        if (prefs().edit().clear().commit()) call.resolve(); else call.reject("Could not clear saved credentials.");
    }
    @PluginMethod public void openWifiSettings(PluginCall call) {
        try { getActivity().startActivity(new Intent(Settings.ACTION_WIFI_SETTINGS)); call.resolve(); }
        catch (Exception e) { call.reject("Wi-Fi settings are unavailable on this device."); }
    }
    @PluginMethod public void openAppSettings(PluginCall call) {
        try { getActivity().startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + getContext().getPackageName()))); call.resolve(); }
        catch (Exception e) { call.reject("App settings are unavailable on this device."); }
    }
}
