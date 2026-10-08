package com.dgentechnologies.adam;

import android.content.ComponentName;
import android.os.Build;
import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.json.JSONObject;

/** Android binds this service only after the user grants notification access. */
public class AdamNotificationListener extends NotificationListenerService {
    static volatile boolean connected = false;
    static volatile String lastError = "";
    // Keep only recent text snapshots during bursts, bounded like the saved inbox.
    private final ExecutorService writer = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
        new ArrayBlockingQueue<>(100), new ThreadPoolExecutor.DiscardOldestPolicy());
    @Override public void onListenerConnected() { connected = true; lastError = ""; }
    @Override public void onListenerDisconnected() {
        connected = false;
        if (Build.VERSION.SDK_INT >= 24 && NotificationStore.capturing(this)) {
            try { requestRebind(new ComponentName(this, AdamNotificationListener.class)); }
            catch (RuntimeException ignored) { /* Access may have just been revoked. */ }
        }
    }
    @Override public void onNotificationPosted(StatusBarNotification notification) {
        if (notification == null || writer.isShutdown()) return;
        long generation = NotificationStore.captureGeneration(this);
        if (generation < 0) return;
        try {
            JSONObject entry = NotificationStore.snapshot(this, notification);
            if (entry == null) return;
            writer.execute(() -> {
                try { NotificationStore.capture(this, entry, generation); lastError = ""; }
                catch (Exception ignored) { lastError = "Notification history could not be saved. Check available storage."; }
            });
        } catch (Exception ignored) {
            // Other apps may post malformed extras. One such notification must not stop the listener.
        }
    }
    @Override public void onDestroy() {
        connected = false;
        writer.shutdownNow();
        super.onDestroy();
    }
}
