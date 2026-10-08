package com.dgentechnologies.adam;

import android.app.Notification;
import android.content.Context;
import android.content.SharedPreferences;
import android.service.notification.StatusBarNotification;
import org.json.JSONArray;
import org.json.JSONObject;

/** Bounded, private on-device history. Never uploaded or included in backups. */
final class NotificationStore {
    // Invalidates queued callbacks after the user pauses or clears history. No queued work
    // survives a process restart, so this token deliberately stays out of persistent data.
    private static volatile long generation = 0;

    private static SharedPreferences prefs(Context context) {
        return context.getSharedPreferences("adam_notifications", Context.MODE_PRIVATE);
    }
    static boolean capturing(Context context) { return prefs(context).getBoolean("capture", false); }
    static long captureGeneration(Context context) {
        long current = generation;
        return capturing(context) ? current : -1;
    }
    static synchronized void setCapturing(Context context, boolean enabled) {
        generation++;
        if (!prefs(context).edit().putBoolean("capture", enabled).commit()) throw new IllegalStateException("Storage unavailable");
    }
    static synchronized JSONArray read(Context context) throws Exception {
        return new JSONArray(prefs(context).getString("items", "[]"));
    }
    static synchronized void clear(Context context, boolean reset) {
        generation++;
        SharedPreferences.Editor edit = prefs(context).edit().remove("items");
        if (reset) edit.putBoolean("capture", false);
        if (!edit.commit()) throw new IllegalStateException("Storage unavailable");
    }
    private static String clip(CharSequence value, int limit) {
        String text = value == null ? "" : value.toString().trim();
        return text.length() > limit ? text.substring(0, limit) : text;
    }
    /** Copy only text and identifiers; never retain notification bitmaps in the work queue. */
    static JSONObject snapshot(Context context, StatusBarNotification sbn) throws Exception {
        if (sbn.getPackageName().equals(context.getPackageName())) return null;
        Notification notification = sbn.getNotification();
        if (notification == null || notification.extras == null) return null;
        if ((notification.flags & (Notification.FLAG_ONGOING_EVENT | Notification.FLAG_GROUP_SUMMARY)) != 0) return null;
        String title = clip(notification.extras.getCharSequence(Notification.EXTRA_TITLE), 200);
        if (title.isEmpty()) title = clip(notification.extras.getCharSequence(Notification.EXTRA_TITLE_BIG), 200);
        CharSequence content = notification.extras.getCharSequence(Notification.EXTRA_BIG_TEXT);
        String body = clip(content, 1500);
        if (body.isEmpty()) body = clip(notification.extras.getCharSequence(Notification.EXTRA_TEXT), 1500);
        if (body.isEmpty()) {
            CharSequence[] lines = notification.extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES);
            if (lines != null) {
                StringBuilder joined = new StringBuilder();
                for (CharSequence line : lines) {
                    String text = clip(line, 1500 - joined.length());
                    if (text.isEmpty()) continue;
                    if (joined.length() > 0) joined.append('\n');
                    joined.append(text);
                    if (joined.length() >= 1500) break;
                }
                body = clip(joined, 1500);
            }
        }
        if (title.isEmpty() && body.isEmpty()) return null;
        return new JSONObject()
            .put("id", sbn.getKey()).put("packageName", sbn.getPackageName())
            .put("title", title).put("body", body)
            .put("postedAt", sbn.getPostTime()).put("receivedAt", System.currentTimeMillis());
    }

    static synchronized void capture(Context context, JSONObject entry, long expectedGeneration) throws Exception {
        if (generation != expectedGeneration || !capturing(context)) return;
        String app = entry.getString("packageName");
        try {
            app = context.getPackageManager().getApplicationLabel(context.getPackageManager().getApplicationInfo(app, 0)).toString();
        } catch (Exception ignored) { }
        entry.put("appName", clip(app, 200));
        JSONArray previous = read(context), next = new JSONArray();
        next.put(entry);
        for (int i = 0; i < previous.length() && next.length() < 100; i++) {
            JSONObject item = previous.getJSONObject(i);
            if (!entry.getString("id").equals(item.optString("id"))) next.put(item);
        }
        if (!prefs(context).edit().putString("items", next.toString()).commit()) throw new IllegalStateException("Storage unavailable");
    }
}
