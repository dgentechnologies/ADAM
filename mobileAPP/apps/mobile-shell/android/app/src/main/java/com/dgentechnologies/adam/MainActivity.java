package com.dgentechnologies.adam;
import android.os.Bundle;
import android.os.Build;
import android.view.View;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import com.getcapacitor.BridgeActivity;
public class MainActivity extends BridgeActivity {
    @Override public void onCreate(Bundle savedInstanceState) {
        registerPlugin(CompanionPlugin.class);
        super.onCreate(savedInstanceState);
        if (Build.VERSION.SDK_INT >= 35) {
            WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
            View content = findViewById(android.R.id.content);
            ViewCompat.setOnApplyWindowInsetsListener(content, (view, windowInsets) -> {
                Insets bars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout());
                Insets keyboard = windowInsets.getInsets(WindowInsetsCompat.Type.ime());
                view.setPadding(bars.left, bars.top, bars.right, Math.max(bars.bottom, keyboard.bottom));
                return WindowInsetsCompat.CONSUMED;
            });
            ViewCompat.requestApplyInsets(content);
        }
    }
    @Override public void onBackPressed() {
        if (getBridge() == null) { moveTaskToBack(true); return; }
        getBridge().getWebView().evaluateJavascript(
            "(function(){if(window.__adamHandleBack)return window.__adamHandleBack();return 'minimize';})()",
            value -> { if (!"\"handled\"".equals(value)) moveTaskToBack(true); });
    }
}
