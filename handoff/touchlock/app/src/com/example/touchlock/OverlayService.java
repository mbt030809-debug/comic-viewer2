package com.example.touchlock;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.os.Handler;
import android.os.IBinder;
import android.view.Display;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;

// 화면 전체에 "Claude 조작 중" 테두리와 빨간 정지 버튼을 그린다(터치는 받지 않음: 터치는 touchlock이 직접 읽음).
// touchlock(127.0.0.1:38999)에 접속해 버튼 위치·화면 회전을 알려 주고, 연결이 끊기면 스스로 사라진다.
public class OverlayService extends Service {
    static final int PORT = 38999;
    WindowManager wm;
    Ov view;
    Handler h;
    volatile OutputStream out;
    volatile boolean alive = true;
    String lastSent = "";

    public IBinder onBind(Intent i) { return null; }

    public int onStartCommand(Intent intent, int flags, int startId) {
        if (view != null) return START_NOT_STICKY;
        h = new Handler();
        goForeground();
        wm = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        view = new Ov(this);
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT,
                2038 /* TYPE_APPLICATION_OVERLAY */,
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE | WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT);
        lp.gravity = Gravity.TOP | Gravity.LEFT;
        lp.alpha = 0.8f; // 0.8 이하여야 아래 앱이 (Claude의) 터치를 '가려짐'으로 거부하지 않음
        setCutoutAlways(lp);
        wm.addView(view, lp);
        new Thread(new Runnable() { public void run() { link(); } }).start();
        h.postDelayed(new Runnable() { public void run() { sendRect(); if (alive) h.postDelayed(this, 500); } }, 300);
        return START_NOT_STICKY;
    }

    void setCutoutAlways(WindowManager.LayoutParams lp) {
        try { lp.getClass().getField("layoutInDisplayCutoutMode").setInt(lp, 3); } catch (Throwable t) {}
    }

    void goForeground() {
        try {
            NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            Class<?> chc = Class.forName("android.app.NotificationChannel");
            Object ch = chc.getConstructor(String.class, CharSequence.class, int.class).newInstance("lock", "터치 잠금", 2);
            nm.getClass().getMethod("createNotificationChannel", chc).invoke(nm, ch);
            Notification.Builder b = Notification.Builder.class.getConstructor(Context.class, String.class).newInstance(this, "lock");
            b.setContentTitle("Claude가 폰을 조작 중").setContentText("빨간 정지 버튼이나 볼륨 아래 키로 멈춤")
                    .setSmallIcon(android.R.drawable.ic_lock_lock);
            startForeground(1, b.build());
        } catch (Throwable t) { t.printStackTrace(); }
    }

    // 버튼 위치(현재 화면 좌표)와 회전을 touchlock에 보낸다
    void sendRect() {
        if (view == null || view.getWidth() == 0) return;
        Display d = wm.getDefaultDisplay();
        int[] loc = new int[2];
        view.getLocationOnScreen(loc);
        RectF r = view.btn;
        String s = "R " + d.getRotation() + " " + view.getWidth() + " " + view.getHeight() + " "
                + (int) (loc[0] + r.left) + " " + (int) (loc[1] + r.top) + " "
                + (int) (loc[0] + r.right) + " " + (int) (loc[1] + r.bottom) + "\n";
        OutputStream o = out;
        if (o == null || s.equals(lastSent)) return;
        final String fs = s;
        lastSent = s;
        new Thread(new Runnable() { public void run() {
            try { o.write(fs.getBytes()); o.flush(); } catch (Throwable t) {}
        } }).start();
    }

    void link() {
        Socket sk = null;
        for (int i = 0; i < 25 && sk == null; i++) {
            try { sk = new Socket("127.0.0.1", PORT); } catch (Throwable t) { try { Thread.sleep(200); } catch (Throwable u) {} }
        }
        if (sk != null) {
            try {
                out = sk.getOutputStream();
                lastSent = "";
                InputStream in = sk.getInputStream();
                byte[] buf = new byte[64];
                while (in.read(buf) >= 0) {}
            } catch (Throwable t) {}
        }
        h.post(new Runnable() { public void run() { stopSelf(); } });
    }

    public void onDestroy() {
        alive = false;
        if (view != null) { try { wm.removeView(view); } catch (Throwable t) {} view = null; }
        super.onDestroy();
    }

    static class Ov extends View {
        final Paint border = new Paint(Paint.ANTI_ALIAS_FLAG), fill = new Paint(Paint.ANTI_ALIAS_FLAG),
                text = new Paint(Paint.ANTI_ALIAS_FLAG), chip = new Paint(Paint.ANTI_ALIAS_FLAG);
        final RectF btn = new RectF();
        final float dp;

        Ov(Context c) {
            super(c);
            dp = c.getResources().getDisplayMetrics().density;
            border.setStyle(Paint.Style.STROKE); border.setColor(Color.rgb(229, 57, 53)); border.setStrokeWidth(4 * dp);
            fill.setColor(Color.rgb(211, 47, 47));
            chip.setColor(Color.argb(200, 0, 0, 0));
            text.setColor(Color.WHITE); text.setTextSize(15 * dp); text.setFakeBoldText(true); text.setTextAlign(Paint.Align.CENTER);
        }

        protected void onDraw(Canvas c) {
            int w = getWidth(), h = getHeight();
            c.drawRect(2 * dp, 2 * dp, w - 2 * dp, h - 2 * dp, border);
            // 정지 버튼: 오른쪽 위, 상태바/카메라 구멍 아래
            float bw = 104 * dp, bh = 52 * dp, top = 56 * dp, right = w - 14 * dp;
            btn.set(right - bw, top, right, top + bh);
            c.drawRoundRect(btn, bh / 2, bh / 2, fill);
            float cy = btn.centerY();
            float sq = 7 * dp, sx = btn.left + 26 * dp;
            Paint wp = new Paint(text); wp.setStyle(Paint.Style.FILL);
            c.drawRect(sx - sq, cy - sq, sx + sq, cy + sq, wp);
            text.setTextAlign(Paint.Align.LEFT);
            c.drawText("정지", sx + 14 * dp, cy + 5.5f * dp, text);
            // 안내 칩
            String msg = "Claude 조작 중 · 터치 잠김";
            text.setTextSize(12 * dp); text.setFakeBoldText(false);
            float tw = text.measureText(msg);
            RectF cr = new RectF(14 * dp, top + 10 * dp, 14 * dp + tw + 24 * dp, top + 42 * dp);
            c.drawRoundRect(cr, 16 * dp, 16 * dp, chip);
            c.drawText(msg, cr.left + 12 * dp, cr.centerY() + 4 * dp, text);
            text.setTextSize(15 * dp); text.setFakeBoldText(true);
        }
    }
}
