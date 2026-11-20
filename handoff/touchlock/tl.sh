#!/system/bin/sh
# 폰(adb shell)에서 실행: 터치 잠금 + 화면 정지 버튼. 사용: tl.sh [초, 기본 600]
appops set com.example.touchlock SYSTEM_ALERT_WINDOW allow 2>/dev/null
( sleep 0.3; am start-foreground-service -n com.example.touchlock/.OverlayService >/dev/null 2>&1 ) &
/data/local/tmp/touchlock "${1:-600}"
am stopservice -n com.example.touchlock/.OverlayService >/dev/null 2>&1
