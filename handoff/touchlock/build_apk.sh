#!/usr/bin/env bash
# Termux에서 오버레이 APK 빌드. 필요: pkg install openjdk-17 aapt2 apksigner d8
# 컴파일용: ~/tl-build/lib/android.jar (maven com.google.android:android:4.1.1.4), framework-res.apk(폰에서 adb pull)
set -e
D=$(cd "$(dirname "$0")" && pwd); B=~/tl-build; W=$B/work; rm -rf $W; mkdir -p $W/cls $W/dex
aapt2 link -I $B/lib/framework-res.apk --manifest $D/app/AndroidManifest.xml -o $W/base.apk
javac -nowarn -source 8 -target 8 -cp $B/lib/android.jar -d $W/cls $(find $D/app/src -name '*.java') 2>&1 | grep -v "^warning\|^Note" || true
d8 --release --min-api 26 --output $W/dex $(find $W/cls -name '*.class')
cp $W/base.apk $W/unsigned.apk; (cd $W/dex && zip -q -j ../unsigned.apk classes.dex)
zipalign -p -f 4 $W/unsigned.apk $W/aligned.apk
apksigner sign --ks $D/../../app/debug.keystore --ks-pass pass:android --key-pass pass:android --ks-key-alias androiddebugkey --out $D/touchlock-overlay.apk $W/aligned.apk
ls -la $D/touchlock-overlay.apk
