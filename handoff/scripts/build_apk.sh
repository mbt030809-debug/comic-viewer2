#!/bin/bash
# 로컬 APK 빌드(안드로이드 SDK 없이): aapt2 + kotlinc + d8 + zipalign + apksigner
# 사용: build_apk.sh <프로젝트 폴더> <versionCode> <versionName> <출력 apk>
set -e
PRJ=$1; VC=$2; VN=$3; OUTAPK=$4
WM=${WM:-$HOME/wm}
T=$WM/tools; W=$WM/work; rm -rf $W; mkdir -p $W/gen $W/classes $W/dex
ANDROID=$T/android-all-14.jar; KC=$T/kc/kotlinc
export PATH=$KC/bin:$PATH
# 1) 매니페스트: package/version 주입, debuggable (AGP 디버그 빌드와 동일)
python3 - "$PRJ" "$VC" "$VN" "$W/AndroidManifest.xml" <<'PY'
import sys,re
prj,vc,vn,out=sys.argv[1:5]
m=open(prj+'/app/src/main/AndroidManifest.xml',encoding='utf-8').read()
m=m.replace('<manifest xmlns:android="http://schemas.android.com/apk/res/android">','<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.example.comicviewer" android:versionCode="%s" android:versionName="%s">'%(vc,vn),1)
m=m.replace('<application\n','<application\n        android:debuggable="true"\n',1)
m=m.replace('<application ','<application android:debuggable="true" ',1) if 'debuggable' not in m else m
m=m.replace('<uses-permission android:name="android.permission.INTERNET" />','<uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34" />\n    <uses-permission android:name="android.permission.INTERNET" />',1)
open(out,'w',encoding='utf-8').write(m)
PY
# 2) 리소스
aapt2 compile --dir $PRJ/app/src/main/res -o $W/res.zip
[ -d $PRJ/app/src/main/assets ] && AST="-A $PRJ/app/src/main/assets" || AST=""
aapt2 link $AST -I $ANDROID --manifest $W/AndroidManifest.xml --min-sdk-version 24 --target-sdk-version 34 \
  --java $W/gen --auto-add-overlay -o $W/base.apk $W/res.zip
# 3) R.java → class
mkdir -p $W/rcls && javac -source 8 -target 8 -d $W/rcls $(find $W/gen -name R.java) 2>&1 | grep -v "warning" || true
# 4) 코틀린 컴파일 (람다/SAM 은 클래스로: 구형 dex 변환 호환)
kotlinc -jvm-target 1.8 -Xlambdas=class -Xsam-conversions=class -Xstring-concat=inline -nowarn \
  -cp "$ANDROID:$W/rcls:$T/ort/classes.jar" -d $W/classes $PRJ/app/src/main/java/com/example/comicviewer/*.kt
# 5) dex
(cd $W/classes && zip -qr ../app.jar .); (cd $W/rcls && zip -qr ../r.jar .)
java -cp $T/r8.jar com.android.tools.r8.D8 --release --min-api 24 --lib $ANDROID --output $W/dex \
  $W/app.jar $W/r.jar $T/ort/classes.jar $KC/lib/kotlin-stdlib.jar 2>&1 | grep -v "^Warning" || true
# 6) 합치기 + 정렬 + 서명
cp $W/base.apk $W/unsigned.apk
(cd $W/dex && zip -q -j ../unsigned.apk classes*.dex)
rm -rf $W/libs; mkdir -p $W/libs/lib
cp -r $T/ort/jni/arm64-v8a $T/ort/jni/armeabi-v7a $W/libs/lib/
if [ -d $PRJ/app/src/main/jniLibs ]; then cp -r $PRJ/app/src/main/jniLibs/. $W/libs/lib/; fi
(cd $W/libs && zip -q -r $W/unsigned.apk lib)
unzip -l $W/unsigned.apk | grep -q "lib/arm64-v8a/libonnxruntime4j_jni.so" || { echo "오류: ORT 네이티브 라이브러리가 APK에 없음"; exit 1; }
zipalign -p -f 4 $W/unsigned.apk $W/aligned.apk
apksigner sign --ks $PRJ/app/debug.keystore --ks-pass pass:android --key-pass pass:android --ks-key-alias androiddebugkey --out $OUTAPK $W/aligned.apk
apksigner verify --verbose $OUTAPK | head -6
aapt2 dump badging $OUTAPK | grep -E "^package|application-label:|launchable|uses-permission|sdkVersion" | head -8
ls -la $OUTAPK
