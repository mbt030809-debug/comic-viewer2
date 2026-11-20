#!/bin/bash
# 새 세션에서 한 번 실행. 사용: bash handoff/setup_tools.sh   (저장소 루트에서; 작업 폴더는 WM=~/wm)
# 하는 일: ①도구 내려받기(kotlinc, android-all, r8, ONNX) ②시험 코드·스크립트를 $WM 로 복사 ③이미지 자료(별도 zip) 풀기 안내
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}; WM=${WM:-$HOME/wm}; T=$WM/tools
mkdir -p $T/ort/jni $T/kc $WM/out $WM/nn $WM/data
ln -sfn $REPO $WM/comic-viewer
cp -r $REPO/handoff/scripts $WM/; cp -r $REPO/handoff/wmtest $WM/; cp $REPO/handoff/reply_to_device_session_20261120.txt $WM/
cp $REPO/app/src/main/assets/wm_net.onnx $WM/nn/wm_net_v3.onnx
cd $T
echo "[1/4] kotlinc 1.9.24"
[ -d kc/kotlinc ] || { curl -sL -o kc.zip https://github.com/JetBrains/kotlin/releases/download/v1.9.24/kotlin-compiler-1.9.24.zip && unzip -q kc.zip -d kc && rm kc.zip; }
echo "[2/4] android-all(안드로이드 API 14 클래스)"
[ -f android-all-14.jar ] || curl -sL -o android-all-14.jar https://repo1.maven.org/maven2/org/robolectric/android-all/14-robolectric-10818077/android-all-14-robolectric-10818077.jar
echo "[3/4] r8(d8)"
[ -f r8.jar ] || curl -sL -o r8.jar https://dl.google.com/android/maven2/com/android/tools/r8/8.1.72/r8-8.1.72.jar
echo "[4/4] ONNX 런타임: 안드로이드용(aar → classes.jar + jni) / 데스크톱용(ort.jar)"
if [ ! -f ort/classes.jar ]; then
  for i in 1 2 3 4 5; do curl -sL -o ort.aar https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.17.3/onnxruntime-android-1.17.3.aar && unzip -tq ort.aar >/dev/null 2>&1 && break; sleep 20; done
  mkdir -p ort_aar && unzip -q -o ort.aar -d ort_aar && cp ort_aar/classes.jar ort/classes.jar && cp -r ort_aar/jni/arm64-v8a ort_aar/jni/armeabi-v7a ort/jni/ && rm -rf ort_aar ort.aar
fi
[ -f ort/ort.jar ] || curl -sL -o ort/ort.jar https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.17.3/onnxruntime-1.17.3.jar
which aapt2 zipalign apksigner java javac >/dev/null || echo "경고: aapt2/zipalign/apksigner/java 가 PATH 에 없음(APK 빌드에 필요)"
[ -d $WM/wmtest/assets ] || echo "알림: 이미지 자료 zip(handoff_data_20261120.zip)을 사용자에게 받아 'unzip handoff_data_20261120.zip -d $WM' 로 풀어야 합니다(wmtest/assets, data/)."
echo "완료. 다음: handoff/README_START_HERE.md 의 '시험 실행' 참고"
