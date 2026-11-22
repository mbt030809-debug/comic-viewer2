#!/data/data/com.termux/files/usr/bin/bash
# 앱과 같은 Kotlin 경로(Wm2Detector/Wm2Engine + 기존 WmCodec 인터페이스)를 JVM 용으로 컴파일. 결과: ~/wm2-work/out_app
set -e
export PATH=$HOME/wm/tools/kc/kotlinc/bin:$PATH JAVA_OPTS="-Xmx2g -Xss16m"
R=$HOME/cv2-repo; A=$R/app/src/main/java/com/example/comicviewer; O=$HOME/wm2-work/out_app; ORT=$HOME/wm/tools/ort/ort.jar
mkdir -p $O
kotlinc -nowarn -cp $ORT -d $O $A/WmTypes.kt $A/wm2/Wm2Detector.kt $A/wm2/Wm2Gallery.kt $A/wm2/Wm2Confirmer.kt $A/wm2/Wm2Eraser.kt $A/wm2/Wm2Pipeline.kt $R/wm2/test/JvmCodec.kt $R/wm2/test/Wm2AppBench.kt $R/wm2/test/Wm2EraseBench.kt 2>&1 | grep -v "^warning\|Picked" || true
