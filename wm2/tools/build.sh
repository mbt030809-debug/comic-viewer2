#!/data/data/com.termux/files/usr/bin/bash
# wm2 엔진 + 시험 하니스 컴파일 (이 폰의 kotlinc). 결과: ~/wm2-work/out
set -e
export PATH=$HOME/wm/tools/kc/kotlinc/bin:$PATH JAVA_OPTS="-Xmx2g -Xss16m"
R=$HOME/cv2-repo; O=$HOME/wm2-work/out
mkdir -p $O
kotlinc -nowarn -d $O $R/app/src/main/java/com/example/comicviewer/wm2/*.kt $R/wm2/test/Wm2Bench.kt 2>&1 | grep -v "^warning\|Picked" || true
