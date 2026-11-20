#!/bin/bash
# 앱 소스의 엔진(WmCore.kt, WmEngine.kt)을 PC용으로 컴파일하고 시험 코드(WmTest.kt, Dbg*.kt)도 컴파일. 사용: compile_core.sh [코어폴더이름=core]
WM=${WM:-$HOME/wm}; export PATH=$WM/tools/kc/kotlinc/bin:$PATH JAVA_OPTS="-Xmx3g -Xss16m"
C=${1:-core}; SRC=$WM/comic-viewer/app/src/main/java/com/example/comicviewer
rm -rf $WM/out/$C $WM/out/test $WM/out/dbg; mkdir -p $WM/out/$C $WM/out/test $WM/out/dbg
kotlinc -nowarn -d $WM/out/$C $SRC/WmCore.kt $SRC/WmEngine.kt 2>&1 | grep -v "^warning\|Picked"
cd $WM/wmtest && kotlinc -nowarn -cp $WM/out/$C:$WM/tools/ort/ort.jar -d $WM/out/test WmTest.kt 2>&1 | grep -v "^warning\|Picked"
for f in Dbg8 Dbg9 Dbg10; do kotlinc -nowarn -cp $WM/out/$C:$WM/out/test -d $WM/out/dbg $f.kt 2>&1 | grep -v "^warning\|Picked"; done
