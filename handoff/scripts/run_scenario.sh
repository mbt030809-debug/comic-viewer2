#!/bin/bash
# 사용: run_scenario.sh <코어폴더이름> <시나리오(빈칸=전체)> <결과이름>   예: run_scenario.sh core FF ff1
# 결과 텍스트: $WM/out/<결과이름>.txt   (WM_PROPS 환경변수로 -Dwm.bg2=1 -Dwm.outdir=/폴더 등 추가)
WM=${WM:-$HOME/wm}; export PATH=$WM/tools/kc/kotlinc/bin:$PATH
KL=$WM/tools/kc/kotlinc/lib/kotlin-stdlib.jar
cd $WM/wmtest
timeout 1500 java -Dwm.sleep=${WM_SLEEP:-1500} -Dwm.nn=$WM/nn/wm_net_v3.onnx ${2:+-Dwm.only=$2} ${WM_PROPS} -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Xmx2g -cp $WM/out/$1:$WM/out/test:$KL:$WM/tools/ort/ort.jar WmTestKt 2>&1 | grep -v Picked > $WM/out/$3.txt
