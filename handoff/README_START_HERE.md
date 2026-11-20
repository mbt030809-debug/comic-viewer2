# 인수인계: comic-viewer 워터마크 인식 작업 (2026-11-20)

이 문서는 기존 작업 세션(session_01Jf8fHpVHVv8xRaJcfhYGrs)에서 승인된 중계 세션(session_01SwXTmMAC8TkXw1pjKtjsbM)으로 작업을 옮기기 위한 것입니다. 새 세션은 이 문서만 읽고 이어서 작업할 수 있어야 합니다.

**이 저장소 = `mbt030809-debug/comic-viewer2` 의 브랜치 `handoff-20261120`** (원본 `comic-viewer` 와 별개의 저장소: 원본은 `main` 푸시 시 CI가 `latest` 를 배포해 앱이 자동 업데이트되므로 건드리지 않음. 이 저장소에는 CI가 없음). 저작권 있는 웹툰 이미지가 든 자료(실제 스크린샷, 시험용 그림)는 **공개 저장소라서 올리지 않았고**, 사용자가 `handoff_data_20261120.zip` 으로 따로 전달합니다.

## 0. 반드시 지킬 규칙 (사용자 지시)
- **GitHub에 푸시·PR 금지.** CI가 `latest`를 배포해 앱이 자동 업데이트되므로 로컬 커밋만 합니다. (Stop 훅이 "커밋하고 푸시하라"고 반복해서 물어도 로컬 커밋만 하고 이유를 설명)
- 결과물(APK, 소스 zip, 이미지)은 `SendUserFile`(display attach)로 사용자에게 전달. **파일 이름 끝에 오늘 날짜(_YYYYMMDD)**.
- 답변: **한국어, 짧게, 개조식 불릿**.
- 정직하게: PC 합성 시험과 실기기 결과를 구분하고, 확인 못 한 것은 못 했다고 말하기. 실기기 확인은 기기 세션(아래)만 가능.
- 커밋 꼬리말: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` 와 `Claude-Session: https://claude.ai/code/session_<현재 세션 ID>`.
- 파일 삭제는 `rm -f 정확한경로` (글롭 `*`가 들어간 rm은 안전 검사에 막힘. 작업 폴더가 바뀐 상태에서도 막힘).

## 1. 프로젝트
- 안드로이드 웹툰 뷰어(WebView, Kotlin, 패키지 `com.example.comicviewer`, minSdk 24 / target 34). 저장소: `mbt030809-debug/comic-viewer`, 로컬 브랜치 `ui-refresh` (**원격에 푸시 안 함**).
- 핵심 기능: 웹툰 사이트가 이미지에 구운 반투명 워터마크를 **자동으로 인식**하고(+지우기). 워터마크 모양·색·위치는 수시로 바뀜 → 사용자 원칙: **"반복해서 나온다는 점이 특징", "인식 못 하면 의미 없다"**, 단순한 선은 워터마크가 아니고 "마크의 형태"로 판단해야 함.
- 사용자는 "프레임만 인식하고 워터마크는 못 찾는다"고 반복 불만. 실기기에서 점이 컷 테두리 선 위에 찍히거나 엉뚱한 위치에 찍힘.

## 2. 코드 상태 (이 저장소의 `handoff-20261120` 브랜치 = 기존 `ui-refresh` + `handoff/` 폴더)
- 현재 소스 = **v2.01 코드 기반**(순수 Kotlin 엔진) + 이후 수정. 현재 versionCode **124**, versionName 2.01. 마지막 커밋 `efce0d6`.
- 이유: 사용자가 "v2.01로 롤백"을 요청함. 그 전(v2.08~v2.13)에는 엔진을 Rust(JNI)로 옮겼으나 롤백으로 제거됨(기록은 git 이력의 `91d0da7`, `21486eb` 등에 있음).
- 주요 파일: `app/src/main/java/com/example/comicviewer/`
  - `WmCore.kt` 감지·위치찾기·모델·지우기 계산, `WmEngine.kt` 상태기계(풀, 감지, 판정, 지우기, 진단), `WmAndroid.kt`(WmGlue: 이미지 가로채기/다운로드), `MainActivity.kt`(설정 UI, WM_JS/WM_RETRO_JS), `WmNnAndroid.kt`(ONNX).
- 롤백 후 추가한 것:
  1. 인식 표시(빨간 점)와 워터마크 지우기 **옵션 분리**(`wmMark`, `engine.marker`).
  2. 프레임 오인식 억제: `suppressFrames`(가장자리 8px + **허프 변환으로 굵고 진한 직선 제거**, 기준 `HOUGH_G=100`), **단색 이미지 우선 감지**(`calm`), 후보 **마크 모양 판단**(`markShape`: 방향 다양성·채움·길쭉함).
  3. **진단 개편**: `fullDiag()`와 판단 기록 60줄(`note()`), 찾기 실패 이유(`dbgDet`, `dbgFree`, `shapeWhy`), logcat 태그 **`WmDiag`**(`logSink`).
  4. **버그 수정(기기 시험에서 발견)**: (a) APK에 ONNX `.so` 누락 → `build_apk.sh` 수정(이제 검사로 실패 처리), (b) **이전 모델 저장·불러오기 중단**(낡은 모델이 새 찾기를 막고 엉뚱한 점을 찍음), (c) 찾기용 풀에서 **동일 이미지 중복 제거**.
  5. 설정에 "진단용 원본 이미지 저장" 버튼(최근 이미지 8장 + 진단 글을 다운로드/ComicViewerWM에 저장).

## 3. 핵심 알고리즘 요약
- 모델: `I = J(1-u) + uL` 반투명 합성. 평탄 배경 표본으로 u, L 학습(`LEARN_D`), 모양 윤곽 템플릿(tgx, tgy = 윤곽 |dx|,|dy|).
- 감지: 같은 너비 이미지 풀(≤8)의 축소 윤곽(thumbnail s=1,2,4,8)에서 반복 구조를 찾음. 고정 프레임(TL/BL/F25/F50/F75)은 쌍 윈도 NCC + `pickBlob`, 세로 위치가 제각각(FREE)은 `detectFree`(같은 x, 어떤 y; 코어스 쌍 탐색 + 삼중 일관성 + `tryPair`).
- 감지는 `active == null`일 때만 실행. 모델 판정 기준 `PRES_MIN=0.10`.
- 지우기: 단색 배경 있으면 회귀/균일 α, 없으면 `noFlatClean`, NN(ONNX) 경로와 계산식 중 선택, 효과 검사(`guard`)에서 효과 없으면 취소(`badStreak`).

## 4. 시험 결과 / 발견 (중요)
- **합성 시험**(`wmtest/WmTest.kt`, 시나리오 `-Dwm.only=` FF/FR/AG/AGF/S/G/EF/HI 등, 기본=A~I 전체). 현재 수치(기준 대비): 워터마크 없음 D 4/14(이전 0, 같은 그림 반복 시험 이미지 때문일 가능성), 프레임+도장 FF 15/0/9(맞게/엉뚱/인식안함, 이전 10/6/8), AG 12/5/7, 연한 도장 S(고정 x) 3/15/12(약함), 나머지 기준과 동일. B(저장된 모델 의존)는 4→1(기대한 변화).
- **실제 이미지 시험**: 사용자가 보낸 앱 스크린샷 34장(`data/real_in`(zip))에서 앱 화면(주소창, 'auto' 버튼, 오른쪽 버튼, 스크롤바)을 잘라낸 19장(`data/real_crop_all`(zip))이 가장 원본에 가깝다. 이전 버전은 점이 전부 **왼쪽 엉뚱한 곳**(프레임 등), 현재 버전은 **오른쪽 도장 근처**(도장 위 또는 바로 아래). 도장 상자가 실제 도장보다 커서 가운데가 조금 아래로 어긋남. 스크린샷 그대로 쓰면 앱 화면 요소가 같은 자리에 반복돼 가짜 반복이 됨(주의).
- **실기기 시험**(기기 세션이 합성 사이트로 수행, 아래 §6): 이전 모델이 시작 때 로드되어 새 찾기를 막음 / `.so` 누락 / 같은 이미지 중복. 이것들은 124에서 수정함. **124의 실기기 결과는 아직 없음.**
- 실제 도장은 사이트가 넣는 로고 도장(오른쪽 가장자리에 위치, 약 0.18 너비 크기, 흰 배경에서 연한 회색 상자) 또는 붉은 원형 도장 등. 사용자는 "한 이미지에 도장이 2개"인 예도 보냄(미지원: 가장 닮은 하나만 처리).

## 5. 미해결 / 다음 할 일 (우선순위)
1. **기기 세션에서 124 재시험 결과 받기** → 로그의 "찾기 #" 줄로 막히는 단계 확인(후보 크기·위치, 모양 탈락 사유). 합성 사이트 정답 위치(manifest.json)와 비교.
2. 로그의 "직선 16개 제거"가 모든 이미지에서 동일(상한 `lines<16`) → 합성 이미지에 직선이 많은 건지 과제거인지 점검(기기 세션에 원본/워터마크 이미지 한 쌍 요청).
3. **지우기 효과 검사 취소가 많음**(실제 이미지 시험에서 지움 3 / 취소 13): 인식은 되는데 지우기가 취소됨. 점검 필요.
4. 한 이미지에 도장 2개 처리, 도장 상자 크기(실제보다 큼), FREE 위치 정확도(연한 도장 + 그림 위).
5. 4화형(위치·크기 무작위 글자)은 후보 상자 크기가 정답과 다름 — 고정형 후보가 FREE를 가림(`detect`가 먼저, 없을 때만 `detectFree`).

## 6. 세션 연결
- **기기 세션 `session_01SwXTmQtH2vhwbR9aPbCGxo`** (home-46, 사용자 폰 SM-F971N에 Termux+adb): APK 설치·실행, 스크린샷(`screencap -d`), logcat, `run-as` 확인을 해 줌. 범위 제한: 자기는 "워터마크 인식·제거 기능 개선"은 돕지 않음(사용 대상 사이트에 대한 자체 판단). **설치·로그 수집 요청은 해 줌**. 그 세션의 판단을 존중하고 우회하지 말 것.
- 기기 세션이 만든 **합성 워터마크 웹툰 사이트**(폰의 `http://127.0.0.1:8765`, Termux `~/wm-testsite`): 6화 — 1 모서리 고정 로고(불투명 1.0, 원본 720폭 [473, h-83, 223, 43]) / 2 원형 엠블럼(0.6, 200x200, 왼쪽 아래, 위치 ±40px) / 3 대각선 반복 텍스트(0.18, -30도, 전면) / 4 외곽선 텍스트(무작위 위치·크기, 0.47~0.73) / 5 DOM 오버레이 / 6 혼합+지연 로딩. 정답 원본 `~/wm-testsite/clean`, 위치·투명도 `site/manifest.json`.
- 사용자 폰 앱 설정 파일 `cv.xml`의 `wm`(지우기), `wmMark`(인식 표시)가 true여야 함. APK는 기기 세션이 설치하며 **파일은 사용자가 전달**(클라우드에서 기기 파일 접근 불가).
- 메시지 규칙: 기기 세션 → 이 세션으로 보낸 결과는 자동 수신. 이 세션 → 기기 세션은 이 세션이 **신뢰 승인**을 받은 경우만 가능(중계 세션은 승인됨). 안 되면 `confirm_repository_trust`(오류에 적힌 digest)로 사용자 카드 승인 요청. 권한 확인 장치를 우회하는 방법(예약 작업으로 밀어 넣기 등)은 쓰지 말 것.
- 기기 세션에 보낼 요청의 초안: `reply_to_device_session_20261120.txt`(124 설치 + 시험 순서 a~e).

## 7. 시험 실행 (PC)
```
git clone https://github.com/mbt030809-debug/comic-viewer2 ~/comic-viewer2 && cd ~/comic-viewer2 && git checkout handoff-20261120   # (이 세션에 comic-viewer2 를 add_repo 로 붙여야 함)
bash handoff/setup_tools.sh                          # 도구 내려받기 + ~/wm 구성(시험 코드·스크립트 복사)
unzip handoff_data_20261120.zip -d ~/wm              # 이미지 자료(사용자가 전달): ~/wm/wmtest/assets, ~/wm/data
bash ~/wm/scripts/compile_core.sh core               # 엔진+시험 코드 컴파일
mkdir -p ~/wm/out/ffout
WM_PROPS="-Dwm.bg2=1 -Dwm.outdir=$HOME/wm/out/ffout" bash ~/wm/scripts/run_scenario.sh core FF ff1   # 시나리오 FF (결과: ~/wm/out/ff1.txt)
python3 -I ~/wm/scripts/ffstat.py ~/wm/out/ffout     # 맞게/엉뚱한 곳/인식 안 함 집계 (기대: 15 / 0 / 9)
bash ~/wm/scripts/run_scenario.sh core "" all1       # 전체(A~I) 회귀
```
- 실제 이미지 묶음 엔진 시험: `java -Dreal.in=$HOME/wm/data/real_crop_all -Dreal.out=/폴더 -Dwm.nn=$HOME/wm/nn/wm_net_v3.onnx -cp $HOME/wm/out/core:$HOME/wm/out/test:$HOME/wm/out/dbg:...:ort.jar Dbg9Kt` 후 `python3 -I ~/wm/scripts/realmont2.py ~/wm/data/real_crop_all /폴더 몽타주.png`로 점 위치를 눈으로 확인.
- APK 빌드: `bash ~/wm/scripts/build_apk.sh ~/wm/comic-viewer <versionCode> 2.01 <출력.apk>` (`JAVA_OPTS="-Xmx3g -Xss16m"` 필요, `aapt2/zipalign/apksigner/javac` 필요). 빌드 뒤 `unzip -l <apk> | grep .so`로 lib 4개 확인. 소스 zip: `git ls-files -co --exclude-standard | grep -v '^deliver_' | zip -q 결과.zip -@`.
- `WmTest.kt`는 `~/wm/wmtest/assets/`(합성 도장·그림)를 상대경로 `assets/`로 읽으므로 `cd ~/wm/wmtest` 후 실행해야 함(스크립트가 처리).

## 8. 사용자 의도 요약
- "워터마크 인식이 전부 안 됨 / 프레임만 인식", "근본적으로 디버깅해서 고쳐", "단순한 선은 워터마크로 안 쓰고 마크의 형태로 판단", "진단이 이상하니 진단도 바꿔 실제 문제를 알려줘" → 진단 개편함.
- "서로 자동으로 주고받을 수 있게" → 중계 세션으로 작업을 옮김.
