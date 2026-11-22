# wm2 인식 모델: 10k 때와 같은 학습 방식, 학습량만 20k (2026-11-22)

10k 시범 모델(`README_10K_MODEL_20261122.md`)과 **같은 설정**(U-Net 0.95M `16,32,64,96,128`, 원래 합성기 `synth.py`, 큰 풀, bs 12, 192×192 조각)에서 **스텝만 20k**로 늘린 모델이다. 하드 네거티브(`synth2`)는 쓰지 않았다. 하드 네거티브를 쓴 추천 모델은 `RETURN_HANDOFF_20261122.md`의 34k 모델이다.

## 파일
| 파일 | 내용 |
|---|---|
| `wm2_unet_ch2_20k_synth_20261122.onnx` | 3.83MB, opset 17, SHA256 `CF074309350E141A0A5D20A1B48574C30495DB208A4FFFEAC9B0DFEABA47D89B` |
| `unet_ch2_20k_synth_20261122.pt` | `{'ch': [16,32,64,96,128], 'sd': state_dict}` |
| `eval/amt_synth_ch2_20k_r_*` | core·wild·val 평가 tsv·fail 목록(임계값 0.5, 0.7) |
| `logs/amt_synth_ch2_20k*_train.log` | 학습 로그(앞 6k, 이어서 14k) |

## 사용법 (10k 모델과 동일)
- 입력 `image` float32 `[1,3,h,w]`, RGB 0~1. 원본을 **0.5배 BOX 축소** → 아래·오른쪽 reflect 패딩으로 **h,w를 16의 배수**로. 출력 `logits` `[1,1,h,w]` → sigmoid.
- 임계값 0.5, 상자 변환은 `wm2/train/infer_eval.py`의 `boxes_from_mask`(12% 초과면 전면 무늬 1개, 6px 팽창·덩어리·30픽셀 미만 제거·좌표 ×2).
- PyTorch↔ORT 로짓 최대 차이 0.0045. 이 PC ORT CPU 약 100ms/장(512×368). 폰 CPU는 측정하지 않았다.

## 결과 (합성 벤치만, 임계값 0.5)
| 모델 | 스텝 | core 인식 | core 음성FP | wild 인식 | wild 음성FP | core ambiguous FP | val 인식 | val 음성FP |
|---|---|---|---|---|---|---|---|---|
| 10k 시범(같은 방식) | 10k | 88.4% | 7.9% | 77.5% | 14.3% | 36.7% | – | – |
| **이 모델(20k)** | 20k | **91.8%** | **6.3%** | **80.5%** | 12.5% | 43.3% | 81.3% | 0.0% (0/21) |
| (참고) 하드 네거티브 34k | 34k | 84.0% | 7.1% | 74.5% | 1.8% | 0% | 73.0% | 0.0% |

- 10k→20k로 core 인식 +3.4%p, wild +3.0%p, core 음성 오인식 −1.6%p. **인식률은 이 모델이 가장 높다.**
- 정밀도는 core 35.6%, wild 21.8%, val 21.5%로 낮다(core에서 상자 약 1,000개가 엉뚱한 곳). 앱에서는 "여러 장에서 같은 위치로 반복되는 후보만 채택" 같은 방어가 필요하다.
- 약점: **wild 음성 오인식 12.5%**, 글자 든 고정 띠(ambiguous) 오인식 43~53%. 이 둘은 하드 네거티브 모델이 크게 낮춘다(1.8%, 0%).
- 목표(§6): core 인식 ≥80% 달성, core 음성 오인식 ≤5% 미달(6.3%), wild 음성 오인식 미달(12.5%).

## 한계·주의
- **실제 웹툰 미검증**(합성 벤치만). **폰 CPU 시간 미측정.** PC 글꼴 벤치라 폰 벤치와 직접 비교 불가.
- **이어 학습 모델이다.** 같은 설정의 20k 학습이 6.6k 스텝에서 앱 재시작으로 끊겨, 6k 체크포인트부터 **14k를 이어서** 학습했다(학습률 스케줄을 새로 시작, 시작 lr 8e-4, seed 2). 끊김 없는 20k와 약간 다를 수 있다. 10k 시범 모델은 끊김 없는 별도 실행이다.
- val은 하드 네거티브 34k 모델을 최종 후보로 고른 뒤 이 모델의 참고용으로 평가했다(모델 선택에 쓰지 않음). val ambiguous 수가 적어 0%는 참고용이다.

## 재현
```
set PYTHONUTF8=1
python wm2/train/train.py --pool <pool> --steps 6000 --bs 12 --workers 4 --device dml --ch 16,32,64,96,128 --synth synth --out <ck6k>   # 앱 재시작 전 구간(실제로는 20k 스케줄의 6k까지)
python wm2/train/train.py --pool <pool> --steps 14000 --bs 12 --workers 4 --device dml --ch 16,32,64,96,128 --synth synth --lr 8e-4 --seed 2 --resume <ck6k>/unet_6000.pt --out <ck20k>
python wm2/train/eval_all.py <ck20k>/unet_14000.pt --bench <bench> --tiers core,wild,val --ths 0.5,0.7 --tag x --device dml
python wm2/train/export_onnx.py <ck20k>/unet_14000.pt out.onnx <이미지>
```
