# 흉부 엑스레이 예시 사진의 출처

모두 Wikimedia Commons에 공개된 사진이며, 실습에 맞게 가로 960픽셀 이하로 줄인 것입니다. 실제 환자 사진이나 병원 자료는 들어 있지 않습니다. `cases.json`의 한글 요약은 각 사진의 원문 설명(사람이 쓴 것)을 옮긴 것입니다.

| 파일 | 내용 | 저작자 | 라이선스 | 출처 |
|---|---|---|---|---|
| `normal_pa.jpg` | 정상 | Mikael Häggström | CC0 | https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg |
| `normal_2.jpg` | 정상 | Doctoroftcm | CC0 | https://commons.wikimedia.org/wiki/File:Chest_X-ray_2346.jpg |
| `lobar_pneumonia.jpg` | 폐렴 (대엽성). 화살표가 없는 원본 (2018-11-07에 올라온 판) | Mikael Häggström | CC0 | https://commons.wikimedia.org/wiki/File:X-ray_of_lobar_pneumonia.jpg |
| `marked/lobar_pneumonia.jpg` | 같은 사진에 병변 위치를 빨간 화살표로 표시한 것 (2026-11-11에 Doc James가 올린 판) | Mikael Häggström, Doc James | CC0 | 같은 쪽의 파일 기록 |
| `pneumonia_2.jpg` | 폐렴 (다발성) | Mikael Häggström | CC0 | https://commons.wikimedia.org/wiki/File:Chest_radiograph_in_influensa_and_H_influenzae,_posteroanterior.jpg |
| `pneumothorax.png` | 기흉 | Alex Bonilla 외 | CC BY 4.0 | https://commons.wikimedia.org/wiki/File:Chest_X-ray_of_pneumothorax.png |
| `heart_failure.jpg` | 심부전 | Mikael Häggström | CC0 | https://commons.wikimedia.org/wiki/File:Chest_radiograph_of_a_lung_with_Kerley_B_lines.jpg |
| `pleural_effusion.jpg` | 흉수 | InvictaHOG | Public domain | https://commons.wikimedia.org/wiki/File:Pleural_effusion.jpg |
| `tuberculosis.jpg` | 폐결핵 | 미상 (미국 CDC 자료) | Public domain | https://commons.wikimedia.org/wiki/File:Tuberculosis-x-ray-1.jpg |
| `rib_fracture.jpg` | 늑골 골절 | Doctoroftcm | CC0 | https://commons.wikimedia.org/wiki/File:Rib_fracture_2347.jpg |

**화살표가 있는 사진과 없는 사진.** 모델에게 주는 사진에 화살표가 있으면 병변 위치를 알려 주는 힌트가 됩니다. 그래서 `lobar_pneumonia.jpg`는 화살표가 없는 원본을 모델에게 주고, 화살표가 있는 판은 `marked/` 폴더에 두어 사람에게 설명할 때만 보여 줍니다. `pneumothorax.png`, `pleural_effusion.jpg`, `tuberculosis.jpg`는 원본에 화살표나 표시가 들어 있는 사진이고, 표시 없는 판을 구하지 못해 그대로 씁니다. 이 사진들에서 모델이 위치를 맞히면 표시를 읽었을 수 있다는 점을 감안하세요.

`pneumothorax.png`는 CC BY 4.0(https://creativecommons.org/licenses/by/4.0)에 따라 저작자를 표시하고 크기만 줄여 씁니다.

더 많은 사진으로 해 보려면 공개 데이터셋을 쓰세요: Kaggle "Chest X-Ray Images (Pneumonia)", NIH ChestX-ray14.
