# data

[AI Hub 물류공간 예측 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71861)를 받아서 이 폴더에 아래 구조로 둡니다.
데이터셋은 재배포가 불가해서 git에 올라가지 않습니다 (이 README만 올라감).

```
data/
├─ Sample/
│  ├─ 01.원천데이터/01_입고물품/<대분류>/<중분류>/<상품ID>/<상품ID>_<shot>_<cam>.jpg
│  └─ 02.라벨링데이터/01_입고물품/<대분류>/<중분류>/<상품ID>/<상품ID>_<shot>_<cam>.json
└─ Validation/
   ├─ 01.원천데이터/VS.zip        # 압축 풀지 않고 그대로
   └─ 02.라벨링데이터/VL.zip      # 압축 풀지 않고 그대로
```

- 다운로드 받은 폴더를 옮기기 싫으면 바로가기(심볼릭 링크)로 둬도 됩니다.
  ```bash
  ln -s "/path/to/Sample" data/Sample
  ln -s "/path/to/47.물류공간 예측 데이터/3.개방데이터/1.데이터/Validation" data/Validation
  ```
- 다른 위치를 쓰려면 `DATA_DIR` 환경변수로 지정합니다.
- 출고물품(`02_출고물품`)은 사용하지 않습니다.
