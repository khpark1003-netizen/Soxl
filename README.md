# 📈 SOXL 승률 조건 역탐색 v6 (Streamlit 앱)

Colab 노트북(`주식SOXL — 승률 조건 역탐색 v6`)을 웹앱으로 옮긴 버전입니다.
PC·폰 브라우저에서 같은 주소로 접속해 쓸 수 있습니다.

## 파일 구성
| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit UI (사이드바 설정 + 6개 탭) |
| `engine.py` | 지표 계산 / 현재 상태 판정 / 9개 조건 조합 탐색 |
| `indicators.py` | RSI·MACD·ATR·ADX·볼린저·통계 함수 |
| `views.py` | HTML 카드/표 렌더링 |

## 1) 내 PC에서 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
브라우저가 자동으로 열립니다 (`http://localhost:8501`).

## 2) 폰에서도 쓰기 — Streamlit Community Cloud (무료, 추천)
1. GitHub에 새 저장소를 만들고 이 폴더 전체를 올립니다.
2. https://share.streamlit.io 에 GitHub 계정으로 로그인
3. **New app** → 저장소 선택 → Main file path: `app.py` → **Deploy**
4. 발급된 주소(`https://xxxx.streamlit.app`)를 폰에서 열기
5. **홈 화면에 추가**하면 앱처럼 실행됩니다
   - iPhone: Safari 공유 버튼 → "홈 화면에 추가"
   - Android: Chrome 메뉴 → "홈 화면에 추가"

> 저장소를 **Private**으로 두면 코드가 공개되지 않습니다.
> 앱 접근 자체를 제한하려면 Cloud 대시보드 → Settings → Sharing 에서 허용할 이메일을 지정하세요.

## 사용법
사이드바에서 기간·목표승률·최소샘플을 정하고 **탐색 실행**을 누릅니다.
(모바일은 좌상단 ☰ 를 누르면 설정이 열립니다.)

| 탭 | 내용 |
|---|---|
| 🎯 지금 진입 | 현재 9개 조건과 완전히 같았던 과거 사례 기준 승률·기대값 |
| 📍 패턴 매칭 | 황금/신뢰 조합과 현재 상태의 일치도 체크리스트 |
| 📋 탐색 결과 | 전체 조합 표 + CSV 다운로드 |
| 🏆 황금 분포 | 황금 조합들의 조건별 분포 |
| 📊 현재 상태 | RSI/MACD/BB/ATR/ADX/시장레짐 현재값 |
| 📖 용어 | 지표 설명 |

## 참고
- 데이터는 Yahoo Finance에서 받아오며 1시간 캐시됩니다. 사이드바의 **데이터 새로고침**으로 강제 갱신.
- 첫 탐색은 수십 초 걸릴 수 있고, 같은 설정으로 다시 열면 캐시로 즉시 뜹니다.
