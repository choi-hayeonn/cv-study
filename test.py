# 실습. 웹캠을 통해 들어오는 영상을 이용해 움직임이 감지될 경우 해당 프레임을 캡쳐해서 파일로 저장하는 프로그램 만들기(강사님의 파이썬을 이용한 풀이)
import cv2
import time
import os

# 폴더생성
SAVE_DIR = "captures"
# 최소영역
MIN_AREA = 1200
# 저장 간격(초단위)
COOLDOWN = 1.0

# 폴더가 없으면 생성
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠 오류")

# 배경제거
# 웹캠이나 cctv영상처럼 움직이는 물건(전경)과 고정된 배경을 자동으로 구분할때 사용
# createBackgroundSubtractorMOG2
# history: 기본값 500, 값이 크면 배경모델이 천천히 변함(안정적), 작으면 빠르게 변함(민감)
# varThreshold : 기본값 16, 픽셀이 전경인지 배경인지 판단
# - 값이 작으면 작은변화에도 움직임으로 판단
# - 값이 크면 큰 변화가 있어야 움직으로 판단
# detectShadows: 기본값 True, 그림자도 검출할지 여부
backsub = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=25, detectShadows=True
)

last_saved = 0.0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1번
    # 전경 마스크 받아오기
    fg = backsub.apply(frame)  # 현재 프레임에서 움직임 부분만 추출
    # 그림자 제거: 200이상 부분만 남기고 나머지는 0으로 처리
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    # 움직임 영역을 키워서 구멍 메우기
    fg = cv2.dilate(fg, None, iterations=2)

    # 2번
    # 움직임 영역 찾기
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion = False  # 움직임이 있었는지 표시하는 변수

    for c in contours:
        # 너무 작은 영역 무시
        if cv2.contourArea(c) < MIN_AREA:
            continue

        # 움직임이 있는 부분에 사각형 표시
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion = True

    # 화면상단에 "Motion:ON/OFF" 글자 표시
    cv2.putText(
        frame,
        f"Motion: {'ON' if motion else 'OFF'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if motion else (0, 0, 255),
        2,
    )

    # 원본 영상과 마스크 영상 출력
    cv2.imshow("frame", frame)
    cv2.imshow("mask", fg)

    # 3. 움직인 사진 저장
    now = time.time()
    if motion and (now - last_saved > COOLDOWN):
        filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
        # 현재 시각으로 파일명 생성
        cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)  # 현재 프레임 저장
        last_saved = now

    # 4. 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 카메라, 창 종료
cap.release()
cv2.destroyAllWindows()





