import torch
import cv2
import os
from datetime import datetime
from gtts import gTTS
import pygame

# YOLOv5 모델 로드 (사용자 학습 모델 경로 직접 지정)
model_path = r"C:\yolov5-master\runs\train\exp12\weights\best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # 사용자 학습 모델 경로

# 임계점 설정
model.conf = 0.1  # Confidence Threshold를 0.1로 설정

# pygame 초기화
pygame.mixer.init()

# 감지 상태 변수
thumbs_up_detected = False


# Thumbs up 감지 후 음성 출력 함수
def play_audio():
    current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")  # 현재 시간 포맷
    print(f"Thumbs up detected! Current time: {current_time}")

    # 음성 합성
    tts = gTTS(f"현재 시간은 {current_time}입니다.", lang='ko')

    # 경로를 시스템 디렉토리로 저장
    system_audio_path = 'C:\\Temp\\current_time.mp3'
    if not os.path.exists('C:\\Temp'):
        os.makedirs('C:\\Temp')  # 디렉토리 생성

    # 음성 파일 저장 및 재생
    try:
        tts.save(system_audio_path)
        print("Audio saved successfully to:", system_audio_path)

        # pygame을 통해 음성 재생
        pygame.mixer.music.load(system_audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # 음악 재생이 끝날 때까지 대기
            pass
    except Exception as e:
        print("Audio save or playback error:", e)


# 웹캠 데이터 가져오기
cap = cv2.VideoCapture(0)  # 웹캠 초기화

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임 읽기 실패")
        break

    # YOLOv5 모델로 감지
    results = model(frame)

    # 감지 결과 렌더링
    results.render()

    # Thumbs up 감지 여부 확인
    if not thumbs_up_detected and results.pred[0].shape[0] > 0:
        for det in results.pred[0]:
            class_id = int(det[5])  # 클래스 ID 확인
            if model.names[class_id] == "thumbs up":  # "thumbs up" 클래스인지 확인
                thumbs_up_detected = True  # 상태 변경
                play_audio()  # 음성 출력
                break  # 한 번 감지하면 루프 중단

    # 화면 출력
    cv2.imshow("YOLOv5 Webcam", frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 코드
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
