'''
RGB(x) BRG(o)

다양한 직선 그리기
cv2.line(img, start, end, color, thickness, lineType)

img: 그림을 그릴 이미지 파일
start: 선 시작 좌표(ex; (0,0))
end: 선 종료 좌표(ex; (500. 500))
color: BGR형태의 선 색상 (ex; (255, 0, 0) -> Blue)
thickness (int): 선의 두께. pixel (default=1)
lineType: 선 그리기 형식 (cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA)
'''

import cv2
img = cv2.imread('img_file/blank_500.jpeg')

cv2.line(img, (50, 50), (150, 50), (255,0,0))   # 파란색 1픽셀 선
cv2.line(img, (200,50), (300,50), (0,255,0)) # 초록색 1픽셀 선
cv2.line(img, (350, 50), (450, 50), (0,0,255)) # 빨간색 1픽셀 선

# 하늘색(파랑+초록) 10픽셀 선
cv2.line(img, (100, 100), (400, 100), (255,255,0), 10)
# 분홍(파랑+빨강) 10픽셀 선
cv2.line(img, (100,150), (400, 150), (255,0,255), 10)
# 노랑(초록+빨강) 10픽셀 선
cv2.line(img, (100, 200), (400, 200), (0,255,255), 10)
# 회색(파랑+초록+빨강) 10픽셀 선
cv2.line(img, (100,250), (400,250), (200,200,200), 10)
# 검정 10픽셀 선
cv2.line(img, (100,300), (400,300), (0,0,0), 10)

# 4연결 선
cv2.line(img, (100, 350), (400, 400), (0,0,255), 20, cv2.LINE_4)
# 8연결 선
cv2.line(img, (100, 400), (400, 450), (0,0,255), 20, cv2.LINE_8)
# 안티에일리어싱 선
cv2.line(img, (100, 450), (400, 500), (0,0,255), 20, cv2.LINE_AA)
# 이미지 전체 대각선
# cv2.line(img, (0,0), (500,500), (0,0,255))
cv2.line(img, (0,0), (img.shape[0],img.shape[1]), (0,0,255))


cv2.imshow('lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()