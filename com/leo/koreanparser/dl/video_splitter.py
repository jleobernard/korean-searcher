import cv2

# Opens the Video file
out_dir = '/opt/projetcs/ich/korean-searcher/com/leo/koreanparser/dl/data/unsorted'
cap= cv2.VideoCapture('/opt/projetcs/ich/korean-searcher/com/leo/koreanparser/dl/data/videos/vid-001.webm')
NB_FRAMES_SKIPPED = 30
i=0
while(cap.isOpened()):
    for j in range(0, NB_FRAMES_SKIPPED):
        ret, frame = cap.read()
        if ret == False:
            exit(0)
    cv2.imwrite(out_dir + '/vid-001-'+str(i)+'.jpg', frame)
    i += 1
    if i % 10 == 0:
        print(f"Exported {i} frames")
cap.release()
cv2.destroyAllWindows()
