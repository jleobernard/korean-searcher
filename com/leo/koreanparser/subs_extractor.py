from videocr import get_subtitles

if __name__ == '__main__':  # This check is mandatory for Windows.
    print(get_subtitles('/opt/projetcs/ich/korean-searcher/test/data/videos/vid-000.mp4', lang='kor',
                        sim_threshold=70, conf_threshold=65))