# kakap developers의 ocr API를 가져다 쓰기 😍

1. 준비과정 :
    cv2 모듈을 임포트 하기 위해 opencv를 다운 받아야하는데 파이썬 버전이 3.8 이상이면 오류가 남. 
    python == 3.7 로 가상환경 설정할 것
2. kakao developers에서 REST API를 받아 인자로 넣어야 하므로 자세한 사항은 아래의 페이지 참고.
    https://developers.kakao.com/docs/latest/ko/vision/dev-guide#common

3. google api 
    https://cloud.google.com/vision/docs/ocr?apix_params=%7B%22resource%22%3A%7B%22requests%22%3A%5B%7B%22features%22%3A%5B%7B%22type%22%3A%22TEXT_DETECTION%22%7D%5D%2C%22image%22%3A%7B%22source%22%3A%7B%22imageUri%22%3A%22https%3A%2F%2Fpost-phinf.pstatic.net%2FMjAyMTA1MDZfMTc4%2FMDAxNjIwMjYwMzc3NzMy._KR8XqolL9Crfi3QPNJoZrnu8GhW_CvOZjNGmv2t2Icg.DujnlxE9B-SDMmE9t3YM0NlcMnbG2VhoHjrMTCZZm7Yg.JPEG%2F20210504_1548040_002.jpg%3Ftype%3Dw1200%22%7D%7D%7D%5D%7D%7D#vision_text_detection-gcloud

4. naver api

    https://clova.ai/ocr