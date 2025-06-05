import requests

# EC2 퍼블릭 IP를 입력하세요
url = "http://3.34.202.223:5000/inference"
# r"C:\Users\happy\Downloads\jy.jpg"
# 전송할 이미지 경로
image_path = r"C:\Users\happy\Downloads\jy.jpg"
# r"C:\Users\happy\Downloads\sehoon.jpg"
# r"C:\Users\happy\Downloads\jy.jpg"


files = {'image': open(image_path, 'rb')}
response = requests.post(url, files=files)

# 결과 출력
print(response.status_code)
print(response.json())
