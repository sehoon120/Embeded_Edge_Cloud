import cv2
import os

input_folder = 'img'
output_folder = 'test_face_2'
target_size = (480, 640)

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 이미지 순회
for fname in os.listdir('C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\ec2_server\img'):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join('C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\ec2_server', input_folder, fname)
        output_path = os.path.join('C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\ec2_server', output_folder, fname)
        print(input_path)

        # 이미지 로드
        img = cv2.imread(input_path)
        if img is None:
            print(f"[WARNING] 이미지 로드 실패: {input_path}")
            continue

        # 리사이즈
        resized_img = cv2.resize(img, target_size)

        # 저장
        cv2.imwrite(output_path, resized_img)
        print(f"[INFO] 저장 완료: {output_path}")

print("[DONE] 모든 이미지 리사이즈 완료!")
