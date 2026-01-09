from ultralytics import YOLO

model = YOLO('yolo11x.pt')
model_fast = YOLO('yolo11n.pt')

fast = False

if fast == True:
    result = model_fast.predict('input_videos/input_video.mp4', save=True)
else:
    result = model.predict('input_videos/input_video.mp4', save=True)

print(result)

'''
print("Boxes: ")
for box in result[0].boxes:
    print(box)
'''
