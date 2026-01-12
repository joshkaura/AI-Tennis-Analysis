from ultralytics import YOLO

'''

# General Models:
model = YOLO('yolo11x.pt')
model_fast = YOLO('yolo11n.pt')

fast = True

if fast == True:
    result = model_fast.predict('input_videos/input_video.mp4', save=True)
else:
    result = model.predict('input_videos/input_video.mp4', save=True)

print(result)

'''

'''
print("Boxes: ")
for box in result[0].boxes:
    print(box)
'''

'''
#trained - tennis ball

model = YOLO('models/tennis_ball_yolo11n_last.pt')

result = model.predict('input_videos/input_video.mp4', conf = 0.2, save=True)

print(result)

'''

# Track Players
model = YOLO('yolo11x.pt')

result = model.track('input_videos/input_video.mp4', device = "mps", conf = 0.2, save=True)

print(result)
