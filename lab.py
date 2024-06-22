import cv2
import numpy as np

# Tạo một ảnh trắng để làm nền
width, height = 400, 200
background_color = (255, 255, 255)  # Màu trắng
background = np.ones((height, width, 3), dtype=np.uint8) * 255

# Tạo thông tin về văn bản
text = "Hello, World!"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
text_color = (0, 0, 0)  # Màu đen
thickness = 2

# Lấy kích thước hình chữ nhật bao quanh văn bản
text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
text_width, text_height = text_size

# Tính toán vị trí để văn bản nằm giữa ảnh
x = int((width - text_width) / 2)
y = int((height + text_height) / 2)

# Vẽ hình chữ nhật bao quanh văn bản
rectangle_color = (0, 255, 0)  # Màu xanh lá cây
rectangle_thickness = -1  # Độ dày âm để tạo hiệu ứng trong suốt
cv2.rectangle(background, (x, y - text_height), (x + text_width, y), rectangle_color, rectangle_thickness)

# Áp dụng hiệu ứng trong suốt cho hình chữ nhật
transparency = 0.5  # Độ trong suốt (0.0 - 1.0)
overlay = background.copy()
cv2.rectangle(overlay, (x, y - text_height), (x + text_width, y), rectangle_color, -1)
background = cv2.addWeighted(overlay, transparency, background, 1 - transparency, 0)

# Vẽ văn bản lên ảnh
cv2.putText(background, text, (x, y), font, font_scale, text_color, thickness)

# Hiển thị ảnh
cv2.imshow("Text with Transparent Rectangle", background)
cv2.waitKey(0)
cv2.destroyAllWindows()