import re

# Đường dẫn đến file chứa dữ liệu
file_path = "ava_action_list_v2.2.pbtxt"

# Đọc nội dung file
with open(file_path, 'r') as file:
    content = file.read()

# Tìm tất cả các khối dict trong nội dung file
pattern = r"label \{([^}]*)\}"
matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

# Lặp qua từng khối dict và trích xuất label_id và name
for match in matches:
    label_id = re.search(r"label_id: (\d+)", match).group(1)
    name = re.search(r'name: "(.*)"', match).group(1)
    print('{} : {}'.format(int(label_id) - 1, name))