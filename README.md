# AI-powered-old-photo-restoration
Phục chế ảnh cũ
1. Thư viện được sử dụng
Các thư viện Python được sử dụng trong mã nguồn bao gồm:

OpenCV (cv2): Xử lý ảnh, đọc/ghi ảnh, tạo các biến dạng như nhiễu Gaussian, biến dạng màu sắc, và vùng mất mảng bất quy tắc.
NumPy (np): Quản lý mảng số học, tạo nhiễu ngẫu nhiên (Gaussian, nhiễu vùng mất mảng).
Random: Tạo các giá trị ngẫu nhiên cho biến dạng (ví dụ: hệ số màu sắc, kích thước vùng mất mảng).
Pathlib (Path): Quản lý đường dẫn tệp, đảm bảo tính tương thích khi làm việc với thư mục ảnh gốc và hỏng.
PyTorch (torch, torch.nn, torch.optim, torch.nn.functional): Xây dựng, huấn luyện, và đánh giá mô hình học sâu (UNetGenerator, PatchGANDiscriminator), hỗ trợ tính toán trên GPU.
Torchvision (torchvision.transforms): Biến đổi ảnh (resize, chuẩn hóa) để chuẩn bị dữ liệu cho mô hình.
PIL (PIL.Image): Đọc và xử lý ảnh định dạng PIL, hỗ trợ biến đổi ảnh trong dataset.
Matplotlib (matplotlib.pyplot): Trực quan hóa kết quả, tạo biểu đồ loss và so sánh ảnh (hỏng, phục hồi, gốc).
Torch.utils.data (Dataset, DataLoader, random_split): Tạo dataset tùy chỉnh (FaceRestorationGANDataset), chia dữ liệu thành tập train/val, và quản lý batch.
