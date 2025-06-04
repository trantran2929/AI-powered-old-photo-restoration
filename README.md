# AI-powered-old-photo-restoration
Phục chế ảnh cũ
## 1. Thư viện được sử dụng<br>
Các thư viện Python được sử dụng trong mã nguồn bao gồm:

- OpenCV (cv2): Xử lý ảnh, đọc/ghi ảnh, tạo các biến dạng như nhiễu Gaussian, biến dạng màu sắc, và vùng mất mảng bất quy tắc.
- NumPy (np): Quản lý mảng số học, tạo nhiễu ngẫu nhiên (Gaussian, nhiễu vùng mất mảng).
- Random: Tạo các giá trị ngẫu nhiên cho biến dạng (hệ số màu sắc, kích thước vùng mất mảng).
- Pathlib (Path): Quản lý đường dẫn tệp, đảm bảo tính tương thích khi làm việc với thư mục ảnh gốc và hỏng.
- PyTorch (torch, torch.nn, torch.optim, torch.nn.functional): Xây dựng, huấn luyện, và đánh giá mô hình học sâu (UNetGenerator, PatchGANDiscriminator), hỗ trợ tính toán trên GPU.
- Torchvision (torchvision.transforms): Biến đổi ảnh (resize, chuẩn hóa) để chuẩn bị dữ liệu cho mô hình.
- PIL (PIL.Image): Đọc và xử lý ảnh định dạng PIL, hỗ trợ biến đổi ảnh trong dataset.
- Matplotlib (matplotlib.pyplot): Trực quan hóa kết quả, tạo biểu đồ loss và so sánh ảnh (hỏng, phục hồi, gốc).
- Torch.utils.data (Dataset, DataLoader, random_split): Tạo dataset tùy chỉnh (FaceRestorationGANDataset), chia dữ liệu thành tập train/val, và quản lý batch.
## 2. Quy trình thực hiện<br>
Quy trình của dự án được chia thành các bước chính, từ tiền xử lý dữ liệu, xây dựng mô hình, huấn luyện, đến đánh giá và trực quan hóa kết quả. Dưới đây là mô tả ngắn gọn:

### Bước 1: Tiền xử lý và tạo ảnh hỏng
- Mục đích: Tạo các ảnh hỏng từ tập dữ liệu gốc (/kaggle/input/fakefaces) để mô phỏng các tình huống thực tế.
- Thực hiện:
  + Đọc ảnh gốc bằng cv2.imread.
  + Áp dụng ba loại biến dạng:
    * Nhiễu Gaussian: Thêm nhiễu ngẫu nhiên (mean=0, sigma=25) bằng add_gaussian_noise.
    * Biến dạng màu sắc: Chuyển ảnh sang không gian HSV, điều chỉnh độ bão hòa ([0.3, 0.7]) và độ sáng ([0.6, 1.2]) bằng distort_colors.
    * Vùng mất mảng bất quy tắc: Tạo 3 đa giác ngẫu nhiên (6-10 đỉnh, kích thước 80-200 pixel), lấp đầy bằng màu đen, trắng, hoặc nhiễu ngẫu nhiên bằng occlude_irregular.
  + Lưu ảnh hỏng vào /kaggle/working/fakefaces_degraded với hậu tố _noisy.jpg, _color.jpg, _occluded.jpg bằng cv2.imwrite.
### Bước 2: Chuẩn bị dữ liệu
- Dataset: Sử dụng lớp FaceRestorationGANDataset để tạo cặp ảnh hỏng và ảnh gốc:
  + Đọc ảnh hỏng từ /kaggle/working/fakefaces_degraded và ảnh gốc từ /kaggle/input/fakefaces.
  + Áp dụng biến đổi (transforms.Compose): resize về 256x256, chuyển thành tensor, chuẩn hóa giá trị pixel về [-1, 1].
- DataLoader: Chia dữ liệu thành 80% train và 20% val (pretrain) hoặc 90% train và 10% val (GAN training), sử dụng random_split và DataLoader với batch size 8 (pretrain) hoặc 4 (GAN).
### Bước 3: Xây dựng mô hình
- UNetGenerator:
  + Kiến trúc U-Net với encoder (8 lớp tích chập, giảm kích thước từ 256x256 xuống 1x1) và decoder (7 lớp tích chập chuyển vị, tăng kích thước về 256x256).
  + Sử dụng skip connections để kết nối encoder và decoder, cải thiện tái tạo chi tiết.
  + Đầu ra sử dụng Tanh để chuẩn hóa trong [-1, 1].
- PatchGANDiscriminator:
  + Kiến trúc PatchGAN với 5 lớp tích chập, nhận đầu vào là cặp ảnh (hỏng + phục hồi/gốc).
  + Xuất ra ma trận xác suất để đánh giá độ chân thực từng vùng ảnh.
### Bước 4: Pretrain Generator
- Mục đích: Huấn luyện trước Generator bằng L1 và L2 loss để ổn định trước khi huấn luyện GAN.
- Thực hiện:
  + Sử dụng hàm mất mát: L1 (khoảng cách pixel-wise) + 0.1*L2 (MSE).
  + Huấn luyện trong 10 epoch, batch size 8, learning rate 0.001 (Adam).
  + Áp dụng scheduler (StepLR) để giảm learning rate sau mỗi 15 epoch.
  + Early stopping nếu validation loss không cải thiện sau 10 epoch.
  + Lưu mô hình tốt nhất (pretrained_generator_best.pth) và mẫu ảnh (pretrain_sample_epoch_*.png).
  + Vẽ biểu đồ loss (pretrain_loss_curves.png) bằng plot_pretrain_losses.
### Bước 5: Huấn luyện GAN
- Mục đích: Huấn luyện Generator và Discriminator trong cơ chế đối kháng để cải thiện độ chân thực.
- Thực hiện:
  + Tải Generator pretrained (nếu có) từ pretrained_generator_best.pth.
  + Sử dụng hàm mất mát:
    * Discriminator: GAN loss (MSE) cho ảnh thật (original + degraded) và giả (restored + degraded).
    * Generator: GAN loss + 100*L1 loss để cân bằng độ chân thực và chi tiết.
  + Huấn luyện trong 10 epoch, batch size 4, learning rate 0.0001 (Generator) và 0.0002 (Discriminator).
  + Áp dụng scheduler (StepLR) giảm learning rate sau mỗi 25 epoch.
  + Generator được huấn luyện mỗi 2 batch sau epoch 10 để ổn định Discriminator.
  + Lưu checkpoint mỗi 10 epoch (face_restoration_gan_checkpoint_epoch_*.pth), mô hình cuối (face_restoration_generator_final.pth, face_restoration_discriminator_final.pth), và mẫu ảnh (gan_sample_epoch_*.png).
### Bước 6: Đánh giá và trực quan hóa
- Đánh giá:
  + Tính validation L1 loss trong quá trình huấn luyện GAN để theo dõi hiệu quả.
  + So sánh trực quan bằng compare_results, hiển thị 5 bộ ảnh (hỏng, phục hồi, gốc) và lưu vào comparison_batch_*.png.
- Phục hồi ảnh: Hàm restore_image_gan nhận ảnh hỏng, xử lý qua Generator, và lưu ảnh phục hồi.
- Trực quan hóa:
  + Biểu đồ loss trong pretrain (pretrain_loss_curves.png).
  + Ảnh mẫu trong pretrain và GAN training (pretrain_sample_*.png, gan_sample_*.png).
  + So sánh trước/sau (comparison_batch_*.png).
