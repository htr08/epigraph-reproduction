# Phân tích kết quả chạy lại EpiGraph trên Kaggle

Nguồn: `logs.txt` (log chạy notebook Kaggle) đối chiếu với paper *EpiGraph* (Sci Rep 2024, 14:27496), Table 1 và Table 3.

## 1. Cấu trúc log

| Đoạn log | Nội dung |
|---|---|
| Dòng 1–17 | Cảnh báo debugger, clone repo, pip install (không quan trọng) |
| Dòng 18–22 | Tìm file pkl, liệt kê 10 checkpoint `Best_GAT_*.pt`, sửa code cho tương thích phiên bản thư viện trên Kaggle |
| Dòng 23–37 | **Lần 1:** inference bằng 10 checkpoint pretrained của tác giả |
| Dòng 38–2619 | **Lần 2:** tự train lại, KFold 10 fold (179 train / 20 val mỗi fold), 50 epoch |
| Dòng 2620–2624 | Metric trên test set của 10 model vừa train |
| Cuối log | Cảnh báo nbconvert (không quan trọng) |

## 2. So sánh với paper (test set epitope3D)

| Metric | Paper | Checkpoint tác giả | Tự train lại |
|---|---|---|---|
| AUC-ROC | 0.73 | 0.73 | 0.72 |
| AUC-PR | 0.23 | 0.23 | 0.22 |
| F1 | 0.29 | 0.29 | 0.29 |
| BACC | 0.62 | 0.62 | 0.63 |
| MCC | 0.19 | 0.19 | 0.19 |

- **Checkpoint tác giả:** khớp hoàn toàn với paper, nên pipeline inference và dữ liệu test đúng.
- **Tự train lại:** lệch tối đa 0.01, nằm trong nhiễu ngẫu nhiên. Cách chia fold thì cố định (`random.seed(30)` rồi `random.shuffle` trước `KFold`), nhưng khởi tạo trọng số của model không được set seed (`torch.manual_seed`) nên mỗi lần train ra model hơi khác nhau; thứ tự mẫu train cũng khác bản gốc do bản vá `set` (xem mục 6). Kết luận: reproduce thành công.
- **So với baseline trong paper:** AUC-ROC 0.72 cao hơn GraphBepi (0.64) và ngang BepiPred-3.0 / DiscoTope-3.0 (0.71). F1, MCC, BACC xấp xỉ GraphBepi.
- **AUC-PR thấp (0.22–0.23)** là bình thường vì bài toán mất cân bằng nặng (rất ít residue epitope); cần so với tỉ lệ dương, không so với 1.0.

## 3. Kết quả từng fold (trên tập validation 20 antigen)

| Fold | Chỉ số test | Best epoch | Val loss | Val AUC-ROC |
|---|---|---|---|---|
| 1 | 0–19 | 14 | 0.327 | 0.628 |
| 2 | 20–39 | 12 | 0.277 | 0.705 |
| 3 | 40–59 | 17 | 0.274 | 0.650 |
| 4 | 60–79 | 16 | 0.301 | 0.713 |
| 5 | 80–99 | 9 | 0.351 | 0.686 |
| 6 | 100–119 | 32 | 0.278 | 0.761 |
| 7 | 120–139 | 8 | 0.299 | 0.657 |
| 8 | 140–159 | 11 | 0.330 | 0.630 |
| 9 | 160–179 | 11 | 0.381 | 0.701 |
| 10 | 180–198 | 17 | 0.264 | 0.662 |

Val AUC trung bình ≈ **0.68**.

**Val AUC là gì (theo `train.py`):** sau mỗi epoch, model được chạy trên 20 antigen bị giữ lại của fold đó (code gọi nhầm là `X_test`, thực chất là tập validation). Chỉ các residue bề mặt (`train_mask`) được tính. 20 antigen chia thành 3 batch (8 + 8 + 4); AUC-ROC được tính riêng cho từng batch rồi lấy trung bình 3 giá trị. Val loss tính tương tự. Hai số này dùng để chọn checkpoint, không phải kết quả cuối; paper không báo cáo chúng, kết quả cuối là test set ở mục 2.

## 4. Nhận xét

1. **Val AUC (0.68) thấp hơn test AUC (0.72) và dao động mạnh giữa các fold (0.63–0.76).** Mỗi fold chỉ có 20 antigen nên số đo nhiễu. Test cao hơn nhờ ensemble trung bình 10 model.
2. **Overfit sớm.** Training loss giảm đều đến ~0.23 nhưng val loss tăng lại sau epoch ~12–17. Best epoch hầu hết chỉ 8–17 (ngoại lệ fold 6: 32).
3. **"Best auc-roc" không phải AUC cao nhất của fold.** Paper chỉ cập nhật checkpoint khi *cả* val loss và val AUC cùng cải thiện, nên epoch được chọn có thể không trùng đỉnh AUC (ví dụ fold 1: AUC lên ~0.635 ở epoch ~22 nhưng model lưu ở epoch 14 với 0.6275).
4. **Mất ổn định cuối quá trình train.** Fold 2 (epoch 46–49) và fold 4 (epoch 44–47) có val loss nhảy lên 0.37–0.38 rồi hạ lại. Không ảnh hưởng vì checkpoint tốt đã lưu từ sớm.
5. **Thời gian train trong log là cộng dồn:** khoảng 190s mỗi fold (fold 10 hiển thị 1919s).
6. **Số antigen:** mỗi fold 179 + 20 = 199, trong khi paper nói 200 antigen. Các file csv của tác giả liệt kê đủ 200 antigen (180 + 20); so mã PDB thì file pkl công bố thiếu đúng một antigen là **3FCS**, các antigen còn lại và 45 antigen test đều khớp. Lý do thiếu không được ghi trong repo; ảnh hưởng không đáng kể.
7. **Bản vá `set` trong `train.py`:** bản gốc dùng `list(set(X) - set(X_test))` nên thứ tự mẫu train không xác định; bản vá giữ thứ tự gốc. Tập mẫu train vẫn như nhau, chỉ thứ tự khác.

## 5. Gợi ý cho báo cáo

- Kết luận: *reproduce thành công, lệch ≤ 0.01 do tính ngẫu nhiên*.
- Muốn chặt hơn: thêm `torch.manual_seed(...)` (split fold đã cố định bằng seed 30 trong code gốc) và chạy nhiều lần với seed khác nhau, báo cáo mean ± std.
- Phần thảo luận có thể dùng: val AUC thấp hơn test và nhiễu theo fold (tập val nhỏ), overfit sớm, cơ chế lưu checkpoint theo hai metric.
