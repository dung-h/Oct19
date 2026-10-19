# Đối chứng mặc định thay đổi theo loss

Phát hiện từ get_all_params của hai saved models, trước bất kỳ score nào của nhánh
mới: MultiClass dùng Bayesian/bagging_temperature1/leaf_estimation_iterations1;
Logloss dùng MVS/subsample0.8/leaf_estimation_iterations10. CatBoost CTR target
borders cũng khác theo cardinality của target. Giữ cùng explicit hyperparameters
trước đây không khóa được các training defaults này.

Thử một gói `ordinal_defaults`: giữ277features,depth4,L2=8,300trees,lr0.05,
seed20261018,PAV và policy gốc; chỉ đặt Bayesian/bagging_temperature1 và
leaf_estimation_iterations1. Không ép CTR target borders4 cho target binary.
Không tuyên bố tách riêng tác động bootstrap và leaf updates; kiểm tra tác động
của cả gói default. Dùng5outer/4inner cũ; cutoff chọninner, final4inner toàn train,
fit/reload4models và giữ CSV riêng. So với multiclass và ordinal gốc trong báo cáo.

Nhánh mới lưu outputs/defaults; không thay đổi protocol/kết quả của hai can thiệp
regularization và bỏneighbor đang chạy. Nghiên cứu exploratory sau khi xem kết quả
ordinal gốc; không upload Kaggle hoặc dùng nhãn outer để chọn config.
