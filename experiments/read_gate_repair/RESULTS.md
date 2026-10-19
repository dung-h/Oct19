# Sửa read gate bằng binary target, author history và calibration

Primary không giảm tổng lỗi theo tiêu chí đã khóa; giữ incumbent làm bản chính và dừng hướng author/calibration này.

## Kết quả OOF: số mẫu nhận/rớt và lỗi

| branch             |   accepted |   dropped |         TP |        FP |        FN |        TN |   total_error |   missed_high |       H |       Q |       R |
|:-------------------|-----------:|----------:|-----------:|----------:|----------:|----------:|--------------:|--------------:|--------:|--------:|--------:|
| incumbent_deployed | 1616.00000 | 878.00000 | 1278.00000 | 338.00000 | 313.00000 | 565.00000 |     651.00000 |      38.00000 | 0.62991 | 0.51809 | 0.80327 |
| incumbent_at_050   | 1773.00000 | 721.00000 | 1355.00000 | 418.00000 | 236.00000 | 485.00000 |     654.00000 |      27.00000 | 0.62589 | 0.49474 | 0.85167 |
| binary_raw         | 1749.00000 | 745.00000 | 1342.00000 | 407.00000 | 249.00000 | 496.00000 |     656.00000 |      31.00000 | 0.62530 | 0.49679 | 0.84349 |
| binary_global      | 1736.00000 | 758.00000 | 1337.00000 | 399.00000 | 254.00000 | 504.00000 |     653.00000 |      33.00000 | 0.62594 | 0.49869 | 0.84035 |
| binary_venue       | 1746.00000 | 748.00000 | 1343.00000 | 403.00000 | 248.00000 | 500.00000 |     651.00000 |      31.00000 | 0.62623 | 0.49775 | 0.84412 |
| author_raw         | 1743.00000 | 751.00000 | 1337.00000 | 406.00000 | 254.00000 | 497.00000 |     660.00000 |      27.00000 | 0.62517 | 0.49773 | 0.84035 |
| author_global      | 1721.00000 | 773.00000 | 1323.00000 | 398.00000 | 268.00000 | 505.00000 |     666.00000 |      30.00000 | 0.62457 | 0.50009 | 0.83155 |
| author_venue       | 1730.00000 | 764.00000 | 1333.00000 | 397.00000 | 258.00000 | 506.00000 |     655.00000 |      28.00000 | 0.62618 | 0.49989 | 0.83784 |

`author_venue` so với incumbent: cứu 70 FN, bỏ 11 FP, tạo 70 FP mới và làm mất 15 TP. Delta tổng lỗi=4.

Tổng OOF `author_venue`: FP=397, FN=258, high-label FN=28. Incumbent: FP=338, FN=313, high-label FN=38. Folds giảm total error: 2/5.

## Test artifact

| branch             |   total |   accepted |   dropped |
|:-------------------|--------:|-----------:|----------:|
| incumbent_deployed |     596 |        356 |       240 |
| incumbent_at_050   |     596 |        390 |       206 |
| binary_raw         |     596 |        386 |       210 |
| binary_global      |     596 |        385 |       211 |
| binary_venue       |     596 |        390 |       206 |
| author_raw         |     596 |        388 |       208 |
| author_global      |     596 |        384 |       212 |
| author_venue       |     596 |        389 |       207 |

Test không có nhãn nên bảng này chỉ là accepted/dropped, không phải TP/FP/FN. `submission_author_venue.csv` là candidate experimental; incumbent cũ không bị ghi đè và không upload Kaggle.

Verifier **PASSED**: replay toàn bộ decision outer, rebuild author features final và reload 2 CatBoost + calibrators cho 8 branch. Outer folds đã được xem nhiều lần nên đây là kết quả exploratory.
