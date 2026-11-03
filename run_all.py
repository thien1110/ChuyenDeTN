# -*- coding: utf-8 -*-
"""
run_all.py  –  PHIÊN BẢN DUY NHẤT

Chạy MỘT lệnh là có toàn bộ số liệu, bảng và hình cho báo cáo (thay cho run_pipeline.py, run_pipeline_v2.py,
run_pipeline_v3.py, run_bo_sung.py, run_cai_tien_lstm.py):

    python run_all.py > nhat_ky.txt 2>&1

Cần có cùng thư mục:  raw_his_data.csv  và  run_all.py.   Kết quả nằm trong thư mục  KET_QUA/  (bảng) và  KET_QUA/hinh/  (hình).
ĐỌC TRƯỚC:  KET_QUA/TOM_TAT_SO_LIEU.txt  và  KET_QUA/MUC_LUC_KET_QUA.txt.

Các phần (mỗi phần chạy độc lập, một phần lỗi thì các phần khác vẫn chạy):
  0. Xây dựng dữ liệu từ raw_his_data.csv (làm sạch, Census, 17 đặc trưng, đã sửa ngày lễ Tết 2026)
  A. Phân tích mô tả (EDA), Bảng 2.1, tương quan, kiểm tra phạm vi
  B. Mô hình cơ sở: naive, seasonal naive, ARIMA/SARIMA/SARIMAX (cuốn chiếu 1 bước), XGBoost trực tiếp
  C. LSTM: chọn cấu hình trong 7 ứng viên (K0–K6) trên tập KIỂM ĐỊNH, chấm trên tập KIỂM THỬ một lần với 10 seed
  D. Bảng 3.1, Bảng 3.2, kiểm định Diebold–Mariano, Hình 3.3
  E. Giải thích: mô hình đại diện XGBoost, TreeSHAP, Permutation Importance, Integrated Gradients (cho LSTM đã chọn, seed 42)
  F. Phân tích ngưỡng cảnh báo công suất

QUY TẮC TRUNG THỰC (không được vi phạm):
  - Biến thể LSTM được chọn bằng tập Kiểm định. Không sửa danh sách UNG_VIEN rồi chạy lại để tìm kết quả kiểm thử đẹp hơn.
  - Bảy cấu hình ứng viên được thiết kế sau khi đã quan sát cấu hình K0 trên tập kiểm thử: phải khai báo điều này trong báo cáo.
  - File removed_cases_log_KHONG_CONG_BO.csv chứa dữ liệu cấp người bệnh: KHÔNG đưa lên GitHub.
"""
import json
import os
import random
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # luôn chạy trong thư mục chứa file này

# =====================================================================
# CẤU HÌNH – chỉ sửa phần này nếu cần
# =====================================================================
THU_MUC = 'KET_QUA'
HINH = THU_MUC + '/hinh'
SO_SEED = 10                       # số seed cuối cùng trên tập Kiểm thử
SO_GIUONG_KE_HOACH = 765           # số giường kế hoạch do Phòng KHTH cung cấp (None = bỏ qua phần F)
NGUONG_CONG_SUAT = 0.85            # ngưỡng cảnh báo 85%
TET_BAT_DAU = '2026-02-05'         # định nghĩa kỳ Tết (xác định sau khi quan sát chuỗi census)
TET_KET_THUC = '2026-03-04'
TAI_TAO_DU_LIEU = False            # True: xây lại dữ liệu từ raw_his_data.csv ngay cả khi KET_QUA/df_final.csv đã có
WINDOW = 7
SEED_CHON = [101, 102, 103]        # seed dùng khi chọn biến thể (trên tập Kiểm định)
SEED_CUOI = ([42] + [s for s in range(1, 100) if s != 42])[:SO_SEED]

# Các kỳ nghỉ Tết (để tô màu biểu đồ census toàn giai đoạn)
TET_NGHI = [('2024-02-08', '2024-02-14'), ('2025-01-25', '2025-02-02'), ('2026-02-14', '2026-02-22')]

# Danh sách các ngày nghỉ lễ chính thức tại Việt Nam (2024 - 2026), đã sửa Tết Bính Ngọ 2026 = 14/02 – 22/02
VN_HOLIDAYS = pd.to_datetime([
    '2024-01-01', '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11',
    '2024-02-12', '2024-02-13', '2024-02-14', '2024-04-18', '2024-04-29',
    '2024-04-30', '2024-05-01', '2024-09-02', '2024-09-03',
    '2025-01-01', '2025-01-25', '2025-01-26', '2025-01-27', '2025-01-28',
    '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02',
    '2025-04-07', '2025-04-30', '2025-05-01', '2025-05-02', '2025-09-01',
    '2025-09-02',
    # [SỬA v2] Tết Bính Ngọ 2026: nghỉ liên tục 14/02 -> 22/02/2026 (Thông báo 9441/TB-BNV).
    # Bản cũ ghi nhầm 13/02 -> 20/02 (13/02 là thứ Sáu đi làm bình thường).
    '2026-01-01', '2026-02-14', '2026-02-15', '2026-02-16',
    '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-02-21', '2026-02-22',
    '2026-04-26', '2026-04-27', '2026-04-30', '2026-05-01'
])

TOM_TAT = []
DU_BAO = {}
PRED_CHON = {}


def ghi(dong):
    print(dong)
    TOM_TAT.append(dong)


def luu(df_, ten, index=True):
    df_.to_csv(os.path.join(THU_MUC, ten), index=index, encoding='utf-8-sig')


def chay(ten, ham):
    print('\n' + '=' * 70)
    print(ten)
    print('=' * 70)
    t0 = time.time()
    try:
        ham()
        print(f'[xong] {ten} ({time.time() - t0:.0f} giây)')
    except Exception as e:  # noqa
        print(f'[LỖI] {ten}: {type(e).__name__}: {e}')
        traceback.print_exc()
        TOM_TAT.append(f'[LỖI – phần này chưa có số liệu] {ten}: {type(e).__name__}: {e}')


# =====================================================================
# PHẦN 0 – XÂY DỰNG DỮ LIỆU TỪ raw_his_data.csv
# =====================================================================
def xay_dung_du_lieu():
    print("BƯỚC 1: Đang nạp và xử lý dữ liệu thô từ raw_his_data.csv...")
    raw_df = pd.read_csv('raw_his_data.csv', dtype={'Patient_ID_Hash': str}, low_memory=False)
    raw_df.columns = raw_df.columns.str.strip()

    # Chuẩn hóa thời gian vào/ra viện
    raw_df['Admission_Time'] = pd.to_datetime(raw_df['Admission_Time'], format='mixed', dayfirst=True, errors='coerce')
    raw_df['Discharge_Time'] = pd.to_datetime(raw_df['Discharge_Time'], format='mixed', dayfirst=True, errors='coerce')
    raw_df['Age'] = raw_df['Admission_Time'].dt.year - raw_df['DoB_Year']

    # 1.1. Trích xuất nhóm đặc trưng lâm sàng và nhân khẩu học theo ngày nhập viện
    raw_df['Date'] = raw_df['Admission_Time'].dt.normalize()
    raw_df['Is_Elderly'] = (raw_df['Age'] >= 60).astype(int)
    raw_df['ICD_Main_Letter'] = raw_df['ICD_Code_Main'].astype(str).str[0].str.upper()
    raw_df['Is_Respiratory'] = (raw_df['ICD_Main_Letter'] == 'J').astype(int)
    raw_df['Is_Cardiovascular'] = (raw_df['ICD_Main_Letter'] == 'I').astype(int)
    raw_df['Is_Infectious'] = (raw_df['ICD_Main_Letter'].isin(['A', 'B'])).astype(int)

    # 1.2. Lọc hồ sơ ảo (< 15 phút) và tổng hợp chuỗi thời gian công suất giường (Census)
    raw_df['Duration_Hours'] = (raw_df['Discharge_Time'] - raw_df['Admission_Time']).dt.total_seconds() / 3600.0
    df_clean = raw_df[(raw_df['Duration_Hours'] >= 0.25) & (raw_df['Discharge_Time'] <= '2026-08-28 23:59:59')].copy()

    daily_clinical_features = raw_df.groupby('Date').agg(
        Avg_ICD_Sub_Count=('ICD_Sub_Count', 'mean'),
        Total_Elderly_Admitted=('Is_Elderly', 'sum'),
        Total_Respiratory=('Is_Respiratory', 'sum'),
        Total_Cardiovascular=('Is_Cardiovascular', 'sum'),
        Total_Infectious=('Is_Infectious', 'sum')
    ).reset_index()



    # [BỔ SUNG v2] Thống kê số bản ghi bị loại ở bước làm sạch (dùng cho mục dữ liệu)
    _thong_ke_lam_sach = {
        'so_ban_ghi_trich_xuat': len(raw_df),
        'thieu_gio_ra_vien': int(raw_df['Discharge_Time'].isna().sum()),
        'nam_vien_duoi_15_phut': int((raw_df['Duration_Hours'] < 0.25).sum()),
        'ra_vien_sau_28_08_2026': int((raw_df['Discharge_Time'] > '2026-08-28 23:59:59').sum()),
        'so_ban_ghi_hop_le': len(df_clean),
    }
    pd.Series(_thong_ke_lam_sach).to_csv(THU_MUC + '/thong_ke_lam_sach.csv', header=['gia_tri'], encoding='utf-8-sig')
    print('--> Thống kê làm sạch:', _thong_ke_lam_sach)

    # Lưu nhật ký hồ sơ ảo (< 15 phút) để đối chiếu vận hành
    df_removed = raw_df[raw_df['Duration_Hours'] < 0.25].copy()
    df_removed.to_csv(THU_MUC + '/removed_cases_log_KHONG_CONG_BO.csv', index=False, encoding='utf-8-sig')

    # Giữ lại các ca hợp lệ (>= 15 phút, chốt sổ xuất viện tại thời điểm trích xuất 28/08/2026)

    min_date = df_clean['Admission_Time'].min().normalize()
    max_date = pd.to_datetime('2026-08-27')
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    adm_norm = df_clean['Admission_Time'].dt.normalize()
    dis_norm = df_clean['Discharge_Time'].dt.normalize()

    daily_data = []
    for current_date in date_range:
        next_date = current_date + pd.Timedelta(days=1)
        active_patients = df_clean[
            (df_clean['Admission_Time'] < next_date) &
            ((df_clean['Discharge_Time'] >= next_date) | (df_clean['Discharge_Time'].isna()))
        ]
        daily_data.append({
            'Date': current_date,
            'Census': int(len(active_patients)),
            'Admissions': int((adm_norm == current_date).sum()),
            'Discharges': int((dis_norm == current_date).sum()),
            'Avg_Age': float(active_patients['Age'].mean()),
            'Emergency_Count': int(active_patients['Admission_Source'].sum())
        })

    df_timeseries = pd.DataFrame(daily_data)

    # Cắt 30 ngày đầu tiên để loại bỏ hiệu ứng khởi động lạnh (Cold-start)
    df_timeseries = df_timeseries.iloc[30:].reset_index(drop=True)

    # Tạo đặc trưng chu kỳ lịch và độ trễ thời gian
    df_timeseries['DayOfWeek'] = df_timeseries['Date'].dt.dayofweek
    df_timeseries['Is_Weekend'] = (df_timeseries['DayOfWeek'] >= 5).astype(int)
    df_timeseries['Month'] = df_timeseries['Date'].dt.month
    df_timeseries['Lag_1'] = df_timeseries['Census'].shift(1)
    df_timeseries['Lag_2'] = df_timeseries['Census'].shift(2)
    df_timeseries['Lag_7'] = df_timeseries['Census'].shift(7)
    df_timeseries['Rolling_Mean_7'] = df_timeseries['Census'].rolling(window=7).mean()
    df_timeseries['Is_Holiday'] = df_timeseries['Date'].isin(VN_HOLIDAYS).astype(int)

    # Loại bỏ các dòng NaN sinh ra do trượt độ trễ và xuất features.csv
    features_df = df_timeseries.dropna().reset_index(drop=True)
    features_df.to_csv(THU_MUC + '/features.csv', index=False)
    print(f"Đã tạo và lưu 'features.csv' thành công ({len(features_df)} ngày).")

    # Hợp nhất với nhóm đặc trưng lâm sàng và khóa sổ tại 31/07/2026
    df_final = pd.merge(features_df, daily_clinical_features, on='Date', how='left')
    df_final = df_final.fillna(0).sort_values(by='Date').reset_index(drop=True)

    cutoff_date = pd.to_datetime('2026-07-31')
    df_final = df_final[df_final['Date'] <= cutoff_date].reset_index(drop=True)
    print(f"Tổng số ngày đưa vào thực nghiệm (đến 31/07/2026): {len(df_final)} ngày.")
    df_final.to_csv(THU_MUC + '/df_final.csv', index=False)   # [BỔ SUNG v2] dùng cho run_bo_sung.py
    return df_final


# Biến dùng chung (được gán trong nap_du_lieu)
df = X_COLS = n = TRAIN_END = VAL_END = train_df = val_df = test_df = y_test = dates_test = mask_tet = None


def nap_du_lieu():
    global df, X_COLS, n, TRAIN_END, VAL_END, train_df, val_df, test_df, y_test, dates_test, mask_tet
    df = pd.read_csv(os.path.join(THU_MUC, 'df_final.csv'), parse_dates=['Date'])
    X_COLS = [c for c in df.columns if c not in ('Census', 'Date')]
    n = len(df)
    TRAIN_END = int(n * 0.70)
    VAL_END = int(n * 0.80)
    train_df = df.iloc[:TRAIN_END]
    val_df = df.iloc[TRAIN_END:VAL_END]
    test_df = df.iloc[VAL_END:]
    y_test = test_df['Census'].values[WINDOW:].astype(float)
    dates_test = pd.DatetimeIndex(test_df['Date'].values[WINDOW:])
    mask_tet = np.asarray((dates_test >= pd.Timestamp(TET_BAT_DAU)) & (dates_test <= pd.Timestamp(TET_KET_THUC)))


def phan_du_lieu():
    if TAI_TAO_DU_LIEU or not os.path.exists(os.path.join(THU_MUC, 'df_final.csv')):
        xay_dung_du_lieu()
    else:
        print('Dùng lại KET_QUA/df_final.csv đã có (đặt TAI_TAO_DU_LIEU = True để xây lại từ raw_his_data.csv).')
    nap_du_lieu()
    ghi(f'[Dữ liệu] {n} ngày; Train {len(train_df)} ngày ({train_df["Date"].iloc[0]:%d/%m/%Y} – {train_df["Date"].iloc[-1]:%d/%m/%Y}), '
        f'Validation {len(val_df)} ngày, Test {len(test_df)} ngày ({test_df["Date"].iloc[0]:%d/%m/%Y} – {test_df["Date"].iloc[-1]:%d/%m/%Y}); '
        f'{len(y_test)} mẫu kiểm thử.')


# =====================================================================
# ĐO SAI SỐ, KIỂM ĐỊNH DIEBOLD–MARIANO
# =====================================================================
def so_do(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    return {'MAE': float(np.mean(np.abs(e))), 'RMSE': float(np.sqrt(np.mean(e ** 2))),
            'MAPE': float(np.mean(np.abs(e) / y) * 100), 'ME': float(np.mean(e))}


def bang_giai_doan(p):
    rows = []
    for ten, m in [('Toàn tập kiểm thử', np.ones(len(y_test), bool)),
                   ('Kỳ Tết Nguyên đán', mask_tet),
                   ('Ngày thường', ~mask_tet)]:
        r = so_do(y_test[m], np.asarray(p)[m])
        r['So_ngay'] = int(m.sum())
        r['Giai_doan'] = ten
        rows.append(r)
    return pd.DataFrame(rows)[['Giai_doan', 'So_ngay', 'MAE', 'RMSE', 'MAPE', 'ME']]


def dm_test(y, p_co_so, p_lstm, luy_thua=2):
    """Kiểm định Diebold–Mariano (h = 1) có hiệu chỉnh Harvey–Leybourne–Newbold.
    luy_thua = 2: hàm tổn thất bình phương; luy_thua = 1: sai số tuyệt đối.
    Thống kê > 0 nghĩa là mô hình cơ sở có sai số LỚN HƠN LSTM. Trả về (thống kê, p hai phía)."""
    y = np.asarray(y, float)
    d = np.abs(y - np.asarray(p_co_so, float)) ** luy_thua - np.abs(y - np.asarray(p_lstm, float)) ** luy_thua
    T = len(d)
    dbar = d.mean()
    var = np.var(d, ddof=0)
    if var == 0:
        return float('nan'), float('nan')
    dm = dbar / np.sqrt(var / T)
    dm *= np.sqrt((T - 1) / T)   # hiệu chỉnh HLN cho h = 1
    p = 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))
    return float(dm), float(p)




# =====================================================================
# PHẦN A, B – EDA VÀ MÔ HÌNH CƠ SỞ
# =====================================================================
# =====================================================================
# PHẦN A – PHÂN TÍCH MÔ TẢ DỮ LIỆU (EDA)
# =====================================================================
def phan_A():
    # A1. Thống kê mô tả
    mo_ta = df[['Census'] + X_COLS].agg(['mean', 'std', 'min', 'max']).T.round(2)
    mo_ta.columns = ['Trung bình', 'Độ lệch chuẩn', 'Nhỏ nhất', 'Lớn nhất']
    luu(mo_ta, 'A1_thong_ke_mo_ta.csv')
    print(mo_ta.to_string())

    # A2. Biểu đồ chuỗi census toàn giai đoạn
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Date'], df['Census'], color='#1f77b4', linewidth=1)
    for i, (a, b) in enumerate(TET_NGHI):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.35, label='Kỳ nghỉ Tết' if i == 0 else None)
    ax.axvline(df['Date'].iloc[TRAIN_END], color='red', linestyle='--', label='Ranh giới Train | Validation')
    ax.axvline(df['Date'].iloc[VAL_END], color='green', linestyle='--', label='Ranh giới Validation | Test')
    ax.set_title('Số người bệnh nội trú (census) theo ngày, 11/2023 – 07/2026')
    ax.set_xlabel('Ngày')
    ax.set_ylabel('Census (người bệnh)')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(THU_MUC, 'A2_chuoi_census_toan_giai_doan.png'), dpi=300)
    plt.close(fig)

    # A3. Ma trận tương quan 17 đặc trưng
    tq = df[X_COLS].corr()
    luu(tq.round(3), 'A3_ma_tran_tuong_quan.csv')
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(tq.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(X_COLS)))
    ax.set_xticklabels(X_COLS, rotation=90, fontsize=8)
    ax.set_yticks(range(len(X_COLS)))
    ax.set_yticklabels(X_COLS, fontsize=8)
    for i in range(len(X_COLS)):
        for j in range(len(X_COLS)):
            ax.text(j, i, f'{tq.values[i, j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(tq.values[i, j]) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Ma trận tương quan Pearson giữa 17 đặc trưng')
    fig.tight_layout()
    fig.savefig(os.path.join(THU_MUC, 'A3_ma_tran_tuong_quan.png'), dpi=300)
    plt.close(fig)

    # A4. Bảng 2.1: nhập viện, ra viện, census theo thứ (toàn bộ 998 ngày)
    ten_thu = {0: 'Thứ Hai', 1: 'Thứ Ba', 2: 'Thứ Tư', 3: 'Thứ Năm', 4: 'Thứ Sáu', 5: 'Thứ Bảy', 6: 'Chủ nhật'}
    g = df.groupby('DayOfWeek').agg(So_ngay=('Census', 'count'),
                                    A_tb=('Admissions', 'mean'), A_sd=('Admissions', 'std'),
                                    D_tb=('Discharges', 'mean'), D_sd=('Discharges', 'std'),
                                    C_tb=('Census', 'mean'), C_sd=('Census', 'std'))
    b21 = pd.DataFrame({
        'Thứ': [ten_thu[i] for i in g.index],
        'Số ngày': g['So_ngay'].values,
        'Admissions (TB ± SD)': [f'{m:.1f} ± {s:.1f}' for m, s in zip(g['A_tb'], g['A_sd'])],
        'Discharges (TB ± SD)': [f'{m:.1f} ± {s:.1f}' for m, s in zip(g['D_tb'], g['D_sd'])],
        'Census (TB ± SD)': [f'{m:.1f} ± {s:.1f}' for m, s in zip(g['C_tb'], g['C_sd'])],
    })
    luu(b21, 'A4_Bang_2_1_theo_thu.csv', index=False)
    print(b21.to_string(index=False))
    ngay_nhap_nhieu = ten_thu[int(g['A_tb'].idxmax())]
    ngay_ra_nhieu = ten_thu[int(g['D_tb'].idxmax())]
    ghi(f'[Bảng 2.1] Ngày nhập viện trung bình cao nhất: {ngay_nhap_nhieu} ({g["A_tb"].max():.1f}); '
        f'ngày ra viện trung bình cao nhất: {ngay_ra_nhieu} ({g["D_tb"].max():.1f}); '
        f'Chủ nhật: nhập {g.loc[6, "A_tb"]:.1f}, ra {g.loc[6, "D_tb"]:.1f}.')

    # A5. Các hệ số tương quan và tỷ lệ cần điền
    r12 = df['Lag_1'].corr(df['Lag_2'])
    r1m = df['Lag_1'].corr(df['Rolling_Mean_7'])
    r2m = df['Lag_2'].corr(df['Rolling_Mean_7'])
    r_ea = df['Total_Elderly_Admitted'].corr(df['Admissions'])
    ty_le_cao_tuoi = df['Total_Elderly_Admitted'].sum() / df['Admissions'].sum() * 100
    ty_le_cc = (df['Emergency_Count'] / df['Census']).mean() * 100
    ghi(f'[Mục 2.2.5 – tương quan] r(Lag_1, Lag_2) = {r12:.3f}; r(Lag_1, Rolling_Mean_7) = {r1m:.3f}; '
        f'r(Lag_2, Rolling_Mean_7) = {r2m:.3f}; r(Total_Elderly_Admitted, Admissions) = {r_ea:.3f}; '
        f'người bệnh ≥ 60 tuổi chiếm {ty_le_cao_tuoi:.2f}% lượt nhập viện; Emergency_Count chiếm trung bình {ty_le_cc:.1f}% census.')

    # A6. Kiểm tra phạm vi giá trị: tập kiểm thử có vượt tập huấn luyện không?
    rows = []
    for c in ['Census', 'Lag_1', 'Lag_2', 'Lag_7', 'Rolling_Mean_7', 'Emergency_Count']:
        mx = train_df[c].max()
        rows.append({'Bien': c, 'Max_train': mx, 'Max_test': test_df[c].max(),
                     'Ty_le_ngay_test_vuot_max_train_%': float((test_df[c] > mx).mean() * 100)})
    kt = pd.DataFrame(rows)
    luu(kt, 'A6_kiem_tra_pham_vi_train_test.csv', index=False)
    print(kt.to_string(index=False))
    test_khong_tet = test_df[~test_df['Date'].between(pd.Timestamp(TET_BAT_DAU), pd.Timestamp(TET_KET_THUC))]
    ghi(f'[Mục 3.3 – kiểm tra phạm vi] Census trung bình: train = {train_df["Census"].mean():.1f}, '
        f'test (không tính kỳ Tết) = {test_khong_tet["Census"].mean():.1f}. '
        f'Tỷ lệ ngày kiểm thử có Census vượt giá trị lớn nhất của train ({train_df["Census"].max():.0f}) = '
        f'{(test_df["Census"] > train_df["Census"].max()).mean() * 100:.1f}%; '
        f'có Lag_1 vượt = {(test_df["Lag_1"] > train_df["Lag_1"].max()).mean() * 100:.1f}%.')

    # A7. Bảng chia tập (mục 3.1)
    rows = []
    for ten, d_ in [('Train (70%)', train_df), ('Validation (10%)', val_df), ('Test (20%)', test_df)]:
        rows.append({'Tap': ten, 'So_ngay': len(d_), 'Tu_ngay': d_['Date'].iloc[0].strftime('%d/%m/%Y'),
                     'Den_ngay': d_['Date'].iloc[-1].strftime('%d/%m/%Y'), 'So_mau_sau_cua_so_truot': len(d_) - WINDOW})
    cs = pd.DataFrame(rows)
    luu(cs, 'A7_chia_tap.csv', index=False)
    print(cs.to_string(index=False))
    for r in rows:
        ghi(f'[Mục 3.1 – chia tập] {r["Tap"]}: {r["So_ngay"]} ngày ({r["Tu_ngay"]} – {r["Den_ngay"]}); {r["So_mau_sau_cua_so_truot"]} mẫu sau cửa sổ trượt.')

    # A8. Tết: số ngày trong kỳ Tết định nghĩa ở trên
    ghi(f'[Bảng 3.1] Kỳ Tết được định nghĩa là {pd.Timestamp(TET_BAT_DAU).strftime("%d/%m/%Y")} – '
        f'{pd.Timestamp(TET_KET_THUC).strftime("%d/%m/%Y")}: {int(mask_tet.sum())} ngày kiểm thử; ngày thường: {int((~mask_tet).sum())} ngày.')


# =====================================================================
# PHẦN B0 – MÔ HÌNH NAIVE VÀ SEASONAL NAIVE
# =====================================================================
def phan_B0():
    DU_BAO['Naive: ŷ(t) = Census(t−1)'] = test_df['Lag_1'].values[WINDOW:].astype(float)
    DU_BAO['Seasonal naive: ŷ(t) = Census(t−7)'] = test_df['Lag_7'].values[WINDOW:].astype(float)
    for ten in ['Naive: ŷ(t) = Census(t−1)', 'Seasonal naive: ŷ(t) = Census(t−7)']:
        r = so_do(y_test, DU_BAO[ten])
        ghi(f'[Baseline] {ten}: MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}; MAPE = {r["MAPE"]:.2f}%; ME = {r["ME"]:.2f}')


# =====================================================================
# PHẦN B1 – ARIMA: chọn bậc + dự báo cuốn chiếu 1 bước (công bằng với LSTM)
# =====================================================================
def phan_B1():
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    census = df['Census'].values.astype(float)
    train = census[:TRAIN_END]

    adf_goc = adfuller(train)
    adf_sai_phan = adfuller(np.diff(train))
    ghi(f'[ARIMA – tính dừng, chỉ dùng tập train] ADF chuỗi census: p = {adf_goc[1]:.4f}; ADF sai phân bậc 1: p = {adf_sai_phan[1]:.4g}.')

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(np.diff(train), lags=30, ax=ax[0], title='ACF của chuỗi sai phân bậc 1 (train)')
    plot_pacf(np.diff(train), lags=30, ax=ax[1], method='ywm', title='PACF của chuỗi sai phân bậc 1 (train)')
    fig.tight_layout()
    fig.savefig(os.path.join(THU_MUC, 'B1_ACF_PACF.png'), dpi=300)
    plt.close(fig)

    rows = []
    for p in [0, 1, 2, 3, 7]:
        for q in [0, 1, 2]:
            try:
                r = ARIMA(train, order=(p, 1, q)).fit()
                rows.append({'Bac': f'({p},1,{q})', 'AIC': round(r.aic, 1), 'BIC': round(r.bic, 1)})
            except Exception as e:  # noqa
                rows.append({'Bac': f'({p},1,{q})', 'AIC': np.nan, 'BIC': np.nan})
    bang_aic = pd.DataFrame(rows).sort_values('AIC')
    luu(bang_aic, 'B1_chon_bac_ARIMA_theo_AIC.csv', index=False)
    print(bang_aic.to_string(index=False))
    ghi(f'[ARIMA – chọn bậc] Trong lưới p ∈ {{0,1,2,3,7}}, d = 1, q ∈ {{0,1,2}}, bậc có AIC thấp nhất trên tập train là '
        f'{bang_aic.iloc[0]["Bac"]} (AIC = {bang_aic.iloc[0]["AIC"]}); AIC của (7,1,1) = '
        f'{bang_aic[bang_aic["Bac"] == "(7,1,1)"]["AIC"].values[0]}.')

    # Dự báo cuốn chiếu 1 bước: tham số ước lượng CHỈ trên train, trạng thái cập nhật hằng ngày
    res = ARIMA(train, order=(7, 1, 1)).fit()
    res_day_du = res.append(census[TRAIN_END:], refit=False)
    du_bao_ca_doan = np.asarray(res_day_du.get_prediction(start=VAL_END, end=n - 1, dynamic=False).predicted_mean)
    DU_BAO['ARIMA(7,1,1) – cuốn chiếu 1 bước'] = du_bao_ca_doan[WINDOW:]
    r = so_do(y_test, DU_BAO['ARIMA(7,1,1) – cuốn chiếu 1 bước'])
    ghi(f'[ARIMA(7,1,1) cuốn chiếu 1 bước] MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}; MAPE = {r["MAPE"]:.2f}%; ME = {r["ME"]:.2f}')

    # Đối chiếu: cách tính cũ trong run_pipeline.py (dự báo nhiều bước liên tục, bắt đầu từ CUỐI TRAIN)
    cu = np.asarray(res.forecast(steps=len(test_df)))[WINDOW:]
    DU_BAO['ARIMA(7,1,1) – nhiều bước liên tục (cách cũ, CHỈ ĐỂ ĐỐI CHIẾU)'] = cu
    r = so_do(y_test, cu)
    ghi(f'[ARIMA(7,1,1) cách cũ – KHÔNG dùng để so sánh] MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}. '
        f'Dự báo cũ gần như đường phẳng: min = {cu.min():.1f}, max = {cu.max():.1f} giường.')


# =====================================================================
# PHẦN B2 – SARIMA và SARIMAX (chu kỳ tuần = 7)
# =====================================================================
def phan_B2():
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    census = df['Census'].values.astype(float)
    train = census[:TRAIN_END]
    ung_vien = [((1, 1, 1), (1, 0, 1, 7)), ((1, 1, 1), (0, 1, 1, 7)),
                ((2, 1, 2), (1, 0, 1, 7)), ((1, 1, 1), (1, 1, 1, 7))]
    rows = []
    tot_nhat = None
    for od, so in ung_vien:
        try:
            r = SARIMAX(train, order=od, seasonal_order=so).fit(disp=False, maxiter=200)
            rows.append({'order': str(od), 'seasonal_order': str(so), 'AIC': round(r.aic, 1)})
            if tot_nhat is None or r.aic < tot_nhat[0]:
                tot_nhat = (r.aic, od, so)
        except Exception as e:  # noqa
            rows.append({'order': str(od), 'seasonal_order': str(so), 'AIC': np.nan})
    bang = pd.DataFrame(rows).sort_values('AIC')
    luu(bang, 'B2_chon_bac_SARIMA_theo_AIC.csv', index=False)
    print(bang.to_string(index=False))
    if tot_nhat is None:
        raise RuntimeError('Không ước lượng được mô hình SARIMA nào.')
    _, od, so = tot_nhat
    ten_sarima = f'SARIMA{od}x{so[:3]}₇'
    ghi(f'[SARIMA – chọn bậc theo AIC trên train] {ten_sarima}')

    res = SARIMAX(train, order=od, seasonal_order=so).fit(disp=False, maxiter=200)
    res_full = res.append(census[TRAIN_END:], refit=False)
    pr = np.asarray(res_full.get_prediction(start=VAL_END, end=n - 1, dynamic=False).predicted_mean)[WINDOW:]
    DU_BAO[ten_sarima + ' – cuốn chiếu 1 bước'] = pr
    r = so_do(y_test, pr)
    ghi(f'[SARIMA cuốn chiếu 1 bước] MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}; MAPE = {r["MAPE"]:.2f}%; ME = {r["ME"]:.2f}')

    # SARIMAX có biến ngoại sinh biết trước: Is_Weekend, Is_Holiday
    ex = df[['Is_Weekend', 'Is_Holiday']].values.astype(float)
    resx = SARIMAX(train, exog=ex[:TRAIN_END], order=od, seasonal_order=so).fit(disp=False, maxiter=200)
    resx_full = resx.append(census[TRAIN_END:], exog=ex[TRAIN_END:], refit=False)
    prx = np.asarray(resx_full.get_prediction(start=VAL_END, end=n - 1, dynamic=False).predicted_mean)[WINDOW:]
    DU_BAO['SARIMAX (Is_Weekend, Is_Holiday) – cuốn chiếu 1 bước'] = prx
    r = so_do(y_test, prx)
    ghi(f'[SARIMAX cuốn chiếu 1 bước] MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}; MAPE = {r["MAPE"]:.2f}%; ME = {r["ME"]:.2f}')


# =====================================================================
# PHẦN B3 – XGBoost dự báo trực tiếp (không qua LSTM)
# =====================================================================
def phan_B3():
    import xgboost as xgb

    def xs_ys(d_):
        return d_[X_COLS].iloc[WINDOW - 1:-1].reset_index(drop=True).round(2), d_['Census'].iloc[WINDOW:].values.astype(float)

    Xtr, ytr = xs_ys(train_df)
    Xva, yva = xs_ys(val_df)
    Xte, yte = xs_ys(test_df)
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=30, eval_metric='rmse', random_state=42)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pr = m.predict(Xte)
    DU_BAO['XGBoost dự báo trực tiếp'] = pr
    r = so_do(y_test, pr)
    ghi(f'[XGBoost trực tiếp] MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}; MAPE = {r["MAPE"]:.2f}%; ME = {r["ME"]:.2f}')




# =====================================================================
# PHẦN C – LSTM: CHỌN BIẾN THỂ TRÊN TẬP KIỂM ĐỊNH, CHẤM TRÊN TẬP KIỂM THỬ MỘT LẦN
# =====================================================================
def make_sequences(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)



# Biến thể ứng viên (cố định trước):
#   census_t : thêm Census(t-1) làm đặc trưng
#   lich     : 'onehot' = thứ trong tuần của ngày CẦN DỰ BÁO (7 cột 0/1) + ngày lễ của ngày đó; 'int' = thứ dạng số 0-6
#   skip     : True = kết nối tắt: đầu ra = Census(t-1) + phần mạng học được (mạng chỉ học PHẦN THAY ĐỔI so với hôm qua)
#   u1,u2,drop : số đơn vị LSTM và dropout.  Lịch của ngày dự báo là thông tin biết trước nên dùng hợp lệ.
UNG_VIEN = [
    ('K0: 17 đặc trưng, không kết nối tắt', dict(census_t=False, lich=None, skip=False, u1=64, u2=32, drop=0.2)),
    ('K1: + Census(t−1)', dict(census_t=True, lich=None, skip=False, u1=64, u2=32, drop=0.2)),
    ('K2: + Census(t−1), có kết nối tắt', dict(census_t=True, lich=None, skip=True, u1=64, u2=32, drop=0.2)),
    ('K3: + Census(t−1) + lịch ngày dự báo (one-hot)', dict(census_t=True, lich='onehot', skip=False, u1=64, u2=32, drop=0.2)),
    ('K4: + Census(t−1) + lịch ngày dự báo (one-hot), có kết nối tắt', dict(census_t=True, lich='onehot', skip=True, u1=64, u2=32, drop=0.2)),
    ('K5: + Census(t−1) + lịch ngày dự báo (số nguyên), có kết nối tắt', dict(census_t=True, lich='int', skip=True, u1=64, u2=32, drop=0.2)),
    ('K6: như K4 nhưng mạng nhỏ (32/16, dropout 0.1)', dict(census_t=True, lich='onehot', skip=True, u1=32, u2=16, drop=0.1)),
]

MODEL_42 = HIST_42 = DU_CHON = CFG_CHON = TEN_CHON = None


def them_dac_trung(d, cfg):
    """Thêm các đặc trưng mới vào bảng 17 đặc trưng cơ bản. Trả về (bảng mới, danh sách cột đầu vào)."""
    d = d.copy()
    them = []
    if cfg.get('census_t'):
        d['Census_t'] = d['Census']          # tại hàng ngày d là Census(d); ở bước cuối cửa sổ (t-1) là Census(t-1)
        them.append('Census_t')
    dow_next = ((d['DayOfWeek'] + 1) % 7).astype(int)
    if cfg.get('lich') == 'onehot':
        for k in range(7):
            d[f'DOW_next_{k}'] = (dow_next == k).astype(int)
            them.append(f'DOW_next_{k}')
    elif cfg.get('lich') == 'int':
        d['DayOfWeek_next'] = dow_next
        them.append('DayOfWeek_next')
    if cfg.get('lich'):
        d['Is_Holiday_next'] = d['Is_Holiday'].shift(-1).fillna(0).astype(int)
        them.append('Is_Holiday_next')
    return d, list(X_COLS) + them


def chuan_bi(cfg):
    d, cols = them_dac_trung(df, cfg)
    tr, va, te = d.iloc[:TRAIN_END], d.iloc[TRAIN_END:VAL_END], d.iloc[VAL_END:]
    sx, sy = MinMaxScaler((0, 1)), MinMaxScaler((0, 1))
    sx.fit(tr[cols])
    sy.fit(tr[['Census']])

    def seq(dd):
        return make_sequences(sx.transform(dd[cols]), sy.transform(dd[['Census']]))

    Xtr, ytr = seq(tr)
    Xva, yva = seq(va)
    Xte, yte = seq(te)
    return dict(cols=cols, sy=sy, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte,
                y_val=va['Census'].values[WINDOW:].astype(float), tr=tr, va=va, te=te)


def _huan_luyen_keras(Xtr, ytr, Xva, yva, X_du_doan, cfg, seed, chi_so_skip, giu_mo_hinh=False):
    """Kiến trúc: LSTM -> Dropout -> LSTM -> Dropout -> Dense16 -> Dense1.
    Nếu cfg['skip']: đầu ra = Census(t-1) (chuẩn hóa, lấy ở bước cuối cửa sổ) + phần mạng học được."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Lambda, Add
    from tensorflow.keras.callbacks import EarlyStopping

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    inp = Input(shape=(Xtr.shape[1], Xtr.shape[2]))
    x = LSTM(cfg['u1'], return_sequences=True)(inp)
    x = Dropout(cfg['drop'])(x)
    x = LSTM(cfg['u2'], return_sequences=False)(x)
    x = Dropout(cfg['drop'])(x)
    x = Dense(16, activation='relu')(x)
    g = Dense(1, activation='linear')(x)
    if cfg['skip']:
        k = int(chi_so_skip)
        tat = Lambda(lambda t: t[:, -1, k:k + 1])(inp)
        out = Add()([tat, g])
    else:
        out = g
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    h = model.fit(Xtr, ytr, epochs=150, batch_size=16, validation_data=(Xva, yva),
                  callbacks=[es], shuffle=False, verbose=0)
    kq = [model.predict(X, verbose=0).ravel() for X in X_du_doan]
    if giu_mo_hinh:
        return kq, len(h.history['loss']), model, h.history
    return kq, len(h.history['loss']), None, None


def huan_luyen(du_lieu, cfg, seed, du_doan_test, giu_mo_hinh=False):
    """du_doan_test = False: chỉ dự báo tập Kiểm định (dùng khi chọn biến thể)."""
    chi_so = du_lieu['cols'].index('Census_t') if 'Census_t' in du_lieu['cols'] else 0
    X_list = [du_lieu['Xva']] + ([du_lieu['Xte']] if du_doan_test else [])
    kq, so_epoch, mdl, hist = _huan_luyen_keras(du_lieu['Xtr'], du_lieu['ytr'], du_lieu['Xva'], du_lieu['yva'],
                                                X_list, cfg, seed, chi_so, giu_mo_hinh)
    sy = du_lieu['sy']
    p_val = sy.inverse_transform(kq[0].reshape(-1, 1)).ravel()
    p_test = sy.inverse_transform(kq[1].reshape(-1, 1)).ravel() if du_doan_test else None
    return p_val, p_test, so_epoch, mdl, hist


def chon_bien_the():
    y_val = val_df['Census'].values[WINDOW:].astype(float)
    mae_naive_val = float(np.mean(np.abs(val_df['Lag_1'].values[WINDOW:].astype(float) - y_val)))
    ghi(f'[Chọn biến thể – tập Kiểm định, {len(y_val)} mẫu] MAE của naive Census(t−1) trên tập Kiểm định = {mae_naive_val:.2f}')
    rows = []
    for ten, cfg in UNG_VIEN:
        du_lieu = chuan_bi(cfg)
        maes, mes = [], []
        t0 = time.time()
        for s in SEED_CHON:
            p_val, _, _, _, _ = huan_luyen(du_lieu, cfg, s, du_doan_test=False)
            e = p_val - du_lieu['y_val']
            maes.append(float(np.mean(np.abs(e))))
            mes.append(float(np.mean(e)))
        rows.append({'Bien_the': ten, 'So_dac_trung': len(du_lieu['cols']), 'Val_MAE_TB': np.mean(maes),
                     'Val_MAE_SD': np.std(maes, ddof=1), 'Val_ME_TB': np.mean(mes)})
        print(f'  {ten}: Val MAE = {np.mean(maes):.2f} ± {np.std(maes, ddof=1):.2f} ({time.time() - t0:.0f} giây)')
    bang = pd.DataFrame(rows)
    luu(bang.round(3), 'C0_chon_bien_the_tren_tap_kiem_dinh.csv', index=False)
    print(bang.round(2).to_string(index=False))
    i_tot = int(bang['Val_MAE_TB'].idxmin())
    ten_tot, cfg_tot = UNG_VIEN[i_tot]
    ghi(f'[Chọn biến thể] Biến thể được chọn (MAE tập Kiểm định thấp nhất = {bang.loc[i_tot, "Val_MAE_TB"]:.2f}): {ten_tot} '
        f'({int(bang.loc[i_tot, "So_dac_trung"])} đặc trưng)')
    with open(os.path.join(THU_MUC, 'cau_hinh_duoc_chon.json'), 'w', encoding='utf-8') as f:
        json.dump({'ten': ten_tot, **cfg_tot}, f, ensure_ascii=False, indent=2)
    return ten_tot, cfg_tot


def chay_10_seed(du_lieu, cfg, ten_nhom, giu_seed42):
    """Huấn luyện SEED_CUOI, dự báo tập Kiểm thử. Trả về (dict dự báo, dict số epoch, mô hình seed 42, lịch sử huấn luyện seed 42)."""
    du_doan, so_epoch, mdl42, hist42 = {}, {}, None, None
    for s in SEED_CUOI:
        t0 = time.time()
        _, p_test, so_epoch[s], mdl, hist = huan_luyen(du_lieu, cfg, s, True, giu_mo_hinh=(giu_seed42 and s == 42))
        du_doan[s] = p_test
        if giu_seed42 and s == 42:
            mdl42, hist42 = mdl, hist
        print(f'  [{ten_nhom}] seed {s}: {time.time() - t0:.0f} giây, dừng ở epoch {so_epoch[s]}')
    return du_doan, so_epoch, mdl42, hist42


def bang_theo_seed(du_doan, so_epoch, ten_file):
    b = pd.DataFrame([{**so_do(y_test, p), 'seed': s, 'so_epoch': so_epoch[s]} for s, p in du_doan.items()]).set_index('seed')
    luu(b.round(3), ten_file)
    print(b.round(2).to_string())
    return b, b[['MAE', 'RMSE', 'MAPE', 'ME']].mean(), b[['MAE', 'RMSE', 'MAPE', 'ME']].std(ddof=1)


def phan_lstm():
    global MODEL_42, HIST_42, DU_CHON, CFG_CHON, TEN_CHON
    TEN_CHON, CFG_CHON = chon_bien_the()
    DU_CHON = chuan_bi(CFG_CHON)

    # --- LSTM đã chọn: 10 seed, chấm một lần trên tập Kiểm thử ---
    du_doan, so_epoch, MODEL_42, HIST_42 = chay_10_seed(DU_CHON, CFG_CHON, 'LSTM', True)
    PRED_CHON.update(du_doan)
    b, m, sd = bang_theo_seed(du_doan, so_epoch, 'C1_LSTM_theo_seed.csv')
    ghi(f'[LSTM – {len(b)} seed, chấm một lần trên tập Kiểm thử] MAE = {m["MAE"]:.2f} ± {sd["MAE"]:.2f}; '
        f'RMSE = {m["RMSE"]:.2f} ± {sd["RMSE"]:.2f}; MAPE = {m["MAPE"]:.2f} ± {sd["MAPE"]:.2f}%; ME = {m["ME"]:.2f} ± {sd["ME"]:.2f}; '
        f'MAE nhỏ nhất/lớn nhất giữa các seed = {b["MAE"].min():.2f}/{b["MAE"].max():.2f}.')
    ghi(f'[Mô hình LSTM seed 42] số tham số = {MODEL_42.count_params()}; dừng sớm sau {so_epoch[42]} epoch; '
        f'epoch tốt nhất = {int(np.argmin(HIST_42["val_loss"])) + 1}; số đặc trưng = {len(DU_CHON["cols"])}.')
    pd.DataFrame({'Date': dates_test, 'Census_thuc_te': y_test, **{f'seed_{s}': p for s, p in du_doan.items()}}).to_csv(
        os.path.join(THU_MUC, 'C1_du_bao_LSTM_theo_seed.csv'), index=False, encoding='utf-8-sig')

    # --- Hình 1 (đường cong mất mát) và Hình 2 (dự báo so với thực tế, 193 ngày) ---
    plt.figure(figsize=(8, 6))
    plt.plot(HIST_42['loss'], label='Sai số tập Train')
    plt.plot(HIST_42['val_loss'], label='Sai số tập Validation')
    plt.title('Quá trình huấn luyện LSTM (seed 42)', pad=25)
    plt.xlabel('Vòng lặp (Epochs)')
    plt.ylabel('Sai số (MSE Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(HINH + '/Hinh_1_LSTM_Loss_Curve.png', dpi=300)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(dates_test, y_test, label='Thực tế', linestyle='dashed', marker='o', markersize=3)
    plt.plot(dates_test, PRED_CHON[42], label='Dự báo (LSTM, seed 42)', marker='x', markersize=4, alpha=0.8)
    plt.axvspan(pd.Timestamp(TET_BAT_DAU), pd.Timestamp(TET_KET_THUC), color='orange', alpha=0.2, label='Kỳ Tết Nguyên đán')
    plt.title('Dự báo so với Thực tế trên toàn bộ tập Kiểm thử (193 ngày: 20/01/2026 - 31/07/2026)', pad=25)
    plt.xlabel('Thời gian')
    plt.ylabel('Số lượng giường bệnh')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.xticks(rotation=45)
    plt.gca().set_xlim(dates_test[0], dates_test[-1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(HINH + '/Hinh_2_LSTM_Forecast_vs_Actual.png', dpi=300)
    plt.close()


# =====================================================================
# PHẦN D – BẢNG 3.1, BẢNG 3.2, DIEBOLD–MARIANO, HÌNH 3.3
# =====================================================================
def xgboost_cung_dac_trung(cfg):
    import xgboost as xgb
    d, cols = them_dac_trung(df, cfg)
    tr, va, te = d.iloc[:TRAIN_END], d.iloc[TRAIN_END:VAL_END], d.iloc[VAL_END:]

    def xs_ys(dd):
        return dd[cols].iloc[WINDOW - 1:-1].reset_index(drop=True), dd['Census'].iloc[WINDOW:].values.astype(float)

    Xtr, ytr = xs_ys(tr)
    Xva, yva = xs_ys(va)
    Xte, _ = xs_ys(te)
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=30, eval_metric='rmse', random_state=42)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict(Xte)


def phan_so_sanh():
    if 42 not in PRED_CHON:
        raise RuntimeError('Chưa có dự báo LSTM (phần C chưa chạy xong).')
    y = y_test
    p42 = PRED_CHON[42]

    # Bảng 3.1: theo giai đoạn
    cac = [bang_giai_doan(p).set_index('Giai_doan') for p in PRED_CHON.values()]
    tb = sum(cac) / len(cac)
    luu(tb.round(2), 'D1_Bang_3_1_LSTM_theo_giai_doan_TB_cac_seed.csv')
    luu(bang_giai_doan(p42).round(2), 'D1_Bang_3_1_LSTM_seed42_theo_giai_doan.csv', index=False)
    print(tb.round(2).to_string())
    for gd, r in tb.iterrows():
        ghi(f'[Bảng 3.1 – LSTM, TB các seed] {gd}: {int(r["So_ngay"])} ngày; MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}; '
            f'MAPE = {r["MAPE"]:.2f}%; ME = {r["ME"]:.2f}')

    # Bảng 3.2: so sánh với các mô hình cơ sở
    du_bao = {k: v for k, v in DU_BAO.items() if not k.startswith('__') and 'CHỈ ĐỂ ĐỐI CHIẾU' not in k}
    try:
        du_bao['XGBoost dự báo trực tiếp + 26 biến đầu vào của LSTM'] = xgboost_cung_dac_trung(CFG_CHON)
    except Exception as e:  # noqa
        print(f'[CẢNH BÁO] Không chạy được XGBoost cùng bộ biến đầu vào: {type(e).__name__}: {e}')
    mae42 = so_do(y, p42)['MAE']

    def nhan_ket_luan(p_val, mae_mo_hinh):
        if p_val < 0.05 and mae_mo_hinh > mae42:
            return 'LSTM thấp hơn có ý nghĩa thống kê'
        if p_val < 0.05 and mae_mo_hinh < mae42:
            return 'LSTM cao hơn có ý nghĩa thống kê'
        return 'Không khác biệt có ý nghĩa thống kê'

    rows = []
    for ten, p in du_bao.items():
        r = so_do(y, p)
        _, p2 = dm_test(y, p, p42, 2)
        _, p1 = dm_test(y, p, p42, 1)
        rows.append({'Mo_hinh': ten, 'MAE': r['MAE'], 'RMSE': r['RMSE'], 'MAPE_%': r['MAPE'], 'ME': r['ME'],
                     'DM_p_binh_phuong': p2, 'DM_p_tuyet_doi': p1,
                     'Ket_luan_binh_phuong': nhan_ket_luan(p2, r['MAE']), 'Ket_luan_tuyet_doi': nhan_ket_luan(p1, r['MAE'])})
    r42 = so_do(y, p42)
    rows.append({'Mo_hinh': 'LSTM – seed 42', 'MAE': r42['MAE'], 'RMSE': r42['RMSE'], 'MAPE_%': r42['MAPE'], 'ME': r42['ME'],
                 'DM_p_binh_phuong': np.nan, 'DM_p_tuyet_doi': np.nan, 'Ket_luan_binh_phuong': '', 'Ket_luan_tuyet_doi': ''})
    bb = pd.DataFrame([so_do(y, p) for p in PRED_CHON.values()])
    m, sd = bb.mean(), bb.std(ddof=1)
    rows.append({'Mo_hinh': f'LSTM – trung bình ± SD ({len(bb)} seed)', 'MAE': f'{m["MAE"]:.2f} ± {sd["MAE"]:.2f}',
                 'RMSE': f'{m["RMSE"]:.2f} ± {sd["RMSE"]:.2f}', 'MAPE_%': f'{m["MAPE"]:.2f} ± {sd["MAPE"]:.2f}',
                 'ME': f'{m["ME"]:.2f} ± {sd["ME"]:.2f}', 'DM_p_binh_phuong': np.nan, 'DM_p_tuyet_doi': np.nan,
                 'Ket_luan_binh_phuong': '', 'Ket_luan_tuyet_doi': ''})
    b32 = pd.DataFrame(rows)
    luu(b32.round(4), 'D2_Bang_3_2_so_sanh_mo_hinh.csv', index=False)
    print(b32.round(3).to_string(index=False))
    for r in rows:
        if isinstance(r['MAE'], float):
            dm = (f'; DM p (bình phương) = {r["DM_p_binh_phuong"]:.4f}, DM p (tuyệt đối) = {r["DM_p_tuyet_doi"]:.4f}'
                  if not np.isnan(r['DM_p_binh_phuong']) else '')
            ghi(f'[Bảng 3.2] {r["Mo_hinh"]}: MAE = {r["MAE"]:.2f}; RMSE = {r["RMSE"]:.2f}; MAPE = {r["MAPE_%"]:.2f}%; ME = {r["ME"]:.2f}{dm}')
        else:
            ghi(f'[Bảng 3.2] {r["Mo_hinh"]}: MAE = {r["MAE"]}; RMSE = {r["RMSE"]}; MAPE = {r["MAPE_%"]}%; ME = {r["ME"]}')

    n_ss = len(du_bao)
    ghi(f'[Ghi chú kiểm định] {n_ss} phép so sánh cùng lúc: ngưỡng Bonferroni = 0.05/{n_ss} = {0.05 / n_ss:.4f}.')
    ghi('\n[TỔNG KẾT – viết kết luận đúng như dưới đây]')
    for tieu_de, khoa in [('hàm tổn thất bình phương', 'Ket_luan_binh_phuong'), ('sai số tuyệt đối', 'Ket_luan_tuyet_doi')]:
        thap = [r['Mo_hinh'] for r in rows if str(r[khoa]).startswith('LSTM thấp hơn')]
        cao = [r['Mo_hinh'] for r in rows if str(r[khoa]).startswith('LSTM cao hơn')]
        hoa = [r['Mo_hinh'] for r in rows if str(r[khoa]).startswith('Không khác biệt')]
        ghi(f'  Theo {tieu_de}:')
        ghi('    LSTM thấp hơn có ý nghĩa thống kê: ' + ('; '.join(thap) if thap else '(không mô hình nào)'))
        ghi('    Không khác biệt có ý nghĩa thống kê: ' + ('; '.join(hoa) if hoa else '(không mô hình nào)'))
        ghi('    LSTM cao hơn có ý nghĩa thống kê: ' + ('; '.join(cao) if cao else '(không mô hình nào)'))

    # Hình 3: so sánh MAE và RMSE giữa các mô hình (LSTM lấy trung bình các seed)
    ten_h, mae_l, rmse_l = [], [], []
    rut_gon = {'Naive: ŷ(t) = Census(t−1)': 'Naive t−1', 'Seasonal naive: ŷ(t) = Census(t−7)': 'Naive mùa t−7',
               'XGBoost dự báo trực tiếp + 26 biến đầu vào của LSTM': 'XGBoost + 26 biến',
               'XGBoost dự báo trực tiếp': 'XGBoost trực tiếp'}
    for ten, p in du_bao.items():
        r = so_do(y, p)
        ngan = ten.split(' – ')[0]
        ten_h.append(rut_gon.get(ngan, ngan))
        mae_l.append(r['MAE'])
        rmse_l.append(r['RMSE'])
    ten_h.append(f'LSTM (TB {len(bb)} seed)')
    mae_l.append(m['MAE'])
    rmse_l.append(m['RMSE'])
    x = np.arange(len(ten_h))
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - 0.2, mae_l, 0.4, label='MAE', color='#5CB85C')
    b2 = ax.bar(x + 0.2, rmse_l, 0.4, label='RMSE', color='#D9534F')
    ax.bar_label(b1, fmt='%.1f', fontsize=8)
    ax.bar_label(b2, fmt='%.1f', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ten_h, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Sai số (giường)')
    ax.set_title('So sánh sai số dự báo 1 ngày giữa các mô hình trên tập kiểm thử')
    ax.legend()
    fig.tight_layout()
    fig.savefig(HINH + '/Hinh_3_So_sanh_cac_mo_hinh.png', dpi=300)
    plt.close(fig)


# =====================================================================
# PHẦN E – GIẢI THÍCH LSTM ĐÃ CHỌN (seed 42): MÔ HÌNH ĐẠI DIỆN XGBOOST, TREESHAP, PERMUTATION, IG
# =====================================================================
def phan_xai():
    import tensorflow as tf
    import shap
    import xgboost as xgb

    if MODEL_42 is None:
        raise RuntimeError('Chưa có mô hình LSTM seed 42 (phần C chưa chạy xong).')
    SEED = 42
    model = MODEL_42
    X_columns = DU_CHON['cols']
    X_train_lstm, X_val_lstm, X_test_lstm = DU_CHON['Xtr'], DU_CHON['Xva'], DU_CHON['Xte']
    scaler_Y = DU_CHON['sy']
    train_df, val_df, test_df = DU_CHON['tr'], DU_CHON['va'], DU_CHON['te']
    window = WINDOW
    dates_test = test_df['Date'].values[window:]
    Y_test_real = y_test.reshape(-1, 1)
    predictions_real = PRED_CHON[42].reshape(-1, 1)
    print("\nBƯỚC 5: Huấn luyện Surrogate XGBoost trên nhãn giả đơn vị GIƯỜNG và đặc trưng GỐC...")

    # 1. Sinh nhãn giả từ LSTM và đảo chuẩn hóa (inverse_transform) về đơn vị GIƯỜNG BỆNH
    Y_train_pseudo_beds = scaler_Y.inverse_transform(model.predict(X_train_lstm, verbose=0)).ravel()
    Y_val_pseudo_beds   = scaler_Y.inverse_transform(model.predict(X_val_lstm,   verbose=0)).ravel()
    Y_test_pseudo_beds  = scaler_Y.inverse_transform(model.predict(X_test_lstm,  verbose=0)).ravel()

    # 2. Lấy đặc trưng ở ĐƠN VỊ GỐC (số ca, tuổi, thứ...) tại bước cuối cửa sổ trượt (t-1)
    X_train_xgb_raw = train_df[X_columns].iloc[window - 1 : -1].reset_index(drop=True).round(2)
    X_val_xgb_raw   = val_df[X_columns].iloc[window - 1 : -1].reset_index(drop=True).round(2)
    X_test_xgb_raw  = test_df[X_columns].iloc[window - 1 : -1].reset_index(drop=True).round(2)

    assert len(X_train_xgb_raw) == len(Y_train_pseudo_beds), "Lỗi lệch dòng tập Train!"
    assert len(X_val_xgb_raw)   == len(Y_val_pseudo_beds),   "Lỗi lệch dòng tập Validation!"
    assert len(X_test_xgb_raw)  == len(Y_test_pseudo_beds) == len(dates_test), "Lỗi lệch dòng tập Test!"

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric='rmse',
        random_state=SEED
    )

    # Huấn luyện trên đơn vị giường và dừng sớm độc lập trên tập Validation
    xgb_model.fit(
        X_train_xgb_raw, Y_train_pseudo_beds,
        eval_set=[(X_train_xgb_raw, Y_train_pseudo_beds), (X_val_xgb_raw, Y_val_pseudo_beds)],
        verbose=False
    )

    xgb_pred_train = xgb_model.predict(X_train_xgb_raw)
    xgb_pred_val   = xgb_model.predict(X_val_xgb_raw)
    xgb_pred_test  = xgb_model.predict(X_test_xgb_raw)

    fidelity_train = r2_score(Y_train_pseudo_beds, xgb_pred_train)
    fidelity_val   = r2_score(Y_val_pseudo_beds,   xgb_pred_val)
    fidelity_test  = r2_score(Y_test_pseudo_beds,  xgb_pred_test)

    print(f"--> Fidelity R2 | Train: {fidelity_train:.4f} | Val: {fidelity_val:.4f} | Test: {fidelity_test:.4f}")

    # [BỔ SUNG v2] Sai số giữa XGBoost và LSTM (đơn vị giường) và R² riêng cho dải 650–900 giường
    _mae_fid = mean_absolute_error(Y_test_pseudo_beds, xgb_pred_test)
    _rmse_fid = float(np.sqrt(mean_squared_error(Y_test_pseudo_beds, xgb_pred_test)))
    _mask = (Y_test_pseudo_beds >= 650) & (Y_test_pseudo_beds <= 900)
    _r2_dai = r2_score(Y_test_pseudo_beds[_mask], xgb_pred_test[_mask])
    print(f"--> Độ trung thành (Test): MAE(XGB vs LSTM) = {_mae_fid:.2f} giường | RMSE = {_rmse_fid:.2f} giường | R² dải 650–900 = {_r2_dai:.4f} (n = {int(_mask.sum())} ngày)")
    pd.Series({'MAE_XGB_vs_LSTM_test': _mae_fid, 'RMSE_XGB_vs_LSTM_test': _rmse_fid,
               'R2_dai_650_900_test': _r2_dai, 'so_ngay_trong_dai_650_900': int(_mask.sum()),
               'R2_train': fidelity_train, 'R2_val': fidelity_val, 'R2_test': fidelity_test}).to_csv(
        THU_MUC + '/surrogate_fidelity_bo_sung.csv', header=['gia_tri'], encoding='utf-8-sig')

    # --- Xuất Hình 4: Biểu đồ phân tán (Scatter Plot) theo đơn vị giường bệnh ---
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(Y_test_pseudo_beds, xgb_pred_test, color='#9B59B6', edgecolor='white', s=45, alpha=0.85)
    min_val = min(Y_test_pseudo_beds.min(), xgb_pred_test.min())
    max_val = max(Y_test_pseudo_beds.max(), xgb_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Đường chuẩn y=x')
    plt.title(f'Độ trung thành của XGBoost so với LSTM ($R^2 = {fidelity_test:.4f}$)',pad=25)
    plt.xlabel('Dự báo của mạng học sâu LSTM (Số giường bệnh)')
    plt.ylabel('Dự báo của XGBoost Surrogate (Số giường bệnh)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HINH + '/Hinh_4_XGBoost_Fidelity_Scatter.png', dpi=300)
    plt.close()

    from scipy.stats import spearmanr

    # --- Kiểm chứng độc lập trực tiếp trên mạng LSTM (Permutation Importance & IG) ---
    rng = np.random.RandomState(SEED)
    Y_test_pseudo_scaled = model.predict(X_test_lstm, verbose=0).ravel()

    lstm_perm_imp = {}
    for j, col in enumerate(X_columns):
        diffs = []
        for _ in range(20):
            perm_idx = rng.permutation(len(X_test_lstm))
            X_perm = X_test_lstm.copy()
            X_perm[:, -1, j] = X_perm[perm_idx, -1, j]
            pred_perm = model.predict(X_perm, verbose=0).ravel()
            diffs.append(mean_absolute_error(Y_test_pseudo_scaled, pred_perm))
        lstm_perm_imp[col] = np.mean(diffs)
    lstm_perm_series = pd.Series(lstm_perm_imp)

    def compute_lstm_ig(lstm_model, X_input, baseline, steps=50):
        X_tf = tf.cast(X_input, tf.float32)
        base_tf = tf.cast(baseline, tf.float32)
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        grads = []
        for alpha in alphas:
            interp = base_tf + alpha * (X_tf - base_tf)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                out = lstm_model(interp, training=False)
            grads.append(tape.gradient(out, interp).numpy())
        avg_grads = np.mean(grads[:-1] + grads[1:], axis=0) / 2.0
        return (X_input - baseline) * avg_grads

    baseline_lstm = np.mean(X_train_lstm, axis=0, keepdims=True)
    ig_values = compute_lstm_ig(model, X_test_lstm, baseline=baseline_lstm, steps=50)
    lstm_ig_series = pd.Series(np.mean(np.abs(ig_values[:, -1, :]), axis=0), index=X_columns)

    rho_ig_perm, p_ig_perm = spearmanr(lstm_ig_series, lstm_perm_series)
    print(f"--> Tương quan hạng Spearman (LSTM IG vs LSTM Permutation): rho = {rho_ig_perm:.4f} (p = {p_ig_perm:.2e})")

    # =====================================================================
    # BƯỚC 6: TREESHAP THEO ĐƠN VỊ GIƯỜNG BỆNH & ĐẶC TRƯNG GỐC (HÌNH 5 -> 9)
    # =====================================================================
    print("\nBƯỚC 6: Xuất Hình 5 -> Hình 9 theo đơn vị GIƯỜNG BỆNH và giá trị đặc trưng GỐC...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values_beds = explainer.shap_values(X_test_xgb_raw)

    # [BỔ SUNG v2] Bảng 3.4: thứ hạng tầm quan trọng của 17 đặc trưng theo SHAP, Permutation và IG
    _shap_mean_abs = pd.Series(np.abs(shap_values_beds).mean(axis=0), index=X_columns)
    bang_3_4 = pd.DataFrame({
        'SHAP_mean_abs_giuong': _shap_mean_abs.round(2),
        'Hang_SHAP': _shap_mean_abs.rank(ascending=False, method='min').astype(int),
        'Permutation_MAE_thang_chuan_hoa': lstm_perm_series.round(5),
        'Hang_Permutation': lstm_perm_series.rank(ascending=False, method='min').astype(int),
        'IG_mean_abs': lstm_ig_series.round(5),
        'Hang_IG': lstm_ig_series.rank(ascending=False, method='min').astype(int),
    }).sort_values('Hang_SHAP')
    bang_3_4.to_csv(THU_MUC + '/Bang_3_4_hang_tam_quan_trong.csv', encoding='utf-8-sig')
    print('\nBảng 3.4 (thứ hạng theo SHAP / Permutation / IG):')
    print(bang_3_4.to_string())
    _r1, _p1 = spearmanr(bang_3_4['SHAP_mean_abs_giuong'], bang_3_4['Permutation_MAE_thang_chuan_hoa'])
    _r2, _p2 = spearmanr(bang_3_4['SHAP_mean_abs_giuong'], bang_3_4['IG_mean_abs'])
    print(f"--> Spearman (SHAP vs Permutation): rho = {_r1:.4f} (p = {_p1:.2e}) | (SHAP vs IG): rho = {_r2:.4f} (p = {_p2:.2e}) | (IG vs Permutation): rho = {rho_ig_perm:.4f}")

    # [BỔ SUNG v2] SHAP theo NHÓM biến tương quan cao: cộng SHAP (có dấu) của các biến trong nhóm rồi lấy trung bình trị tuyệt đối
    _nhom = {
        'Quán tính census (Census_t nếu có, Lag_1, Lag_2, Lag_7, Rolling_Mean_7)': ['Lag_1', 'Lag_2', 'Lag_7', 'Rolling_Mean_7'] + [c for c in X_columns if c == 'Census_t'],
        'Luồng nhập viện (Admissions, Total_Elderly_Admitted)': ['Admissions', 'Total_Elderly_Admitted'],
        'Nhóm bệnh ICD-10 (Total_Cardiovascular, Total_Respiratory, Total_Infectious, Avg_ICD_Sub_Count)':
            ['Total_Cardiovascular', 'Total_Respiratory', 'Total_Infectious', 'Avg_ICD_Sub_Count'],
        'Lịch của ngày t−1 (DayOfWeek, Is_Weekend, Month, Is_Holiday)': ['DayOfWeek', 'Is_Weekend', 'Month', 'Is_Holiday'],
        'Lịch của ngày dự báo t (nếu có)': [c for c in X_columns if c.startswith('DOW_next_') or c in ('DayOfWeek_next', 'Is_Holiday_next')],
    }
    _rows = []
    for _ten, _cols in _nhom.items():
        if not _cols:
            continue
        _ids = [X_columns.index(c) for c in _cols]
        _rows.append({'Nhom_bien': _ten,
                      'mean_abs_SHAP_cua_tong_nhom_giuong': round(float(np.abs(shap_values_beds[:, _ids].sum(axis=1)).mean()), 2),
                      'tong_mean_abs_tung_bien_giuong': round(float(np.abs(shap_values_beds[:, _ids]).mean(axis=0).sum()), 2)})
    _bang_nhom = pd.DataFrame(_rows)
    _bang_nhom.to_csv(THU_MUC + '/Bang_SHAP_theo_nhom_bien.csv', index=False, encoding='utf-8-sig')
    print('\nSHAP theo nhóm biến tương quan cao:'); print(_bang_nhom.to_string(index=False))

    # --- Xuất Hình 5: SHAP Feature Importance (Đơn vị: Giường bệnh) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_beds, X_test_xgb_raw, plot_type="bar", show=False)
    plt.xlabel("mean(|SHAP value|) - Mức tác động trung bình (Số lượng giường bệnh)", fontsize=11)
    ax = plt.gca()
    for patch in ax.patches:
        w = patch.get_width()
        if w > 0:
            ax.text(w + 0.3, patch.get_y() + patch.get_height() / 2, f'{w:.2f} giường',
                    ha='left', va='center', fontsize=9.5, color='black')
    plt.title("Tầm quan trọng của các yếu tố ảnh hưởng đến nhu cầu giường bệnh", pad=20)
    plt.margins(y=0.02)
    plt.xlim(right=ax.get_xlim()[1] * 1.18)
    plt.tight_layout()
    plt.savefig(HINH + '/Hinh_5_SHAP_Feature_Importance.png', dpi=300)
    plt.close()

    # --- Xuất Hình 6: SHAP Beeswarm Plot (Đơn vị: Giường bệnh) ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_beds, X_test_xgb_raw, show=False)
    plt.xlabel("Giá trị SHAP (Mức độ làm Tăng/Giảm số lượng giường bệnh)")
    plt.title("Biểu đồ bầy đàn SHAP: Phân tích chiều tác động (Đơn vị: Giường bệnh)")
    plt.tight_layout()
    plt.savefig(HINH + '/Hinh_6_SHAP_Beeswarm.png', dpi=300)
    plt.close()

    # --- Xuất Hình 7, 8, 9: SHAP Waterfall hiển thị ĐƠN VỊ GIƯỜNG và ĐẶC TRƯNG GỐC ---
    dates_test_dt = pd.to_datetime(dates_test)
    waterfall_configs = [
        ('2026-07-12', 'Chủ nhật', 'Hinh_7_SHAP_Waterfall_2026_07_12.png'),
        ('2026-07-13', 'Thứ Hai',  'Hinh_8_SHAP_Waterfall_2026_07_13.png'),
        ('2026-07-17', 'Thứ Sáu',  'Hinh_9_SHAP_Waterfall_2026_07_17.png')
    ]

    base_val_beds = float(np.squeeze(explainer.expected_value))

    for date_str, day_name, fig_name in waterfall_configs:
        target_dt = pd.to_datetime(date_str)
        idx_array = np.where(dates_test_dt == target_dt)[0]
        if len(idx_array) > 0:
            idx = int(idx_array[0])
            plt.figure(figsize=(10, 6))
            explanation = shap.Explanation(
                values=shap_values_beds[idx],
                base_values=base_val_beds,
                data=X_test_xgb_raw.iloc[idx],
                feature_names=X_columns
            )
            shap.waterfall_plot(explanation, show=False)
            plt.title(f"Giải thích cục bộ (Đơn vị: Giường): {day_name} - Ngày {target_dt.strftime('%d/%m/%Y')}", pad=25)
            plt.tight_layout()
            plt.savefig(f'{HINH}/{fig_name}', dpi=300)
            plt.close()


    # [BỔ SUNG v2] Census thực tế và dự báo cho 3 ngày minh họa (12, 13, 17/07/2026)
    _rows = []
    for _date_str, _day_name, _ in waterfall_configs:
        _ia = np.where(dates_test_dt == pd.to_datetime(_date_str))[0]
        if len(_ia) > 0:
            _i = int(_ia[0])
            _rows.append({'Ngay': _date_str, 'Thu': _day_name, 'Census_thuc_te': float(Y_test_real.ravel()[_i]),
                          'LSTM_du_bao': round(float(predictions_real.ravel()[_i]), 1),
                          'XGB_surrogate_du_bao': round(float(xgb_pred_test[_i]), 1)})
    pd.DataFrame(_rows).to_csv(THU_MUC + '/Bang_3_ngay_minh_hoa.csv', index=False, encoding='utf-8-sig')
    print('\nBa ngày minh họa:'); print(pd.DataFrame(_rows).to_string(index=False))

    # [BỔ SUNG v2] SHAP của DayOfWeek theo từng thứ của ngày t-1 (đúng với giá trị đặc trưng mô hình nhìn thấy)
    _names = {0: 'Thứ Hai', 1: 'Thứ Ba', 2: 'Thứ Tư', 3: 'Thứ Năm', 4: 'Thứ Sáu', 5: 'Thứ Bảy', 6: 'Chủ nhật'}
    _sd = pd.DataFrame({'dow_t_tru_1': X_test_xgb_raw['DayOfWeek'].astype(int).values,
                        'SHAP_DayOfWeek': shap_values_beds[:, X_columns.index('DayOfWeek')]})
    _bang_dow = _sd.groupby('dow_t_tru_1')['SHAP_DayOfWeek'].agg(['count', 'mean', 'std']).round(2)
    _bang_dow.index = [f"{_names[i]} (t-1) -> dự báo cho {_names[(i + 1) % 7]}" for i in _bang_dow.index]
    _bang_dow.to_csv(THU_MUC + '/Bang_SHAP_DayOfWeek_theo_thu_t_tru_1.csv', encoding='utf-8-sig')
    print('\nSHAP DayOfWeek theo thứ của ngày t-1:'); print(_bang_dow.to_string())
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(_bang_dow)), _bang_dow['mean'].values, color='#4C72B0')
    plt.xticks(range(len(_bang_dow)), [s.split(' (')[0] for s in _bang_dow.index], rotation=30)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('SHAP trung bình của DayOfWeek (giường)')
    plt.title('SHAP của DayOfWeek theo thứ của ngày t-1')
    plt.tight_layout()
    plt.savefig(HINH + '/Hinh_SHAP_DayOfWeek_theo_thu.png', dpi=300)
    plt.close()

    # --- Xuất Bảng 4.1: Thống kê giá trị thật và SHAP trung bình theo Thứ trong tuần (Yêu cầu A7) ---
    dow_names = {0: 'Thứ Hai', 1: 'Thứ Ba', 2: 'Thứ Tư', 3: 'Thứ Năm', 4: 'Thứ Sáu', 5: 'Thứ Bảy', 6: 'Chủ nhật'}
    df_dow_analysis = test_df.iloc[window:].reset_index(drop=True)[
        ['DayOfWeek', 'Census', 'Admissions', 'Discharges', 'Emergency_Count', 'Total_Elderly_Admitted']
    ].copy()

    shap_df_beds = pd.DataFrame(shap_values_beds, columns=[f'SHAP_{c}' for c in X_columns])
    df_dow_full = pd.concat([df_dow_analysis, shap_df_beds], axis=1)

    dow_report = df_dow_full.groupby('DayOfWeek').agg(
        N_Days=('Census', 'count'),
        Census_Mean=('Census', 'mean'),
        Admissions_Mean=('Admissions', 'mean'),
        Discharges_Mean=('Discharges', 'mean'),
        Emergency_Mean=('Emergency_Count', 'mean'),
        SHAP_Lag1_Mean=('SHAP_Lag_1', 'mean'),
        SHAP_DayOfWeek_Mean=('SHAP_DayOfWeek', 'mean'),
        SHAP_Emergency_Mean=('SHAP_Emergency_Count', 'mean'),
        SHAP_Discharges_Mean=('SHAP_Discharges', 'mean')
    ).round(2)
    dow_report.index = [dow_names[i] for i in dow_report.index]
    dow_report.to_csv(THU_MUC + '/Bang_4_1_Thong_Ke_SHAP_Theo_Thu.csv', encoding='utf-8-sig')
    print("\nĐã xuất Bảng 4.1 (Thống kê SHAP theo Thứ trong tuần) trên toàn tập Kiểm thử (193 ngày):")
    print(dow_report.to_string())

    ghi(f'[Độ trung thành – mô hình đại diện] R² train = {fidelity_train:.4f}; validation = {fidelity_val:.4f}; test = {fidelity_test:.4f}; '
        f'MAE(XGBoost vs LSTM) = {_mae_fid:.2f} giường; RMSE = {_rmse_fid:.2f}; R² dải 650–900 = {_r2_dai:.4f} (n = {int(_mask.sum())} ngày).')
    ghi(f'[Spearman] SHAP vs Permutation: rho = {_r1:.3f} (p = {_p1:.3g}); SHAP vs IG: rho = {_r2:.3f} (p = {_p2:.3g}); '
        f'IG vs Permutation: rho = {rho_ig_perm:.3f} (p = {p_ig_perm:.3g}).')
    print("\nHoàn tất phần giải thích (mô hình đại diện, SHAP, Permutation, IG).")


# =====================================================================
# PHẦN F – NGƯỠNG CẢNH BÁO CÔNG SUẤT (LSTM, seed 42)
# =====================================================================
def phan_nguong():
    if SO_GIUONG_KE_HOACH is None:
        ghi('[Phần F] Bỏ qua: chưa điền SO_GIUONG_KE_HOACH.')
        return
    nguong = NGUONG_CONG_SUAT * SO_GIUONG_KE_HOACH
    c_all = df['Census']
    vuot_all = c_all[c_all > nguong]
    ghi(f'[Ngưỡng {NGUONG_CONG_SUAT:.0%} của {SO_GIUONG_KE_HOACH} giường = {nguong:.2f}] Toàn giai đoạn: '
        f'{len(vuot_all)}/{len(c_all)} ngày vượt ngưỡng ({len(vuot_all) / len(c_all) * 100:.1f}%); '
        f'mức vượt trung bình = {(vuot_all - nguong).mean() if len(vuot_all) else 0:.1f} giường; '
        f'mức vượt lớn nhất = {(vuot_all - nguong).max() if len(vuot_all) else 0:.1f} giường.')
    v100 = c_all[c_all > SO_GIUONG_KE_HOACH]
    ghi(f'[Vượt 100% số giường kế hoạch] {len(v100)}/{len(c_all)} ngày ({len(v100) / len(c_all) * 100:.1f}%); '
        f'mức vượt trung bình = {(v100 - SO_GIUONG_KE_HOACH).mean() if len(v100) else 0:.1f}; lớn nhất = {(v100 - SO_GIUONG_KE_HOACH).max() if len(v100) else 0:.1f} giường.')
    if 42 in PRED_CHON:
        rm = so_do(y_test, PRED_CHON[42])['RMSE']
        canh_bao = (PRED_CHON[42] + rm) > nguong
        thuc = y_test > nguong
        tp = int((canh_bao & thuc).sum())
        fp = int((canh_bao & ~thuc).sum())
        fn = int((~canh_bao & thuc).sum())
        ghi(f'[Cảnh báo = dự báo LSTM + 1 RMSE ({rm:.1f}) > ngưỡng] Tập kiểm thử: số ngày cảnh báo = {int(canh_bao.sum())}; '
            f'cảnh báo đúng = {tp}; sai = {fp}; bỏ sót = {fn}; số ngày thực tế vượt ngưỡng = {int(thuc.sum())}.')
        thuc100 = y_test > SO_GIUONG_KE_HOACH
        cb100 = (PRED_CHON[42] + rm) > SO_GIUONG_KE_HOACH
        ghi(f'[Cảnh báo ở ngưỡng 100% = {SO_GIUONG_KE_HOACH}] số ngày cảnh báo = {int(cb100.sum())}; đúng = {int((cb100 & thuc100).sum())}; '
            f'sai = {int((cb100 & ~thuc100).sum())}; bỏ sót = {int((~cb100 & thuc100).sum())}; số ngày thực tế vượt = {int(thuc100.sum())}.')
        if thuc100.sum() > 0:
            ghi(f'[Mức vượt 100% trong tập kiểm thử] trung bình = {(y_test[thuc100] - SO_GIUONG_KE_HOACH).mean():.1f} giường; '
                f'lớn nhất = {(y_test[thuc100] - SO_GIUONG_KE_HOACH).max():.1f} giường.')


MUC_LUC = """MỤC LỤC KẾT QUẢ: số liệu nào dùng cho chỗ nào của báo cáo

Chương 2 (dữ liệu, EDA)
  thong_ke_lam_sach.csv            số bản ghi bị loại (mục 2.2.2)
  A1_thong_ke_mo_ta.csv, hinh/ -> A2_*.png, A3_*.png   thống kê mô tả, biểu đồ chuỗi, ma trận tương quan (mục 2.2.5)
  A4_Bang_2_1_theo_thu.csv         Bảng 2.1
  A7_chia_tap.csv                  số ngày, khoảng thời gian, số mẫu từng tập (mục 2.3)
  TOM_TAT_SO_LIEU.txt              các hệ số tương quan, phạm vi train/test (dòng [Mục 2.2.5], [Mục 3.3])
Chương 3
  C0_chon_bien_the_tren_tap_kiem_dinh.csv   bảng quy trình chọn biến thể (Phụ lục)
  C1_LSTM_theo_seed.csv, C1_du_bao_LSTM_theo_seed.csv   kết quả và dự báo từng seed
  D1_Bang_3_1_*.csv                Bảng 3.1 (theo giai đoạn)
  D2_Bang_3_2_so_sanh_mo_hinh.csv  Bảng 3.2 (có kiểm định Diebold–Mariano)
  B1_*.csv, B1_ACF_PACF.png        căn cứ chọn bậc ARIMA (mục 3.3); B2_*.csv: chọn bậc SARIMA
  hinh/Hinh_1_*.png (loss), Hinh_2_*.png (dự báo vs thực tế), Hinh_3_*.png (so sánh mô hình)
Chương 3.4 và 4 (giải thích)
  surrogate_fidelity_bo_sung.csv   độ trung thành của mô hình đại diện (R², MAE, RMSE, R² dải 650–900)
  Bang_3_4_hang_tam_quan_trong.csv Bảng 3.4 (SHAP, Permutation, IG)
  Bang_SHAP_theo_nhom_bien.csv, Bang_SHAP_DayOfWeek_theo_thu_t_tru_1.csv, Bang_3_ngay_minh_hoa.csv, Bang_4_1_*.csv
  hinh/Hinh_4_*.png ... Hinh_9_*.png, Hinh_SHAP_DayOfWeek_theo_thu.png
Mục 4.4.2 (ngưỡng cảnh báo): các dòng [Ngưỡng ...], [Cảnh báo ...] trong TOM_TAT_SO_LIEU.txt
"""


if __name__ == '__main__':
    os.makedirs(THU_MUC, exist_ok=True)
    os.makedirs(HINH, exist_ok=True)
    chay('0. Dữ liệu', phan_du_lieu)
    if df is None:
        raise SystemExit('Không có dữ liệu nên dừng. Kiểm tra raw_his_data.csv có nằm cùng thư mục với run_all.py không.')
    chay('A. Phân tích mô tả dữ liệu (EDA)', phan_A)
    chay('B0. Naive và seasonal naive', phan_B0)
    chay('B1. ARIMA (chọn bậc + dự báo cuốn chiếu 1 bước)', phan_B1)
    chay('B2. SARIMA và SARIMAX', phan_B2)
    chay('B3. XGBoost dự báo trực tiếp', phan_B3)
    chay('C. LSTM: chọn biến thể (tập Kiểm định) và chấm tập Kiểm thử (10 seed)', phan_lstm)
    chay('D. Bảng 3.1, 3.2, kiểm định Diebold–Mariano, Hình 3.3', phan_so_sanh)
    chay('E. Giải thích: mô hình đại diện, SHAP, Permutation, IG', phan_xai)
    chay('F. Ngưỡng cảnh báo công suất', phan_nguong)
    with open(os.path.join(THU_MUC, 'TOM_TAT_SO_LIEU.txt'), 'w', encoding='utf-8-sig') as f:
        f.write('TÓM TẮT SỐ LIỆU ĐỂ ĐIỀN VÀO BÁO CÁO\n(mỗi dòng ghi rõ số này dùng cho chỗ nào)\n\n')
        f.write('\n'.join(TOM_TAT))
        f.write('\n')
    with open(os.path.join(THU_MUC, 'MUC_LUC_KET_QUA.txt'), 'w', encoding='utf-8-sig') as f:
        f.write(MUC_LUC)
    print('\nHOÀN TẤT. Mở KET_QUA/TOM_TAT_SO_LIEU.txt và KET_QUA/MUC_LUC_KET_QUA.txt')
