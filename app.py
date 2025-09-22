# -*- coding: utf-8 -*-
# Teacher Dashboard: Satellite SST + Generative-AI Friendly
# 요구 패키지: streamlit, netCDF4, numpy, matplotlib, pandas

import io, zipfile, urllib.request
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset, num2date

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

st.set_page_config(page_title="SST Classroom Dashboard (Teacher)", layout="wide")

# =========================
# 유틸 함수
# =========================
def to_python_datetimes(time_like):
    """cftime / python datetime → python datetime 리스트"""
    if time_like is None:
        return None
    out = []
    for d in time_like:
        if hasattr(d, "year") and hasattr(d, "month") and hasattr(d, "day"):
            out.append(datetime(int(d.year), int(d.month), int(d.day),
                                int(getattr(d, "hour", 0)),
                                int(getattr(d, "minute", 0)),
                                int(getattr(d, "second", 0))))
        else:
            out.append(d)
    return out

def wrap_and_sort_lon(lon):
    """경도 0–360 → -180–180 래핑 후 정렬"""
    lon_wrapped = (lon + 180) % 360 - 180
    sort_idx = np.argsort(lon_wrapped)
    return lon_wrapped[sort_idx], sort_idx

def mask_fill(arr, var):
    """missing_value/_FillValue/비유효값 → NaN"""
    arr = np.array(arr, dtype=float, copy=True)
    fv = None
    for key in ("missing_value", "_FillValue"):
        if hasattr(var, key):
            fv = getattr(var, key)
            break
    if fv is not None:
        arr[np.isclose(arr, float(fv))] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr

def bbox_subset(lat, lon_sorted, arr_sorted, latmin, latmax, lonmin, lonmax):
    """
    arr_sorted: (..., lat, lon) 또는 (lat, lon)
    날짜변경선 교차(예: 170E ~ -170W)도 처리
    """
    lat_mask = (lat >= latmin) & (lat <= latmax)
    if lonmin <= lonmax:
        lon_mask = (lon_sorted >= lonmin) & (lon_sorted <= lonmax)
        return (lat[lat_mask], lon_sorted[lon_mask],
                arr_sorted[..., lat_mask, :][..., :, lon_mask])
    else:
        lon_mask1 = (lon_sorted >= lonmin)
        lon_mask2 = (lon_sorted <= lonmax)
        lat_sub = lat[lat_mask]
        lon_sub = np.concatenate([lon_sorted[lon_mask1], lon_sorted[lon_mask2]])
        left  = arr_sorted[..., lat_mask, :][..., :, lon_mask1]
        right = arr_sorted[..., lat_mask, :][..., :, lon_mask2]
        arr_sub = np.concatenate([left, right], axis=-1)
        return lat_sub, lon_sub, arr_sub

def nan_arg_extreme_2d(arr2d, mode="max"):
    """2D에서 NaN 무시 극값과 위치(j,i)"""
    if np.all(np.isnan(arr2d)):
        return None
    if mode == "max":
        idx = np.nanargmax(arr2d); val = np.nanmax(arr2d)
    else:
        idx = np.nanargmin(arr2d); val = np.nanmin(arr2d)
    j, i = np.unravel_index(idx, arr2d.shape)
    return val, j, i

def format_date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except Exception:
        formatter = mdates.DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(formatter)
    ax.figure.autofmt_xdate()

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.read()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def read_nc_from_bytes(data: bytes) -> Dataset:
    """메모리에서 NetCDF 읽기"""
    return Dataset("inmemory.nc", memory=data)

def load_dataset(uploaded_file, url: str) -> Optional[Dataset]:
    """업로드 또는 URL에서 Dataset 로드"""
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as zf:
                nc_names = [n for n in zf.namelist() if n.lower().endswith((".nc", ".nc4"))]
                if not nc_names:
                    st.error("ZIP 안에 .nc/.nc4 파일이 없습니다.")
                    return None
                with zf.open(nc_names[0]) as f:
                    data = f.read()
                return read_nc_from_bytes(data)
        else:
            data = uploaded_file.read()
            return read_nc_from_bytes(data)
    elif url:
        try:
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            return read_nc_from_bytes(data)
        except Exception as e:
            st.error(f"URL 다운로드 중 문제: {e}")
            return None
    else:
        return None

def get_preset_box(name: str) -> Tuple[float, float, float, float]:
    """한국 근해 프리셋 박스"""
    presets = {
        "직접 입력": (None, None, None, None),
        # 대략 범위(교육용): 필요시 조정 가능
        "한국 근해(기본)": (20.0, 50.0, 120.0, 150.0),
        "동해(Sea of Japan)": (33.0, 50.0, 128.0, 142.0),
        "황해(Yellow Sea)": (31.0, 40.0, 118.0, 126.0),
        "남해(South Sea)": (32.0, 36.0, 126.0, 132.0),
    }
    return presets.get(name, (None, None, None, None))

def try_ice_mask(ds: Dataset, sst_sub: np.ndarray, lat, lon_sorted, sort_idx, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, thr_pct: float):
    """ice 변수(%)가 있으면 thr 이상을 NaN으로 마스킹"""
    if "ice" not in ds.variables:
        return sst_sub, None
    try:
        ice_var = ds.variables["ice"]
        ice = ice_var[:]
        if ice.ndim == 2:
            ice = ice[None, ...]
        ice = mask_fill(ice, ice_var)
        ice = standardize_to_time_lat_lon(ice_var, ice, lat, lon)  # ← 추가
        ice_sorted = ice[:, :, sort_idx]
        _, _, ice_sub = bbox_subset(lat, lon_sorted, ice_sorted, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        sst_masked = sst_sub.copy()
        # 시간 길이 맞추기(혹시 time=1 vs N)
        tmin = min(sst_masked.shape[0], ice_sub.shape[0])
        mask = ice_sub[:tmin] >= thr_pct
        sst_masked[:tmin][mask] = np.nan
        return sst_masked, ice_sub
    except Exception:
        # ice가 이상한 경우 원본 그대로 반환
        return sst_sub, None
    

def standardize_to_time_lat_lon(var, arr, lat, lon):
    """
    어떤 차원 순서로 와도 sst/anom/ice를 (time, lat, lon)로 맞춰준다.
    - 2D면 (lat, lon) 또는 (lon, lat)을 감지해서 (1, lat, lon)으로 만든다.
    - 3D면 (time, lat, lon) 또는 (time, lon, lat) 등을 감지해 표준화한다.
    """
    arr = np.array(arr)
    lat_len, lon_len = len(lat), len(lon)
    dims = tuple(getattr(var, "dimensions", ()))  # e.g. ('time','lon','lat') 등

    # 2D: (lat, lon) or (lon, lat)
    if arr.ndim == 2:
        if arr.shape == (lat_len, lon_len):
            return arr[None, ...]  # (1, lat, lon)
        if arr.shape == (lon_len, lat_len):
            return arr.T[None, ...]  # (1, lat, lon)
        # 이름으로도 못 맞추면 크기 기준 추론
        lat_axis = 0 if arr.shape[0] == lat_len else (1 if arr.shape[1] == lat_len else None)
        lon_axis = 0 if arr.shape[0] == lon_len else (1 if arr.shape[1] == lon_len else None)
        if lat_axis is not None and lon_axis is not None and lat_axis != lon_axis:
            if (lat_axis, lon_axis) == (0, 1):
                return arr[None, ...]
            if (lat_axis, lon_axis) == (1, 0):
                return arr.T[None, ...]
        raise ValueError(f"Unexpected 2D shape {arr.shape}; expected ({lat_len},{lon_len}) or ({lon_len},{lat_len})")

    # 3D: 다양한 순서 → (time, lat, lon)로
    if arr.ndim == 3:
        # ① 크기만으로 빠르게 처리
        t, a1, a2 = arr.shape
        if (a1, a2) == (lat_len, lon_len):
            return arr  # (time, lat, lon)
        if (a1, a2) == (lon_len, lat_len):
            return np.transpose(arr, (0, 2, 1))  # (time, lat, lon)

        # ② 차원 이름이 있으면 이름으로 맞추기
        if len(dims) == 3:
            names = [d.lower() for d in dims]
            # time, lat, lon 축 찾기 (name에 lat/lon/y/x 등 포함 가능성 고려)
            try:
                t_axis = names.index("time")
            except ValueError:
                t_axis = 0  # 못 찾으면 0 가정
            lat_axis = next((i for i, n in enumerate(names) if ("lat" in n) or (n in ("y","latitude"))), None)
            lon_axis = next((i for i, n in enumerate(names) if ("lon" in n) or (n in ("x","longitude"))), None)
            if None not in (t_axis, lat_axis, lon_axis):
                return np.moveaxis(arr, (t_axis, lat_axis, lon_axis), (0, 1, 2))

        # ③ 마지막 수단: 크기 매칭으로 축 잡기
        axes_sizes = list(arr.shape)
        lat_cands = [i for i, s in enumerate(axes_sizes) if s == lat_len]
        lon_cands = [i for i, s in enumerate(axes_sizes) if s == lon_len]
        # time 축 후보: lat/lon이 아닌 나머지
        other = [0, 1, 2]
        if lat_cands and lon_cands:
            # 우선 하나씩 선택
            lat_axis = lat_cands[0]
            lon_axis = lon_cands[0]
            # 남은 축을 time으로
            other.remove(lat_axis); other.remove(lon_axis)
            t_axis = other[0]
            return np.moveaxis(arr, (t_axis, lat_axis, lon_axis), (0, 1, 2))

        raise ValueError(f"Cannot standardize shape {arr.shape} to (time, lat, lon)")

    raise ValueError(f"Array with ndim={arr.ndim} not supported")
  

# =========================
# 사이드바 UI
# =========================
st.sidebar.header("1) 데이터 입력")
up = st.sidebar.file_uploader("NetCDF 업로드 (.nc / .nc4 / .zip)", type=["nc", "nc4", "zip"])
st.sidebar.markdown("또는")
url = st.sidebar.text_input("원격 파일 URL (HTTP/HTTPS)")

st.sidebar.header("2) 해역 선택")
preset = st.sidebar.selectbox("프리셋", ["직접 입력", "한국 근해(기본)", "동해(Sea of Japan)", "황해(Yellow Sea)", "남해(South Sea)"])
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = get_preset_box(preset)
if LAT_MIN is None:
    LAT_MIN = st.sidebar.number_input("LAT_MIN", value=20.0, step=0.5)
    LAT_MAX = st.sidebar.number_input("LAT_MAX", value=50.0, step=0.5)
    LON_MIN = st.sidebar.number_input("LON_MIN", value=120.0, step=0.5)
    LON_MAX = st.sidebar.number_input("LON_MAX", value=150.0, step=0.5)
else:
    st.sidebar.write(f"위도 {LAT_MIN}~{LAT_MAX}°, 경도 {LON_MIN}~{LON_MAX}°")

st.sidebar.header("3) 분석 모드")
mode = st.sidebar.selectbox("유형", ["특정 하루", "전체 기간", "특정 기간(월)", "특정 기간(날짜)"])
TIME_INDEX = st.sidebar.number_input("특정 하루: TIME_INDEX (0-based)", min_value=0, value=100, step=1)

MONTH_START = st.sidebar.number_input("월 시작", min_value=1, max_value=12, value=3, step=1)
MONTH_END   = st.sidebar.number_input("월 끝",   min_value=1, max_value=12, value=6, step=1)
DATE_START  = st.sidebar.text_input("날짜 시작 (예: 2014-03-01)", value="2014-03-01")
DATE_END    = st.sidebar.text_input("날짜 끝 (예: 2014-06-30)", value="2014-06-30")

st.sidebar.header("4) 옵션")
SST_VMIN = st.sidebar.number_input("SST 색상 최소", value=-2.0)
SST_VMAX = st.sidebar.number_input("SST 색상 최대", value=32.0)
ANOM_CMAP = st.sidebar.selectbox("아노말리 컬러맵", ["bwr", "coolwarm", "RdBu_r", "jet"])
use_ice_mask = st.sidebar.checkbox("ice ≥ 15% 마스크 적용(있을 때)", value=False)
ice_thr = st.sidebar.slider("ice 임계값(%)", min_value=0, max_value=100, value=15, step=5, disabled=not use_ice_mask)

st.sidebar.header("5) 저장")
want_png = st.sidebar.checkbox("그림 PNG 다운로드 버튼 표시", value=True)
want_csv = st.sidebar.checkbox("CSV 다운로드 버튼 표시", value=True)

run = st.sidebar.button("실행")

st.title("SST Classroom Dashboard — 교사용 데모")
st.caption("lat=720, lon=1440 가정. time=1(하루) 또는 365(연간 일자료) 대응. 업로드 제한이 크면 ZIP 또는 URL 사용.")

# =========================
# 메인 로직
# =========================
if run:
    ds = load_dataset(up, url)
    if ds is None:
        st.warning("파일을 업로드하거나 URL을 입력하세요.")
        st.stop()

    # 변수 읽기
    try:
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        # sst 후보 탐색
        cand = ["sst", "analysed_sst", "sea_surface_temperature"]
        sst_name = next((v for v in cand if v in ds.variables), None)
        if sst_name is None:
            # 2D/3D 그리드형 첫 변수 사용(예외적 파일)
            sst_name = next((k for k,v in ds.variables.items() if getattr(v, "ndim", 0) in (2,3) and k not in ("lat","lon","time")), None)
        if sst_name is None:
            st.error("SST 변수명을 찾을 수 없습니다. 파일 변수 목록을 확인하세요.")
            st.stop()
        sst_var = ds.variables[sst_name]
        sst = sst_var[:]
        sst = standardize_to_time_lat_lon(sst_var, sst, lat, lon)  # ← 추가

    except Exception as e:
        st.error(f"필수 변수(lat/lon/sst) 읽기 실패: {e}")
        st.stop()

    # time → datetime
    time_py = None
    if "time" in ds.variables:
        try:
            tvar = ds.variables["time"]
            time_vals = tvar[:]
            time_py = to_python_datetimes(num2date(time_vals, units=getattr(tvar,"units",None),
                                                   calendar=getattr(tvar,"calendar","standard")))
        except Exception:
            time_py = None

    # 차원 통일
    if sst.ndim == 2:
        sst = sst[None, ...]  # (1, lat, lon)

    # 결측/정렬/서브셋
    sst = mask_fill(sst, sst_var)
    lon_sorted, sort_idx = wrap_and_sort_lon(lon)
    sst_sorted = sst[:, :, sort_idx]
    lat_sub, lon_sub, sst_sub = bbox_subset(lat, lon_sorted, sst_sorted, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

    # ice 마스크(선택)
    if use_ice_mask:
        sst_sub, ice_sub = try_ice_mask(ds, sst_sub, lat, lon_sorted, sort_idx, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ice_thr)
        if ice_sub is None:
            st.info("이 파일에는 'ice' 변수가 없어 마스킹을 건너뜁니다.")

    # 데이터 개요
    c1, c2, c3 = st.columns(3)
    c1.metric("time", sst_sub.shape[0])
    c2.metric("lat", sst_sub.shape[1])
    c3.metric("lon", sst_sub.shape[2])
    if time_py:
        st.caption(f"날짜 범위: {time_py[0].date()} ~ {time_py[-1].date()}  (총 {len(time_py)}일)")

    # 지도 그리기 함수(다운로드 포함)
    def show_map(field2d: np.ndarray, title: str, cbar_label: str, vmin=None, vmax=None, cmap="jet", key="map"):
        lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        pc = ax.pcolormesh(lon2d, lat2d, field2d, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        cb = fig.colorbar(pc, ax=ax, label=cbar_label)
        ax.set_title(title)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.tight_layout()
        st.pyplot(fig)
        if want_png:
            st.download_button("⬇️ PNG 저장", data=fig_to_png_bytes(fig), file_name=f"{key}.png", mime="image/png")
        plt.close(fig)

    # 시계열 그리기 함수
    def show_timeseries(time_axis, values: np.ndarray, title: str, ylab="°C", key="ts"):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        if time_axis is None:
            ax.plot(values)
            ax.set_xlabel("time index")
        else:
            ax.plot(time_axis, values)
            format_date_axis(ax)
            ax.set_xlabel("date")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.tight_layout()
        st.pyplot(fig)
        if want_png:
            st.download_button("⬇️ PNG 저장", data=fig_to_png_bytes(fig), file_name=f"{key}.png", mime="image/png")
        plt.close(fig)

    # ---------- 분석 분기 ----------
    if mode == "특정 하루":
        ti = int(TIME_INDEX)
        if ti < 0 or ti >= sst_sub.shape[0]:
            st.error(f"TIME_INDEX {ti} 범위 오류 (0..{sst_sub.shape[0]-1})")
            st.stop()
        tlabel = f"t={ti}" if time_py is None else str(time_py[ti].date())
        s2d = sst_sub[ti, :, :]

        # 지도
        show_map(s2d, f"SST @ {tlabel}  ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)",
                 cbar_label="SST (°C)", vmin=SST_VMIN, vmax=SST_VMAX, cmap="jet", key=f"sst_{tlabel}")

        # 평균·극값
        mean_sst = float(np.nanmean(s2d))
        rmax = nan_arg_extreme_2d(s2d, "max")
        rmin = nan_arg_extreme_2d(s2d, "min")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("영역 평균(°C)", f"{mean_sst:.3f}")
        if rmax: mcol2.write(f"**최대** {rmax[0]:.3f} °C @ (lat,lon)=({lat_sub[rmax[1]]:.3f}, {lon_sub[rmax[2]]:.3f})")
        if rmin: mcol3.write(f"**최소** {rmin[0]:.3f} °C @ (lat,lon)=({lat_sub[rmin[1]]:.3f}, {lon_sub[rmin[2]]:.3f})")

        # 아노말리(해당 날짜의 영역 평균 기준)
        anom = s2d - mean_sst
        show_map(anom, f"Anomaly vs Regional Mean @ {tlabel}", cbar_label="Anomaly (°C)",
                 vmin=None, vmax=None, cmap=ANOM_CMAP, key=f"anom_reg_{tlabel}")

        # 파일 제공 anom(있다면)
        if "anom" in ds.variables:
            anom_var = ds.variables["anom"]
            a = anom_var[:]
            if a.ndim == 2: a = a[None, ...]
            a = mask_fill(a, anom_var)
            a = standardize_to_time_lat_lon(anom_var, a, lat, lon)  # ← 추가
            a_sorted = a[:, :, sort_idx]
            _, _, a_sub = bbox_subset(lat, lon_sorted, a_sorted, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            a2d = a_sub[ti, :, :]
            show_map(a2d, f"Provided ANOM @ {tlabel}", cbar_label="Anomaly (°C)",
                     vmin=None, vmax=None, cmap=ANOM_CMAP, key=f"anom_provided_{tlabel}")
        else:
            st.info("이 파일에는 'anom' 변수가 없습니다.")

        # CSV(선택): 픽셀 요약 대신, 이 날의 스칼라 요약만 제공
        if want_csv:
            df = pd.DataFrame({
                "date": [tlabel],
                "mean_C": [mean_sst],
                "max_C": [None if rmax is None else rmax[0]],
                "max_lat": [None if rmax is None else float(lat_sub[rmax[1]])],
                "max_lon": [None if rmax is None else float(lon_sub[rmax[2]])],
                "min_C": [None if rmin is None else rmin[0]],
                "min_lat": [None if rmin is None else float(lat_sub[rmin[1]])],
                "min_lon": [None if rmin is None else float(lon_sub[rmin[2]])],
            })
            st.download_button("⬇️ CSV 저장(하루 요약)", data=df_to_csv_bytes(df),
                               file_name=f"summary_{tlabel}.csv", mime="text/csv")

    elif mode == "전체 기간":
        # 시계열(영역 평균)
        mean_ts = np.nanmean(sst_sub, axis=(1,2))
        st.write("**[시계열 요약]**",
                 f"min={float(np.nanmin(mean_ts)):.3f}, mean={float(np.nanmean(mean_ts)):.3f}, max={float(np.nanmax(mean_ts)):.3f}")
        show_timeseries(time_py, mean_ts, f"Regional Mean SST  ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)", key="ts_all")

        # 기간 평균 지도
        mean2d = np.nanmean(sst_sub, axis=0)
        show_map(mean2d, "Period-Mean SST", cbar_label="SST (°C)",
                 vmin=SST_VMIN, vmax=SST_VMAX, cmap="jet", key="meanmap_all")

        # CSV(선택): 영역평균 시계열
        if want_csv:
            if time_py is None:
                df = pd.DataFrame({"time_index": np.arange(len(mean_ts)), "mean_C": mean_ts})
            else:
                df = pd.DataFrame({"date": [d.date().isoformat() for d in time_py], "mean_C": mean_ts})
            st.download_button("⬇️ CSV 저장(영역평균 시계열)", data=df_to_csv_bytes(df),
                               file_name="regional_mean_timeseries.csv", mime="text/csv")

    else:  # 특정 기간(월) / 특정 기간(날짜)
        if time_py is None:
            st.error("time 좌표를 날짜로 변환할 수 없어 기간 필터를 적용할 수 없습니다.")
            st.stop()

        tp = np.array(time_py)
        if mode == "특정 기간(월)":
            months = np.array([d.month for d in tp])
            if MONTH_START <= MONTH_END:
                tmask = (months >= MONTH_START) & (months <= MONTH_END)
            else:  # 11~2월 같은 케이스
                tmask = (months >= MONTH_START) | (months <= MONTH_END)
            label_period = f"{MONTH_START}~{MONTH_END}월"
        else:
            try:
                y1,m1,d1 = map(int, DATE_START.split("-"))
                y2,m2,d2 = map(int, DATE_END.split("-"))
                start_dt = datetime(y1,m1,d1); end_dt = datetime(y2,m2,d2)
                tmask = (tp >= start_dt) & (tp <= end_dt)
                label_period = f"{start_dt.date()} ~ {end_dt.date()}"
            except Exception:
                st.error("날짜 형식이 잘못되었습니다. 예: 2014-03-01")
                st.stop()

        if not np.any(tmask):
            st.warning(f"선택 기간({label_period})에 해당하는 데이터가 없습니다.")
            st.stop()

        sst_sel = sst_sub[tmask, :, :]
        time_sel = tp[tmask]

        # 기간 평균 지도
        mean2d = np.nanmean(sst_sel, axis=0)
        show_map(mean2d, f"Period-Mean SST [{label_period}]", cbar_label="SST (°C)",
                 vmin=SST_VMIN, vmax=SST_VMAX, cmap="jet", key=f"meanmap_{label_period}")

        # 기간 전체 영역 평균(스칼라)
        mean_scalar = float(np.nanmean(sst_sel))
        st.metric("기간 전체 영역 평균 SST (°C)", f"{mean_scalar:.3f}")

        # 기간 내 극값(시간·공간)
        vmax = float(np.nanmax(sst_sel)); vmin = float(np.nanmin(sst_sel))
        idx_max = np.nanargmax(sst_sel); idx_min = np.nanargmin(sst_sel)
        t_max, j_max, i_max = np.unravel_index(idx_max, sst_sel.shape)
        t_min, j_min, i_min = np.unravel_index(idx_min, sst_sel.shape)
        cmax, cmin = st.columns(2)
        cmax.write(f"**최대** {vmax:.3f} °C  @ {time_sel[t_max].date()}, (lat,lon)=({lat_sub[j_max]:.3f}, {lon_sub[i_max]:.3f})")
        cmin.write(f"**최소** {vmin:.3f} °C  @ {time_sel[t_min].date()}, (lat,lon)=({lat_sub[j_min]:.3f}, {lon_sub[i_min]:.3f})")

        # 아노말리 1: (시간별 영역 평균) 기준
        mean_ts_sel = np.nanmean(sst_sel, axis=(1,2))
        anom_regional_sel = sst_sel - mean_ts_sel[:, None, None]
        show_map(anom_regional_sel[0], f"Anomaly vs Daily Regional Mean (first day in {label_period})",
                 cbar_label="Anomaly (°C)", vmin=None, vmax=None, cmap=ANOM_CMAP, key=f"anom_reg_{label_period}")

        # 아노말리 2: (기간 평균 지도) 기준
        anom_period_map = sst_sel - mean2d[None, :, :]
        show_map(anom_period_map[0], f"Anomaly vs Period-Mean (first day in {label_period})",
                 cbar_label="Anomaly (°C)", vmin=None, vmax=None, cmap=ANOM_CMAP, key=f"anom_map_{label_period}")

        # CSV(선택): 스칼라 요약 + 기간 극값
        if want_csv:
            df = pd.DataFrame({
                "period": [label_period],
                "mean_C": [mean_scalar],
                "max_C": [vmax],
                "max_date": [time_sel[t_max].date().isoformat()],
                "max_lat": [float(lat_sub[j_max])],
                "max_lon": [float(lon_sub[i_max])],
                "min_C": [vmin],
                "min_date": [time_sel[t_min].date().isoformat()],
                "min_lat": [float(lat_sub[j_min])],
                "min_lon": [float(lon_sub[i_min])],
            })
            st.download_button("⬇️ CSV 저장(기간 요약)", data=df_to_csv_bytes(df),
                               file_name=f"summary_{label_period}.csv", mime="text/csv")

    ds.close()

