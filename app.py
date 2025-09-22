# --- STREAMLIT CLOUD SAFE BOOT HEADER ---
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)  # ← 폴더 보장 (중요)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(page_title="SST Classroom Dashboard", layout="wide")
# -----------------------------------------
import io
import zipfile
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# ==============================
# UI 기본 설정
# ==============================
st.set_page_config(page_title="SST Classroom Dashboard", layout="wide")
st.title("SST Classroom Dashboard – Teacher")

st.caption("lat=720, lon=1440 가정. time=1(하루) 또는 365(연간 일자료) 대응. 업로드 제한이 크면 URL 또는 ZIP도 가능.")

# ==============================
# 유틸 함수
# ==============================
def to_py_dt(time_like):
    """cftime → datetime(표준 파이썬)로 변환. 실패하면 None."""
    if time_like is None:
        return None
    out = []
    for d in time_like:
        try:
            # cftime이나 numpy.datetime64 모두 처리
            y = int(getattr(d, "year", str(d)[:4]))
            m = int(getattr(d, "month", str(d)[5:7]))
            dd = int(getattr(d, "day", str(d)[8:10]))
            h = int(getattr(d, "hour", 0))
            mi = int(getattr(d, "minute", 0))
            s = int(getattr(d, "second", 0))
            out.append(datetime(y, m, dd, h, mi, s))
        except Exception:
            return None
    return out

def wrap_sort_lon(lon):
    """경도를 -180~180으로 변환하고 정렬 인덱스를 반환."""
    lon_wrapped = (lon + 180) % 360 - 180
    idx = np.argsort(lon_wrapped)
    return lon_wrapped[idx], idx

def mask_fill(arr, var):
    """_FillValue/missing_value/비정상값을 NaN으로."""
    arr = np.array(arr, dtype=float, copy=True)
    fv = None
    for k in ("missing_value", "_FillValue"):
        if hasattr(var, k):
            fv = getattr(var, k)
            break
    if fv is not None:
        arr[np.isclose(arr, float(fv))] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr

def standardize_to_time_lat_lon(var, arr, lat, lon):
    """
    어떤 구조든 (time, lat, lon)으로 변환.
    - 4D(예: time=1, zlev=1, lat, lon) → squeeze → 2D/3D
    - 2D: (lat,lon)/(lon,lat) 판단
    - 3D: (time,lat,lon)/(time,lon,lat) 판단
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    lat_len, lon_len = len(lat), len(lon)
    dims = tuple(getattr(var, "dimensions", ()))

    if arr.ndim == 2:
        if arr.shape == (lat_len, lon_len):    # (lat,lon)
            return arr[None, ...]
        if arr.shape == (lon_len, lat_len):    # (lon,lat)
            return arr.T[None, ...]
        # 크기 기준 추론
        lat_axis = 0 if arr.shape[0] == lat_len else (1 if arr.shape[1] == lat_len else None)
        lon_axis = 0 if arr.shape[0] == lon_len else (1 if arr.shape[1] == lon_len else None)
        if None not in (lat_axis, lon_axis) and lat_axis != lon_axis:
            return arr[None, ...] if (lat_axis, lon_axis) == (0, 1) else arr.T[None, ...]
        raise ValueError(f"Unexpected 2D shape {arr.shape}")

    if arr.ndim == 3:
        # 빠른 크기 매칭
        if arr.shape[1:] == (lat_len, lon_len):          # (t,lat,lon)
            return arr
        if arr.shape[1:] == (lon_len, lat_len):          # (t,lon,lat)
            return np.transpose(arr, (0, 2, 1))
        # 차원 이름 기반
        if len(dims) == 3:
            names = [d.lower() for d in dims]
            t_axis = names.index("time") if "time" in names else 0
            lat_axis = next((i for i, n in enumerate(names) if ("lat" in n) or (n in ("y", "latitude"))), None)
            lon_axis = next((i for i, n in enumerate(names) if ("lon" in n) or (n in ("x", "longitude"))), None)
            if None not in (t_axis, lat_axis, lon_axis):
                return np.moveaxis(arr, (t_axis, lat_axis, lon_axis), (0, 1, 2))
        # 최후: 크기 매칭
        sizes = list(arr.shape)
        lat_c = [i for i, s in enumerate(sizes) if s == lat_len]
        lon_c = [i for i, s in enumerate(sizes) if s == lon_len]
        if lat_c and lon_c:
            lat_axis, lon_axis = lat_c[0], lon_c[0]
            axes = [0, 1, 2]
            axes.remove(lat_axis)
            axes.remove(lon_axis)
            t_axis = axes[0]
            return np.moveaxis(arr, (t_axis, lat_axis, lon_axis), (0, 1, 2))
        raise ValueError(f"Cannot standardize shape {arr.shape} to (time,lat,lon)")

    raise ValueError(f"Array with ndim={arr.ndim} not supported")

def subset_bbox(lat, lon_sorted, arr_sorted, latmin, latmax, lonmin, lonmax):
    """경도 범위가 dateline(180E) 교차해도 안전하게 잘라낸다."""
    lat_mask = (lat >= latmin) & (lat <= latmax)
    if lonmin <= lonmax:
        lon_mask = (lon_sorted >= lonmin) & (lon_sorted <= lonmax)
        return lat[lat_mask], lon_sorted[lon_mask], arr_sorted[..., lat_mask, :][..., :, lon_mask]
    # 교차
    m1 = lon_sorted >= lonmin
    m2 = lon_sorted <= lonmax
    lsub = lat[lat_mask]
    xsub = np.concatenate([lon_sorted[m1], lon_sorted[m2]])
    left = arr_sorted[..., lat_mask, :][..., :, m1]
    right = arr_sorted[..., lat_mask, :][..., :, m2]
    return lsub, xsub, np.concatenate([left, right], axis=-1)

def nan_arg_extreme_2d(arr2d, mode="max"):
    if np.all(np.isnan(arr2d)):
        return None
    if mode == "max":
        idx = np.nanargmax(arr2d); val = np.nanmax(arr2d)
    else:
        idx = np.nanargmin(arr2d); val = np.nanmin(arr2d)
    j, i = np.unravel_index(idx, arr2d.shape)
    return val, j, i

def fig_map(lon, lat, field2d, title, cbar="°C", vmin=None, vmax=None, cmap="jet", figsize=(7.5, 5.2)):
    X, Y = np.meshgrid(lon, lat)
    fig, ax = plt.subplots(figsize=figsize)
    pc = ax.pcolormesh(X, Y, field2d, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    cb = fig.colorbar(pc, ax=ax, label=cbar)
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.tight_layout()
    return fig

def fig_ts(time_axis, values, title, ylab="°C"):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    if time_axis is None:
        ax.plot(values); ax.set_xlabel("time index")
    else:
        ax.plot(time_axis, values)
        loc = mdates.AutoDateLocator(); ax.xaxis.set_major_locator(loc)
        try:
            fmt = mdates.ConciseDateFormatter(loc)
        except Exception:
            fmt = mdates.DateFormatter("%Y-%m")
        ax.xaxis.set_major_formatter(fmt); fig.autofmt_xdate()
        ax.set_xlabel("date")
    ax.set_ylabel(ylab); ax.set_title(title)
    fig.tight_layout()
    return fig

# ==============================
# 안전 로더 (netCDF4 → 실패 시 xarray(h5netcdf))
# ==============================
class VarLike:
    def __init__(self, data, dims, attrs):
        self._data = np.asarray(data)
        self.dimensions = tuple(dims)
        for k, v in attrs.items():
            setattr(self, k, v)
    def __getitem__(self, idx): return self._data[idx]
    @property
    def shape(self): return self._data.shape
    @property
    def dtype(self): return self._data.dtype
    def __array__(self): return self._data

class DSLike:
    def __init__(self, lat, lon, time, variables):
        self.lat, self.lon, self.time = lat, lon, time
        self.variables = variables  # dict-like: name → VarLike or netCDF4.Variable

def open_nc_any(b: bytes) -> DSLike:
    # 1) netCDF4 먼저 시도
    try:
        from netCDF4 import Dataset  # requirements에 포함
        ds = Dataset(io.BytesIO(b))
        lat = np.asarray(ds.variables["lat"][:])
        lon = np.asarray(ds.variables["lon"][:])
        time = np.asarray(ds.variables["time"][:]) if "time" in ds.variables else None
        return DSLike(lat, lon, time, ds.variables)
    except Exception:
        pass

    # 2) 실패 시 xarray(h5netcdf)
    import xarray as xr
    d = xr.open_dataset(io.BytesIO(b), engine="h5netcdf")
    lat = np.asarray(d["lat"].values)
    lon = np.asarray(d["lon"].values)
    time = np.asarray(d["time"].values) if "time" in d.variables else None

    vars_dict = {}
    for name, v in d.variables.items():
        if name in ("lat", "lon", "time"):
            continue
        vars_dict[name] = VarLike(
            v.values,
            tuple(str(x) for x in getattr(v, "dims", ())),
            dict(v.attrs),
        )
    return DSLike(lat, lon, time, vars_dict)

# ==============================
# 입력(업로드/URL)
# ==============================
with st.sidebar:
    st.header("입력 데이터")
    mode_in = st.radio("가져오기 방식", ["파일 업로드", "URL"], horizontal=True)
    uploaded = None
    url = None
    if mode_in == "파일 업로드":
        uploaded = st.file_uploader("NetCDF(.nc/.nc4) 또는 ZIP", type=["nc", "nc4", "zip"])
    else:
        url = st.text_input("NetCDF/.zip URL (HTTPS 권장)", placeholder="https://example.com/file.nc")

    st.divider()
    st.header("영역 선택")
    preset = st.selectbox("프리셋", ["직접입력", "한반도 주변(20~50N, 120~150E)", "동해(35~45N, 130~142E)", "황해(30~40N, 119~126E)", "남해(32~35N, 125~130E)"])
    if preset == "한반도 주변(20~50N, 120~150E)":
        lat_min, lat_max, lon_min, lon_max = 20.0, 50.0, 120.0, 150.0
    elif preset == "동해(35~45N, 130~142E)":
        lat_min, lat_max, lon_min, lon_max = 35.0, 45.0, 130.0, 142.0
    elif preset == "황해(30~40N, 119~126E)":
        lat_min, lat_max, lon_min, lon_max = 30.0, 40.0, 119.0, 126.0
    elif preset == "남해(32~35N, 125~130E)":
        lat_min, lat_max, lon_min, lon_max = 32.0, 35.0, 125.0, 130.0
    else:
        lat_min = st.number_input("LAT_MIN", value=20.0, step=0.5)
        lat_max = st.number_input("LAT_MAX", value=50.0, step=0.5)
        lon_min = st.number_input("LON_MIN", value=120.0, step=0.5)
        lon_max = st.number_input("LON_MAX", value=150.0, step=0.5)

    st.divider()
    st.header("모드")
    mode = st.selectbox("분석 모드", ["특정 하루", "전체 기간", "특정 기간(월)", "특정 기간(날짜)"])
    time_index = st.number_input("TIME_INDEX (특정 하루)", value=0, step=1)
    m_start = st.number_input("MONTH_START", value=3, step=1, min_value=1, max_value=12)
    m_end   = st.number_input("MONTH_END",   value=6, step=1, min_value=1, max_value=12)
    date_start = st.text_input("DATE_START (YYYY-MM-DD)", value="2014-03-01")
    date_end   = st.text_input("DATE_END (YYYY-MM-DD)",   value="2014-06-30")

    st.divider()
    st.header("시각화 옵션")
    sst_vmin = st.number_input("SST_vmin", value=-2.0, step=0.5)
    sst_vmax = st.number_input("SST_vmax", value=32.0, step=0.5)
    anom_cmap = st.selectbox("Anomaly colormap", ["bwr", "coolwarm", "RdBu_r", "jet"])

# ==============================
# 데이터 로드
# ==============================
def fetch_url(u: str) -> bytes:
    import requests
    r = requests.get(u, timeout=30)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=False)
def load_bytes(mode_in, uploaded, url):
    if mode_in == "파일 업로드":
        if not uploaded:
            return None
        content = uploaded.read()
    else:
        if not url:
            return None
        content = fetch_url(url)

    # ZIP이면 첫 번째 .nc를 펼쳐서 사용
    try:
        if zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                nc_names = [n for n in z.namelist() if n.lower().endswith((".nc", ".nc4"))]
                if not nc_names:
                    raise ValueError("ZIP 안에 .nc/.nc4 파일이 없습니다.")
                with z.open(nc_names[0]) as f:
                    content = f.read()
    except Exception as e:
        raise e
    return content

content = load_bytes(mode_in, uploaded, url)

if content is None:
    st.info("좌측에서 파일을 업로드하거나 URL을 입력하세요.")
    st.stop()

# ==============================
# 파이프라인 실행 (예외는 화면에 표시)
# ==============================
try:
    ds = open_nc_any(content)

    # 좌표
    lat = ds.lat
    lon = ds.lon

    # 시간 변환(있을 때)
    time_py = None
    try:
        if ds.time is not None:
            # netCDF4가 열었을 때만 num2date 사용 가능. 실패해도 None.
            try:
                from netCDF4 import num2date
                tvar_units = None
                tvar_cal = "standard"
                # netCDF4일 때만 변수에서 직접 읽는다(VarLike에는 없음)
                if hasattr(ds.variables.get("time", object()), "units"):
                    tvar_units = getattr(ds.variables["time"], "units", None)
                if hasattr(ds.variables.get("time", object()), "calendar"):
                    tvar_cal = getattr(ds.variables["time"], "calendar", "standard")
                if tvar_units is not None:
                    tdt = num2date(ds.time, units=tvar_units, calendar=tvar_cal)
                    time_py = to_py_dt(tdt)
            except Exception:
                time_py = None
    except Exception:
        time_py = None

    # 사용할 SST 변수명 탐색
    candidates = ["sst", "analysed_sst", "sea_surface_temperature"]
    sst_name = next((v for v in candidates if v in ds.variables), None)
    if sst_name is None:
        # 좌표 제외, 2D/3D/4D 그리드 첫 변수
        sst_name = next((k for k, v in ds.variables.items()
                         if getattr(v, "shape", ()) and k not in ("lat", "lon", "time")), None)
    if sst_name is None:
        st.error("SST 변수명을 찾지 못했습니다. 파일 변수 목록을 확인하세요.")
        st.write("변수들:", list(ds.variables.keys())[:15])
        st.stop()

    sst_var = ds.variables[sst_name]
    sst_raw = sst_var[:]
    sst = mask_fill(sst_raw, sst_var)
    sst = standardize_to_time_lat_lon(sst_var, sst, lat, lon)  # (time,lat,lon)

    # 경도 정렬
    lon_sorted, sort_idx = wrap_sort_lon(lon)
    sst_sorted = sst[:, :, sort_idx]

    # 메타 정보 표시
    with st.expander("파일 요약", expanded=False):
        st.write(f"SST shape (standardized): {sst.shape} → after sort: {sst_sorted.shape}")
        st.write(f"lat range: {float(np.nanmin(lat))} ~ {float(np.nanmax(lat))}")
        st.write(f"lon range (wrapped): {float(np.nanmin(lon_sorted))} ~ {float(np.nanmax(lon_sorted))}")
        if time_py is not None:
            st.write(f"time: {len(time_py)} steps, {time_py[0].date()} ~ {time_py[-1].date()}")

    # 영역 서브셋
    lat_sub, lon_sub, sst_sub = subset_bbox(lat, lon_sorted, sst_sorted, lat_min, lat_max, lon_min, lon_max)

    # -----------------------
    # 모드별 실행
    # -----------------------
    if mode == "특정 하루":
        ti = int(time_index)
        if ti < 0 or ti >= sst_sub.shape[0]:
            st.error(f"TIME_INDEX {ti} 범위 오류 (0..{sst_sub.shape[0]-1})")
            st.stop()
        label = f"t={ti}" if time_py is None else str(time_py[ti].date())
        s2d = sst_sub[ti]

        col1, col2 = st.columns(2)
        with col1:
            fig = fig_map(lon_sub, lat_sub, s2d, f"SST @ {label}", vmin=sst_vmin, vmax=sst_vmax, cmap="jet")
            st.pyplot(fig)
        with col2:
            mean_reg = float(np.nanmean(s2d))
            rmax = nan_arg_extreme_2d(s2d, "max")
            rmin = nan_arg_extreme_2d(s2d, "min")
            st.subheader("통계")
            st.write(f"영역 평균: **{mean_reg:.3f} °C**")
            if rmax:
                st.write(f"최대: **{rmax[0]:.3f} °C** @ (lat, lon)=({lat_sub[rmax[1]]:.3f}, {lon_sub[rmax[2]]:.3f})")
            if rmin:
                st.write(f"최소: **{rmin[0]:.3f} °C** @ (lat, lon)=({lat_sub[rmin[1]]:.3f}, {lon_sub[rmin[2]]:.3f})")

        # 아노말리(해당 날짜의 영역 평균 기준)
        anom_reg = s2d - mean_reg
        fig = fig_map(lon_sub, lat_sub, anom_reg, f"Anomaly vs Regional Mean @ {label}",
                      cbar="Anomaly (°C)", cmap=anom_cmap)
        st.pyplot(fig)

        # 제공 anom 변수도 있으면 시각화
        if "anom" in ds.variables:
            try:
                anom_var = ds.variables["anom"]
                a = mask_fill(anom_var[:], anom_var)
                a = standardize_to_time_lat_lon(anom_var, a, lat, lon)
                a_sorted = a[:, :, sort_idx]
                _, _, a_sub = subset_bbox(lat, lon_sorted, a_sorted, lat_min, lat_max, lon_min, lon_max)
                a2d = a_sub[ti]
                fig = fig_map(lon_sub, lat_sub, a2d, f"Provided ANOM @ {label}",
                              cbar="Anomaly (°C)", cmap=anom_cmap)
                st.pyplot(fig)
            except Exception as e:
                st.info("※ 'anom' 변수를 읽는 중 문제가 있어 건너뜀.")
                st.exception(e)
        else:
            st.info("※ 이 파일에는 'anom' 변수가 없습니다.")

    elif mode == "전체 기간":
        # 시계열(영역 평균)
        mean_ts = np.nanmean(sst_sub, axis=(1, 2))
        st.subheader("시계열 요약")
        st.write(f"min/mean/max = {float(np.nanmin(mean_ts)):.3f} / {float(np.nanmean(mean_ts)):.3f} / {float(np.nanmax(mean_ts)):.3f} °C")
        fig = fig_ts(to_py_dt(time_py) if time_py else None, mean_ts,
                     f"Regional Mean SST  ({lat_min}-{lat_max}N, {lon_min}-{lon_max}E)")
        st.pyplot(fig)

        # 기간 평균 지도
        mean2d = np.nanmean(sst_sub, axis=0)
        fig = fig_map(lon_sub, lat_sub, mean2d, "Period-Mean SST", vmin=sst_vmin, vmax=sst_vmax, cmap="jet")
        st.pyplot(fig)

    elif mode == "특정 기간(월)":
        if time_py is None:
            st.error("time 좌표가 없어 월 필터를 적용할 수 없습니다.")
            st.stop()
        tp = np.array(time_py)
        months = np.array([d.month for d in tp])
        if m_start <= m_end:
            tmask = (months >= m_start) & (months <= m_end)
        else:
            tmask = (months >= m_start) | (months <= m_end)
        if not np.any(tmask):
            st.warning("선택한 월 범위에 데이터가 없습니다.")
            st.stop()
        sel = sst_sub[tmask]; tp = tp[tmask]

        mean2d = np.nanmean(sel, axis=0)
        fig = fig_map(lon_sub, lat_sub, mean2d, f"Period-Mean SST ({m_start}~{m_end}월)",
                      vmin=sst_vmin, vmax=sst_vmax, cmap="jet")
        st.pyplot(fig)
        st.write("기간 전체 영역 평균:", float(np.nanmean(sel)))

        vmax = float(np.nanmax(sel)); vmin = float(np.nanmin(sel))
        imax = np.nanargmax(sel); t_max, j_max, i_max = np.unravel_index(imax, sel.shape)
        imin = np.nanargmin(sel); t_min, j_min, i_min = np.unravel_index(imin, sel.shape)
        st.write(f"최대 {vmax:.3f} °C @ {tp[t_max].date()}, (lat,lon)=({lat_sub[j_max]:.3f}, {lon_sub[i_max]:.3f})")
        st.write(f"최소 {vmin:.3f} °C @ {tp[t_min].date()}, (lat,lon)=({lat_sub[j_min]:.3f}, {lon_sub[i_min]:.3f})")

        mean_ts_sel = np.nanmean(sel, axis=(1, 2))
        anom_reg = sel - mean_ts_sel[:, None, None]
        fig = fig_map(lon_sub, lat_sub, anom_reg[0],
                      f"Anomaly vs Daily Regional Mean (first day in {m_start}~{m_end}월)",
                      cbar="Anomaly (°C)", cmap=anom_cmap)
        st.pyplot(fig)

        anom_map = sel - mean2d[None, :, :]
        fig = fig_map(lon_sub, lat_sub, anom_map[0],
                      f"Anomaly vs Period-Mean (first day in {m_start}~{m_end}월)",
                      cbar="Anomaly (°C)", cmap=anom_cmap)
        st.pyplot(fig)

    elif mode == "특정 기간(날짜)":
        if time_py is None:
            st.error("time 좌표가 없어 날짜 필터를 적용할 수 없습니다.")
            st.stop()
        try:
            y1, m1, d1 = map(int, date_start.split("-"))
            y2, m2, d2 = map(int, date_end.split("-"))
            start = datetime(y1, m1, d1); end = datetime(y2, m2, d2)
        except Exception:
            st.error("DATE_START/DATE_END 형식이 잘못되었습니다. 예: 2014-03-01")
            st.stop()

        tp = np.array(time_py)
        tmask = (tp >= start) & (tp <= end)
        if not np.any(tmask):
            st.warning("선택한 날짜 범위에 데이터가 없습니다.")
            st.stop()
        sel = sst_sub[tmask]; tp = tp[tmask]

        mean2d = np.nanmean(sel, axis=0)
        fig = fig_map(lon_sub, lat_sub, mean2d, f"Period-Mean SST ({date_start}~{date_end})",
                      vmin=sst_vmin, vmax=sst_vmax, cmap="jet")
        st.pyplot(fig)
        st.write("기간 전체 영역 평균:", float(np.nanmean(sel)))

        vmax = float(np.nanmax(sel)); vmin = float(np.nanmin(sel))
        imax = np.nanargmax(sel); t_max, j_max, i_max = np.unravel_index(imax, sel.shape)
        imin = np.nanargmin(sel); t_min, j_min, i_min = np.unravel_index(imin, sel.shape)
        st.write(f"최대 {vmax:.3f} °C @ {tp[t_max].date()}, (lat,lon)=({lat_sub[j_max]:.3f}, {lon_sub[i_max]:.3f})")
        st.write(f"최소 {vmin:.3f} °C @ {tp[t_min].date()}, (lat,lon)=({lat_sub[j_min]:.3f}, {lon_sub[i_min]:.3f})")

        mean_ts_sel = np.nanmean(sel, axis=(1, 2))
        anom_reg = sel - mean_ts_sel[:, None, None]
        fig = fig_map(lon_sub, lat_sub, anom_reg[0],
                      f"Anomaly vs Daily Regional Mean (first day in {date_start}~{date_end})",
                      cbar="Anomaly (°C)", cmap=anom_cmap)
        st.pyplot(fig)

        anom_map = sel - mean2d[None, :, :]
        fig = fig_map(lon_sub, lat_sub, anom_map[0],
                      f"Anomaly vs Period-Mean (first day in {date_start}~{date_end})",
                      cbar="Anomaly (°C)", cmap=anom_cmap)
        st.pyplot(fig)

    else:
        st.info("모드를 선택하세요.")

except Exception as e:
    st.error("앱 실행 중 예외가 발생했습니다.")
    st.exception(e)

