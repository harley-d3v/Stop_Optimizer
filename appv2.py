import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from geopy.distance import geodesic
import pydeck as pdk
import re

st.set_page_config(layout="wide")

title_col, info_col = st.columns([11, 1])
with title_col:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:7px; margin-bottom:8px;'>
            <img src='https://img.freepik.com/premium-vector/public-bus-location-icon_194117-883.jpg' 
                 width='60' height='60' 
                 style='border-radius:8px;'
                 alt='IMG'>
        <span style='font-size:2.8rem; font-weight:800; line-height:1.15; letter-spacing:-0.5px;'>
            STPH Stop Optimizer
        </span>
    </div>
    """, unsafe_allow_html=True)
with info_col:
    st.markdown("""
    <style>
    div[data-testid="stPopover"] button {
        width: 45px !important;
        height: 45px !important;
        min-height: 38px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin-top: 12px !important;
        cursor: pointer !important;
        transition: opacity 0.2s ease, transform 0.2s ease !important;
        color: transparent !important;
        font-size: 0 !important;
    }
    div[data-testid="stPopover"] button::after {
        content: "" !important;
        display: block !important;
        width: 100px !important;
        height: 100px !important;
        background-image: url("https://cdn-icons-png.freepik.com/256/69/69544.png?semt=ais_hybrid viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='44' fill='none' stroke='%23222' stroke-width='6'/%3E%3Ccircle cx='50' cy='28' r='6' fill='%23222'/%3E%3Cpath d='M44 42 Q38 44 40 48 L44 48 Q44 44 50 43 L50 75 Q44 74 42 78 L58 78 Q56 74 50 75 L50 43 Q56 44 56 48 L60 48 Q62 44 56 42 Z' fill='%23222'/%3E%3C/svg%3E") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
    }
    div[data-testid="stPopover"] button:hover {
        opacity: 0.65 !important;
        transform: scale(1.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.popover("i"):
        st.markdown("""
### About This App
The **STPH Stop Optimizer** analyzes passenger boarding and alighting data to suggest optimal bus stop locations.

---

### Input Files

**Boarding & Alighting CSV** *(required)*
- Must contain coordinate columns: `lon`/`lng`/`longitude` and `lat`/`latitude`
- Must contain boarding/alighting columns: `isBoarding`/`board` and `isAlighting`/`alight` *(case-insensitive)*
- `isBoarding` / `isAlighting` should be `True`/`False`

**Reference Stops CSV** *(optional)*
- Existing bus stop locations to snap clusters to
- Accepted formats:
  - `WKT` column: `POINT (lon lat)`
  - `X` and `Y` columns (longitude, latitude)
  - Optional `name` column for stop labels

---

### How It Works

1. **Upload** your boarding & alighting data
2. **Explore** the map — view passenger density heatmap and color-coded boarding/alighting points
3. **Run Clustering** — HDBSCAN groups nearby passenger activity into candidate stop locations
4. **Snap Threshold** — if reference stops are loaded, clusters within this distance (meters) snap to the nearest existing stop
5. **Download** the final optimized stop list as CSV

---

### Map Legend
| Color | Meaning |
|---|---|
| 🟢 Green | Boarding points / Snapped stops |
| 🔴 Red | Alighting points / New proposed stops |
| 🟠 Orange | Reference stops |
| Gradient | Passenger density heatmap |
        """)

# ----------------------------
# Floating Feedback Button (Google Form)
# ----------------------------
GOOGLE_FORM_URL = "https://forms.gle/W9w8CmaCeVzCxNir8"

st.markdown(f"""
<style>
.feedback-fab {{
    position: fixed;
    top: 65px;
    right: 15px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    pointer-events: auto;
}}
.feedback-fab a {{
    display: flex;
    align-items: center;
    gap: 9px;
    background: #0496C7;
    color: #ffffff !important;
    text-decoration: none !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 11px 20px 11px 15px;
    border-radius: 50px;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.45), 0 1px 3px rgba(0,0,0,0.25);
    transition: transform 0.18s cubic-bezier(.34,1.56,.64,1), box-shadow 0.18s ease, background 0.18s ease;
    border: 1.5px solid rgba(255,255,255,0.18);
    white-space: nowrap;
}}
.feedback-fab a:hover {{
    transform: translateY(-3px) scale(1.04);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.55), 0 2px 6px rgba(0,0,0,0.2);
    background: #1a3fc4;
    color: #ffffff !important;
    text-decoration: none !important;
}}
.feedback-fab .fab-icon {{
    font-size: 16px;
    line-height: 1;
    flex-shrink: 0;
}}
@keyframes fab-slide-in {{
    from {{ opacity: 0; transform: translateY(20px) scale(0.9); }}
    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
.feedback-fab {{
    animation: fab-slide-in 0.5s cubic-bezier(.34,1.56,.64,1) 0.8s both;
}}
</style>
<div class="feedback-fab">
    <div style="position:relative;display:inline-block;">
        <a href="{GOOGLE_FORM_URL}" target="_blank" rel="noopener noreferrer">
            <span class="fab-icon">💬</span>
            Give Feedback
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Uploader styling — taller drop zones
# ----------------------------
st.markdown("""
<style>
[data-testid="stFileUploader"] section {
    min-height: 140px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: 2px dashed rgba(120,130,160,0.55) !important;
    border-radius: 12px !important;
    background: rgba(30, 40, 60, 0.18) !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
    padding: 20px 16px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(99, 140, 255, 0.85) !important;
    background: rgba(40, 60, 100, 0.28) !important;
}
[data-testid="stFileUploader"] section > div {
    text-align: center !important;
}
[data-testid="stFileUploader"] label p {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Side-by-side uploaders
# ----------------------------
upload_left, upload_right = st.columns(2)

with upload_left:
    st.markdown("##### 📄 Boarding & Alighting Data *(required)*")
    uploaded_file = st.file_uploader(
        "Upload a CSV with boarding/alighting points",
        type=["csv"],
        key="ba_upload",
        label_visibility="collapsed",
    )

with upload_right:
    st.markdown("##### 📍 Reference Stops *(optional)*")
    reference_file = st.file_uploader(
        "Upload a CSV with existing bus stop locations",
        type=["csv"],
        key="reference",
        label_visibility="collapsed",
    )

# ----------------------------
# Inject a legend that floats OVER the pydeck map
# ----------------------------
LEGEND_STYLE = """
<style>
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]
  > div[data-testid="element-container"] > div.stMarkdown > div > div.map-legend-float {
    position: relative;
    z-index: 100;
    pointer-events: none;
}
.map-legend-float {
    margin-top: -440px;
    margin-right: 10px;
    float: right;
    display: inline-block;
    background: rgba(15, 20, 30, 0.75);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    color: #f0f0f0;
    padding: 7px 10px;
    border-radius: 7px;
    font-size: 11px;
    font-family: sans-serif;
    line-height: 1.4;
    border: 1px solid rgba(255,255,255,0.1);
    min-width: 0;
    pointer-events: none;
}
.map-legend-float .legend-title {
    font-weight: 700;
    font-size: 9px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 6px;
}
.map-legend-float .legend-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.map-legend-float .legend-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.map-legend-float .legend-label { font-size: 11px; }
.map-legend-float .legend-desc  { display: none; }
.map-legend-float .legend-divider {
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 5px 0;
}
.map-legend-float .gradient-bar {
    width: 60px; height: 7px; border-radius: 2px; flex-shrink: 0;
    background: linear-gradient(90deg,
        rgba(0,0,255,.7), rgba(0,200,255,.8), rgba(0,255,100,.85),
        rgba(255,255,0,.9), rgba(255,140,0,.95), rgba(255,0,0,1));
}
</style>
"""

def make_ba_legend(show_points, show_heatmap):
    rows = ""
    if show_points:
        rows += """
        <div class='legend-row'>
            <span class='legend-dot' style='background:#22c55e'></span>
            <div><div class='legend-label'>Boarding</div>
                 <div class='legend-desc'>Passengers getting on the bus</div></div>
        </div>
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ef4444'></span>
            <div><div class='legend-label'>Alighting</div>
                 <div class='legend-desc'>Passengers getting off the bus</div></div>
        </div>"""

    if show_heatmap:
        rows += """
        <div class='legend-divider'></div>
        <div><div class='legend-label' style='margin-bottom:4px;'>Density</div>
             <div style='display:flex;align-items:center;gap:4px;'>
                 <div class='gradient-bar'></div>
             </div>
        </div>"""
    return f"<div class='map-legend-float'><div class='legend-title'>Legend</div>{rows}</div>"


def make_opt_legend(has_reference):
    if has_reference:
        rows = """
        <div class='legend-row'>
            <span class='legend-dot' style='background:#00ff00'></span>
            <div><div class='legend-label'>Snapped to Reference</div>
                 <div class='legend-desc'>Matched to existing stop within snap threshold</div></div>
        </div>
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ff4444'></span>
            <div><div class='legend-label'>New Optimized Stop</div>
                 <div class='legend-desc'>Proposed stop — no nearby reference found</div></div>
        </div>
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ffa500'></span>
            <div><div class='legend-label'>Reference Stop</div>
                 <div class='legend-desc'>Existing stop from uploaded reference dataset</div></div>
        </div>"""
    else:
        rows = """
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ff4444'></span>
            <div><div class='legend-label'>Optimized Stop</div>
                 <div class='legend-desc'>Proposed stop derived from passenger clusters</div></div>
        </div>"""
    return f"<div class='map-legend-float'><div class='legend-title'>Legend</div>{rows}</div>"


# ----------------------------
# Helper: Flexible column resolver
# ----------------------------
def find_column(df_columns, candidates):
    col_map = {c.lower(): c for c in df_columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def parse_bool_col(series):
    """Robustly parse a column into boolean values."""
    if series.dtype == object:
        normalized = series.str.strip().str.lower()
        return normalized.isin(['true', '1', '1.0', 'yes'])
    elif series.dtype in [np.float64, np.float32]:
        return series.fillna(0).astype(int).astype(bool)
    else:
        return series.fillna(0).astype(bool)


def resolve_coords_and_flags(df):
    cols = df.columns.tolist()

    lon_col    = find_column(cols, ['lon', 'lng', 'longitude', 'long','location.coordinates[0]'])
    lat_col    = find_column(cols, ['lat', 'latitude','location.coordinates[1]'])
    board_col  = find_column(cols, ['isBoarding', 'boarding', 'board', 'is_boarding'])
    alight_col = find_column(cols, ['isAlighting', 'alighting', 'alight', 'is_alighting'])

    missing = []
    if lon_col   is None: missing.append("longitude (tried: lon, lng, longitude, long)")
    if lat_col   is None: missing.append("latitude (tried: lat, latitude)")
    if board_col is None: missing.append("boarding (tried: isBoarding, boarding, board)")
    # isAlighting is NOT required — inferred as inverse of isBoarding if absent

    if missing:
        return None, "Could not find required columns:\n- " + "\n- ".join(missing)

    df = df.copy()
    df['lon'] = df[lon_col]
    df['lat'] = df[lat_col]
    df['isBoarding'] = parse_bool_col(df[board_col])

    # Use isAlighting column if it exists and is distinct from isBoarding column.
    # If both columns end up all-True (common parsing error), fall back to inference.
    if alight_col is not None and alight_col != board_col:
        df['isAlighting'] = parse_bool_col(df[alight_col])
        if df['isBoarding'].all() and df['isAlighting'].all():
            # Both all-True means something went wrong — infer instead
            df['isAlighting'] = ~df['isBoarding']
    else:
        # No separate alighting column found — alighting = NOT boarding
        df['isAlighting'] = ~df['isBoarding']

    return df, None


# ----------------------------
# Process Reference Stops
# ----------------------------
reference_stops = None

if reference_file is not None:
    reference_df = pd.read_csv(reference_file)
    if 'WKT' in reference_df.columns:
        def parse_wkt(wkt_string):
            match = re.search(r'POINT \(([0-9.-]+) ([0-9.-]+)\)', str(wkt_string))
            if match:
                return float(match.group(1)), float(match.group(2))
            return None, None
        reference_df[['ref_lon', 'ref_lat']] = reference_df['WKT'].apply(
            lambda x: pd.Series(parse_wkt(x))
        )
    elif 'X' in reference_df.columns and 'Y' in reference_df.columns:
        reference_df['ref_lon'] = reference_df['X']
        reference_df['ref_lat'] = reference_df['Y']
    else:
        st.error("Reference file must contain either 'WKT' or 'X'/'Y' columns")

    if 'ref_lon' in reference_df.columns:
        reference_df = reference_df.dropna(subset=['ref_lon', 'ref_lat'])
        if 'name' not in reference_df.columns:
            reference_df['name'] = [f"Reference Stop {i+1}" for i in range(len(reference_df))]
        reference_stops = reference_df[['ref_lon', 'ref_lat', 'name']].copy()
        with upload_right:
            st.success(f"✅ {len(reference_stops)} reference stops loaded")


# ----------------------------
# Main App
# ----------------------------
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    df, col_error = resolve_coords_and_flags(raw_df)
    if col_error:
        st.error(col_error)
        st.stop()

    df = df[(df['isBoarding'] == True) | (df['isAlighting'] == True)]
    df = df.dropna(subset=['lon', 'lat'])

    boarding_df  = df[df['isBoarding']  == True].copy()
    alighting_df = df[df['isAlighting'] == True].copy()

    with upload_left:
        st.success(f"✅ Data loaded — {len(df)} passenger events")

    MAP_STYLES = {
        "Road (built-in)": "road",
        "Carto Dark":      "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Carto Light":     "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    }

    # ----------------------------
    # Section 1: Boarding & Alighting Map
    # ----------------------------
    st.markdown("""
    <style>
    details > summary p {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.expander("Passenger Activity Visual Map", expanded=True):
        st.markdown("#### 📍 Boarding & Alighting Points")
        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown("**Map Options**")
            map_style_choice   = st.selectbox("Map Style", list(MAP_STYLES.keys()), index=0)
            selected_map_style = MAP_STYLES[map_style_choice]
            show_points        = st.checkbox("Show B/A Points",  value=True)
            show_heatmap       = st.checkbox("Show Heatmap", value=True)
            heatmap_intensity  = st.slider("Heatmap Intensity",   1,  20,  5)
            heatmap_radius     = st.slider("Heatmap Radius (px)", 10, 80, 30)

        layers = []
        if show_heatmap:
            layers.append(pdk.Layer(
                "HeatmapLayer", data=df[['lon', 'lat']].copy(),
                get_position=['lon', 'lat'], aggregation="SUM",
                intensity=heatmap_intensity, radius_pixels=heatmap_radius,
                threshold=0.05, pickable=False,
                color_range=[
                    [0,   0,   255, 120], [0,   200, 255, 160],
                    [0,   255, 100, 180], [255, 255, 0,   200],
                    [255, 140, 0,   220], [255, 0,   0,   255],
                ],
            ))
        if show_points:

            if len(boarding_df) > 0:
                layers.append(pdk.Layer(
                "ScatterplotLayer", data=boarding_df,
                get_position=['lon', 'lat'], get_radius=25,
                get_fill_color=[34, 197, 94, 200],
                get_line_color=[22, 163, 74, 255],  # darker green border
                stroked=True, line_width_min_pixels=1,
                pickable=True,
            ))
            
            if len(alighting_df) > 0:
                layers.append(pdk.Layer(
                "ScatterplotLayer", data=alighting_df,
                get_position=['lon', 'lat'], get_radius=25,
                get_fill_color=[239, 68, 68, 200],
                get_line_color=[185, 28, 28, 255],  # darker red border
                stroked=True, line_width_min_pixels=1,
                pickable=True,
            ))

        with col1:
            st.markdown(LEGEND_STYLE, unsafe_allow_html=True)
            st.pydeck_chart(pdk.Deck(
                map_style=selected_map_style,
                initial_view_state=pdk.ViewState(
                    latitude=df['lat'].mean(), longitude=df['lon'].mean(),
                    zoom=12, pitch=0,
                ),
                layers=layers,
                tooltip={"text": "Lat: {lat}\nLon: {lon}"},
            ))
            st.markdown(make_ba_legend(show_points, show_heatmap), unsafe_allow_html=True)

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Total Passenger Events", len(df))
        with stat_col2:
            st.metric("🟢 Boarding", len(boarding_df))
        with stat_col3:
            st.metric("🔴 Alighting", len(alighting_df))

    # ----------------------------
    # Run Clustering button
    # ----------------------------
    # Determine button state for rendering
    is_done    = st.session_state.get('clustering_done', False)
    is_running = st.session_state.get('clustering_running', False)

    st.markdown("""
    <style>
    /* Base cluster button */
    .cluster-btn > div[data-testid="stButton"] > button {
        width: 100% !important;
        height: 54px !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
        letter-spacing: 0.03em;
        position: relative;
        overflow: hidden;
        transition: background 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
    }

    /* Running state — animated shimmer */
    .cluster-btn-running > div[data-testid="stButton"] > button {
        width: 100% !important;
        height: 54px !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
        letter-spacing: 0.03em;
        background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 40%, #1d4ed8 60%, #1d4ed8 100%) !important;
        background-size: 200% 100% !important;
        animation: shimmer 1.2s infinite linear !important;
        color: white !important;
        border: none !important;
        cursor: not-allowed !important;
    }
    @keyframes shimmer {
        0%   { background-position: 200% center; }
        100% { background-position: -200% center; }
    }

    /* Done state — green */
    .cluster-btn-done > div[data-testid="stButton"] > button {
        width: 100% !important;
        height: 54px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
        letter-spacing: 0.03em;
        background: #16a34a !important;
        color: white !important;
        border: none !important;
    }
    .cluster-btn-done > div[data-testid="stButton"] > button:hover {
        background: #15803d !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if is_done:
            btn_class = "cluster-btn-done"
            btn_label = "✅ Clustering Complete"
        else:
            btn_class = "cluster-btn"
            btn_label = "🔄 Run Clustering"

        st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
        run_clustering = st.button(btn_label, use_container_width=True, key="cluster_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    if run_clustering:
        st.session_state['clustering_done']    = False
        st.session_state['clustering_running'] = True

        # Show shimmer button while working
        _, btn_col2, _ = st.columns([1, 2, 1])
        with btn_col2:
            st.markdown('<div class="cluster-btn-running">', unsafe_allow_html=True)
            st.button("⏳ Clustering...", use_container_width=True, key="cluster_btn_running", disabled=True)
            st.markdown('</div>', unsafe_allow_html=True)

        coordinates = df[['lon', 'lat']].values
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=4)
        df['Cluster'] = clusterer.fit_predict(np.radians(coordinates))
        df = df[df['Cluster'] != -1]
        centroids = df.groupby('Cluster')[['lon', 'lat']].mean().reset_index()

        def filter_min_distance(centroids_df, min_distance_m=300):
            kept = []
            for _, row in centroids_df.iterrows():
                point = (row['lat'], row['lon'])
                if all(geodesic(point, (k['lat'], k['lon'])).meters >= min_distance_m for k in kept):
                    kept.append(row)
            return pd.DataFrame(kept)

        st.session_state['filtered_centroids']  = filter_min_distance(centroids, 200)
        st.session_state['n_clusters']          = df['Cluster'].nunique()
        st.session_state['n_points']            = len(df)
        st.session_state['clustering_done']     = True
        st.session_state['clustering_running']  = False
        st.rerun()

    # ----------------------------
    # Section 2: Optimized Stops
    # ----------------------------
    st.markdown("""
        <style>
        details > summary p {
            font-size: 1.8rem !important;
            font-weight: 600 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if 'filtered_centroids' in st.session_state:
        with st.expander("Optimized Stops Visual Map", expanded=True):
            st.markdown("#### Optimized Stops")
            filtered_centroids = st.session_state['filtered_centroids']

            st.caption(f"Clustered points: {st.session_state['n_points']}  |  Clusters found: {st.session_state['n_clusters']}")

            if reference_stops is not None:
                snap_threshold = st.slider("Snap Threshold (meters)", 50, 500, 150, 25)
                snapped_centroids, unsnapped_centroids = [], []
                for _, centroid in filtered_centroids.iterrows():
                    centroid_point = (centroid['lat'], centroid['lon'])
                    min_distance = float('inf')
                    nearest_ref = None
                    for _, ref in reference_stops.iterrows():
                        distance = geodesic(centroid_point, (ref['ref_lat'], ref['ref_lon'])).meters
                        if distance < min_distance:
                            min_distance = distance
                            nearest_ref = ref
                    if min_distance <= snap_threshold:
                        snapped_centroids.append({
                            'Cluster': centroid['Cluster'],
                            'lon': nearest_ref['ref_lon'], 'lat': nearest_ref['ref_lat'],
                            'snapped': True, 'snap_distance': min_distance,
                            'reference_name': nearest_ref['name']
                        })
                    else:
                        unsnapped_centroids.append({
                            'Cluster': centroid['Cluster'],
                            'lon': centroid['lon'], 'lat': centroid['lat'],
                            'snapped': False, 'snap_distance': None, 'reference_name': None
                        })
                final_centroids = pd.DataFrame(snapped_centroids + unsnapped_centroids)
                st.caption(f" Snapped: {len(snapped_centroids)}  |  New stops: {len(unsnapped_centroids)}  |  Total: {len(final_centroids)}")
            else:
                final_centroids = filtered_centroids.copy()
                final_centroids['snapped'] = False
                st.caption(f"Total optimized stops: {len(final_centroids)}")

            if len(final_centroids) > 0:
                has_ref = reference_stops is not None

                # ── Side-by-side: map (left) + layer toggles (right) ──────
                opt_map_col, opt_ctrl_col = st.columns([3, 1])

                with opt_ctrl_col:
                    st.markdown("**LAYER LEGEND**")
                    show_new = st.checkbox(
                        "🔴 Unsnapped Stops" if has_ref else "🔴 Unsnapped Stops",
                        value=True, key="show_new_stops"
                    )
                    if has_ref:
                        show_snapped = st.checkbox("🟢 Snapped to Reference Stop", value=True, key="show_snapped")
                        show_ref     = st.checkbox("🟠 Reference Stops",       value=True, key="show_ref_stops")
                    else:
                        show_snapped = False
                        show_ref     = False

                # Rebuild layers based on checkbox state
                opt_layers_filtered = []
                if has_ref and show_ref:
                    opt_layers_filtered.append(pdk.Layer("ScatterplotLayer", data=reference_stops,
                    get_position=['ref_lon', 'ref_lat'],
                    get_fill_color=[255, 165, 0, 200],
                    get_line_color=[200, 110, 0, 255],
                    stroked=True, line_width_min_pixels=1,
                    get_radius=35, pickable=True,
                    ))
                if 'snapped' in final_centroids.columns:
                    snapped_df   = final_centroids[final_centroids['snapped'] == True]
                    unsnapped_df = final_centroids[final_centroids['snapped'] == False]
                    if has_ref and show_snapped and len(snapped_df) > 0:
                        opt_layers_filtered.append(pdk.Layer(
                            "ScatterplotLayer", data=snapped_df,
                            get_position=['lon', 'lat'],
                            get_color=[0, 255, 0, 200],
                            get_fill_color=[0, 200, 80, 200],
                            get_line_color=[0, 150, 50, 255],
                            stroked=True, line_width_min_pixels=1,
                            get_radius=45, pickable=True,
                        ))
                if show_new and len(unsnapped_df) > 0:
                        opt_layers_filtered.append(pdk.Layer("ScatterplotLayer", data=unsnapped_df,
                            get_position=['lon', 'lat'],
                            get_fill_color=[239, 68, 68, 200],
                            get_line_color=[185, 28, 28, 255],
                            stroked=True, line_width_min_pixels=1,
                            get_radius=45, pickable=True,
                        ))
                else:
                    if show_new:
                        opt_layers_filtered.append(pdk.Layer(
                            "ScatterplotLayer", data=final_centroids,
                            get_position=['lon', 'lat'], get_color=[255, 0, 0, 200], get_radius=45,
                        ))

                with opt_map_col:
                    st.pydeck_chart(pdk.Deck(
                        map_style=selected_map_style,
                        initial_view_state=pdk.ViewState(
                            latitude=final_centroids['lat'].mean(),
                            longitude=final_centroids['lon'].mean(),
                            zoom=13, pitch=0,
                        ),
                        layers=opt_layers_filtered,
                    ))

                # ── Centered Download button ────────────────────────────────
                csv = final_centroids.to_csv(index=False).encode('utf-8')
                dl_done = st.session_state.get('download_done', False)

                st.markdown("""
                <style>
                .dl-btn > div[data-testid="stDownloadButton"] > button {
                    width: 100% !important;
                    height: 54px !important;
                    font-size: 18px !important;
                    font-weight: 700 !important;
                    border-radius: 30px !important;
                    letter-spacing: 0.03em;
                    transition: background 0.3s ease, color 0.3s ease !important;
                }
                .dl-btn-running > div[data-testid="stDownloadButton"] > button {
                    width: 100% !important;
                    height: 54px !important;
                    font-size: 18px !important;
                    font-weight: 700 !important;
                    border-radius: 30px !important;
                    letter-spacing: 0.03em;
                    background: linear-gradient(90deg, #0369a1 0%, #38bdf8 40%, #0369a1 60%, #0369a1 100%) !important;
                    background-size: 200% 100% !important;
                    animation: dl-shimmer 1.2s infinite linear !important;
                    color: white !important;
                    border: none !important;
                }
                @keyframes dl-shimmer {
                    0%   { background-position: 200% center; }
                    100% { background-position: -200% center; }
                }
                .dl-btn-done > div[data-testid="stDownloadButton"] > button {
                    width: 100% !important;
                    height: 54px !important;
                    font-size: 18px !important;
                    font-weight: 700 !important;
                    border-radius: 30px !important;
                    letter-spacing: 0.03em;
                    background: #16a34a !important;
                    color: white !important;
                    border: none !important;
                }
                .dl-btn-done > div[data-testid="stDownloadButton"] > button:hover {
                    background: #15803d !important;
                    color: white !important;
                    border: none !important;
                }
                </style>
                """, unsafe_allow_html=True)

                _, dl_col, _ = st.columns([1, 2, 1])
                with dl_col:
                    dl_class = "dl-btn-done" if dl_done else "dl-btn"
                    dl_label = "✅ Download Successful" if dl_done else "⬇️ Download Optimized Stops CSV"
                    st.markdown(f'<div class="{dl_class}">', unsafe_allow_html=True)
                    clicked = st.download_button(
                        dl_label, csv,
                        "optimized_stops.csv", "text/csv",
                        use_container_width=True,
                        key="dl_btn",
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    if clicked:
                        st.session_state['download_done'] = True
                        st.rerun()
            else:
                st.warning("No centroids found after filtering!")

else:
    st.info("Please upload a Boarding & Alighting CSV file to begin.")