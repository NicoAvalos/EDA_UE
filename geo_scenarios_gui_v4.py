# -*- coding: utf-8 -*-
"""
GeoScenarios GUI — v0.4
Mejoras agregadas sobre v0.2:
- Filtro numérico previo a todo análisis (columna, comparador, valor)
- Mínimo de muestras por grupo (min_count)
- Ejes fijos para Swath (y opción de estandarizar X en ECDF)
    * Ámbito de ejes: por "set" (lvl1,lvl2) o "global"
    * Clip por percentil simétrico (p%, 0=off) para reducir outliers

Requisitos: pandas, numpy, matplotlib, tkinter
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.max_open_warning'] = 0


APP_TITLE = "GeoScenarios GUI — Lit/Alt/Mnz"
APP_VERSION = "v0.4"


def ecdf(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.array([]), np.array([])
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / float(len(x))
    return x, y


def swath_bins(coord, values, nbins=30, agg="mean"):
    coord = np.asarray(coord, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = ~np.isnan(coord) & ~np.isnan(values)
    coord = coord[mask]
    values = values[mask]
    if coord.size == 0:
        return np.array([]), np.array([]), np.array([])
    lo, hi = np.min(coord), np.max(coord)
    if lo == hi:
        return np.array([lo]), np.array([np.nanmean(values)]), np.array([len(values)])
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(coord, edges) - 1, 0, nbins - 1)
    bin_vals = [[] for _ in range(nbins)]
    for i, v in zip(idx, values):
        bin_vals[i].append(v)
    agg_vals, counts, centers = [], [], []
    for b in range(nbins):
        arr = np.asarray(bin_vals[b], dtype=float)
        if arr.size == 0:
            agg_vals.append(np.nan)
            counts.append(0)
        else:
            if agg == "mean":
                agg_vals.append(np.nanmean(arr))
            elif agg == "median":
                agg_vals.append(np.nanmedian(arr))
            elif agg == "p75":
                agg_vals.append(np.nanpercentile(arr, 75))
            else:
                agg_vals.append(np.nanmean(arr))
            counts.append(arr.size)
        centers.append((edges[b] + edges[b + 1]) / 2.0)
    return np.asarray(centers), np.asarray(agg_vals), np.asarray(counts)


class GeoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} — {APP_VERSION}")
        self.geometry("1150x780")
        self.minsize(1100, 700)

        self.csv_path = None
        self.df = None

        self.sep_var = tk.StringVar(value=",")
        self.easting_col = tk.StringVar()
        self.northing_col = tk.StringVar()
        self.elevation_col = tk.StringVar()

        self.scenario_level1 = tk.StringVar()
        self.scenario_level2 = tk.StringVar()
        self.scenario_level3 = tk.StringVar()

        self.swath_axis = tk.StringVar(value="Easting")
        self.swath_agg = tk.StringVar(value="mean")
        self.swath_bins_var = tk.IntVar(value=30)

        self.numeric_target = tk.StringVar()
        self.scatter_x = tk.StringVar()
        self.scatter_y = tk.StringVar()
        self.scatter_hue = tk.StringVar()

        # Column sets
        self.pool_cols = []
        self.categorical_cols = []
        self.numeric_cols = []

        # New: analysis controls
        self.min_count = tk.IntVar(value=30)
        self.fix_axes = tk.BooleanVar(value=True)
        self.axis_scope = tk.StringVar(value="set")  # 'set' or 'global'
        self.y_clip_pct = tk.DoubleVar(value=0.0)    # 0 = off, 1..10 typical

        # New: numeric filter
        self.filter_col = tk.StringVar()
        self.filter_op = tk.StringVar(value=">=")
        self.filter_val = tk.DoubleVar(value=0.0)

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="CSV:").pack(side="left")
        self.file_entry = ttk.Entry(top, width=70)
        self.file_entry.pack(side="left", padx=4)
        ttk.Button(top, text="Seleccionar...", command=self.select_csv).pack(side="left", padx=4)
        ttk.Label(top, text="Separador:").pack(side="left", padx=(12,2))
        self.sep_combo = ttk.Combobox(top, textvariable=self.sep_var, values=[",", ";", "|", "\\t"], width=5)
        self.sep_combo.pack(side="left")
        ttk.Button(top, text="Cargar CSV", command=self.load_csv).pack(side="left", padx=8)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab_cols = ttk.Frame(nb, padding=8)
        tab_scen = ttk.Frame(nb, padding=8)
        nb.add(tab_cols, text="1) Columnas")
        nb.add(tab_scen, text="2) Escenario y Salidas")

        # ----- Tab 1: Columns
        left = ttk.Frame(tab_cols)
        left.pack(side="left", fill="y", padx=(0,8))

        g_coords = ttk.LabelFrame(left, text="Coordenadas", padding=8)
        g_coords.pack(fill="x", pady=(0,8))

        self.combo_east = ttk.Combobox(g_coords, textvariable=self.easting_col, state="readonly")
        self.combo_north = ttk.Combobox(g_coords, textvariable=self.northing_col, state="readonly")
        self.combo_elev = ttk.Combobox(g_coords, textvariable=self.elevation_col, state="readonly")
        ttk.Label(g_coords, text="Easting (X):").pack(anchor="w")
        self.combo_east.pack(fill="x", pady=2)
        ttk.Label(g_coords, text="Northing (Y):").pack(anchor="w")
        self.combo_north.pack(fill="x", pady=2)
        ttk.Label(g_coords, text="Elevation (Z):").pack(anchor="w")
        self.combo_elev.pack(fill="x", pady=2)

        mid = ttk.Frame(tab_cols)
        mid.pack(side="left", fill="both", expand=True)

        g_pool = ttk.LabelFrame(mid, text="Todas las columnas (doble clic para mover)")
        g_pool.pack(side="left", fill="both", expand=True, padx=(0,8))

        fbar = ttk.Frame(g_pool)
        fbar.pack(fill="x", padx=6, pady=4)
        ttk.Label(fbar, text="Filtro:").pack(side="left")
        self.filter_text_var = tk.StringVar()
        self.filter_entry = ttk.Entry(fbar, textvariable=self.filter_text_var)
        self.filter_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.filter_entry.bind("<KeyRelease>", lambda e: self.refresh_lists())

        self.list_pool = tk.Listbox(g_pool, selectmode="extended")
        self.list_pool.pack(fill="both", expand=True, padx=6, pady=6)
        self.list_pool.bind("<Double-Button-1>", self._dbl_to_cat)

        ctrls = ttk.Frame(mid)
        ctrls.pack(side="left", fill="y")
        ttk.Button(ctrls, text="→ Categórica", command=self.move_to_cat).pack(pady=(40,6))
        ttk.Button(ctrls, text="→ Numérica", command=self.move_to_num).pack(pady=6)
        ttk.Button(ctrls, text="← Quitar", command=self.remove_from_lists).pack(pady=(6,40))

        right = ttk.Frame(tab_cols)
        right.pack(side="left", fill="both", expand=True)

        g_cat = ttk.LabelFrame(right, text="Categóricas (doble clic para quitar)")
        g_cat.pack(fill="both", expand=True, padx=(0,8), pady=(0,6))
        self.list_cat = tk.Listbox(g_cat, selectmode="extended")
        self.list_cat.pack(fill="both", expand=True, padx=6, pady=6)
        self.list_cat.bind("<Double-Button-1>", self._dbl_remove_cat)

        g_num = ttk.LabelFrame(right, text="Numéricas (doble clic para quitar)")
        g_num.pack(fill="both", expand=True, padx=(0,8), pady=(6,0))
        self.list_num = tk.Listbox(g_num, selectmode="extended")
        self.list_num.pack(fill="both", expand=True, padx=6, pady=6)
        self.list_num.bind("<Double-Button-1>", self._dbl_remove_num)

        btns = ttk.Frame(tab_cols)
        btns.pack(side="bottom", fill="x", pady=8)
        ttk.Button(btns, text="Guardar selección", command=self.save_selection).pack(side="left")

        # ----- Tab 2: Scenario & outputs
        g_scen = ttk.LabelFrame(tab_scen, text="Escenario (jerarquía)", padding=8)
        g_scen.pack(fill="x")

        self.scen1 = ttk.Combobox(g_scen, textvariable=self.scenario_level1, state="readonly")
        self.scen2 = ttk.Combobox(g_scen, textvariable=self.scenario_level2, state="readonly")
        self.scen3 = ttk.Combobox(g_scen, textvariable=self.scenario_level3, state="readonly")
        ttk.Label(g_scen, text="Nivel 1:").grid(row=0, column=0, sticky="w")
        self.scen1.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(g_scen, text="Nivel 2:").grid(row=1, column=0, sticky="w")
        self.scen2.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(g_scen, text="Nivel 3:").grid(row=2, column=0, sticky="w")
        self.scen3.grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        g_scen.columnconfigure(1, weight=1)

        g_opts = ttk.LabelFrame(tab_scen, text="Opciones de salida", padding=8)
        g_opts.pack(fill="x", pady=8)

        self.combo_target = ttk.Combobox(g_opts, textvariable=self.numeric_target, state="readonly")
        ttk.Label(g_opts, text="Variable numérica (N) para ECDF / Swath:").grid(row=0, column=0, sticky="w")
        self.combo_target.grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(g_opts, text="Swath eje:").grid(row=1, column=0, sticky="w")
        self.combo_axis = ttk.Combobox(g_opts, textvariable=self.swath_axis, values=["Easting", "Northing", "Elevation"], state="readonly", width=12)
        self.combo_axis.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(g_opts, text="Agregado:").grid(row=1, column=2, sticky="w")
        self.combo_agg = ttk.Combobox(g_opts, textvariable=self.swath_agg, values=["mean", "median", "p75"], state="readonly", width=10)
        self.combo_agg.grid(row=1, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(g_opts, text="Bins:").grid(row=1, column=4, sticky="w")
        self.entry_bins = ttk.Entry(g_opts, textvariable=self.swath_bins_var, width=6)
        self.entry_bins.grid(row=1, column=5, sticky="w", padx=4, pady=2)

        # New controls row
        ttk.Label(g_opts, text="Mín. muestras/grupo:").grid(row=2, column=0, sticky="w")
        ttk.Entry(g_opts, textvariable=self.min_count, width=8).grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Checkbutton(g_opts, text="Ejes fijos", variable=self.fix_axes).grid(row=2, column=2, sticky="w")

        ttk.Label(g_opts, text="Ámbito ejes:").grid(row=2, column=3, sticky="w")
        ttk.Combobox(g_opts, textvariable=self.axis_scope, values=["set","global"], state="readonly", width=8)\
            .grid(row=2, column=4, sticky="w", padx=4)

        ttk.Label(g_opts, text="Clip p% (Y/ECDF X):").grid(row=2, column=5, sticky="w")
        ttk.Entry(g_opts, textvariable=self.y_clip_pct, width=6).grid(row=2, column=6, sticky="w", padx=4)

        # Numeric filter
        g_filter = ttk.LabelFrame(tab_scen, text="Filtro base numérica (aplicado antes del análisis)", padding=8)
        g_filter.pack(fill="x", pady=4)

        self.combo_filter_col = ttk.Combobox(g_filter, textvariable=self.filter_col, state="readonly")
        ttk.Label(g_filter, text="Columna:").grid(row=0, column=0, sticky="w")
        self.combo_filter_col.grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        self.combo_filter_op = ttk.Combobox(g_filter, textvariable=self.filter_op, values=[">=", ">", "<=", "<", "==", "!="], state="readonly", width=6)
        ttk.Label(g_filter, text="Operador:").grid(row=0, column=2, sticky="w")
        self.combo_filter_op.grid(row=0, column=3, sticky="w", padx=4, pady=2)

        self.entry_filter_val = ttk.Entry(g_filter, textvariable=self.filter_val, width=10)
        ttk.Label(g_filter, text="Valor:").grid(row=0, column=4, sticky="w")
        self.entry_filter_val.grid(row=0, column=5, sticky="w", padx=4, pady=2)

        g_filter.columnconfigure(1, weight=1)

        # Actions
        actions = ttk.Frame(tab_scen)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Generar ECDF (probability plot)", command=self.action_ecdf).pack(side="left", padx=4)
        ttk.Button(actions, text="Generar Swath plot", command=self.action_swath).pack(side="left", padx=4)
        ttk.Button(actions, text="Exportar estadísticas (min/max/mean/p75)", command=self.action_stats).pack(side="left", padx=4)
        ttk.Button(actions, text="Scatter plot", command=self.action_scatter).pack(side="left", padx=4)

        self.status = tk.StringVar(value="Listo.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=2)

    # ---------- Column moving helpers ----------
    def refresh_lists(self):
        if self.df is None:
            return
        cols = list(self.df.columns)
        chosen = set(self.categorical_cols + self.numeric_cols + [self.easting_col.get(), self.northing_col.get(), self.elevation_col.get()])
        flt = self.filter_text_var.get().strip().lower()
        pool = [c for c in cols if c not in chosen and (flt in c.lower())]
        self.list_pool.delete(0, "end")
        for c in pool:
            self.list_pool.insert("end", c)
        self.list_cat.delete(0, "end")
        for c in self.categorical_cols:
            self.list_cat.insert("end", c)
        self.list_num.delete(0, "end")
        for c in self.numeric_cols:
            self.list_num.insert("end", c)

    def _get_selected(self, lb):
        return [lb.get(i) for i in lb.curselection()]

    def move_to_cat(self):
        sel = self._get_selected(self.list_pool)
        for c in sel:
            if c not in self.categorical_cols:
                self.categorical_cols.append(c)
        self.refresh_lists()

    def move_to_num(self):
        sel = self._get_selected(self.list_pool)
        for c in sel:
            if c not in self.numeric_cols:
                self.numeric_cols.append(c)
        self.refresh_lists()

    def remove_from_lists(self):
        sel_cat = self._get_selected(self.list_cat)
        sel_num = self._get_selected(self.list_num)
        self.categorical_cols = [c for c in self.categorical_cols if c not in sel_cat]
        self.numeric_cols = [c for c in self.numeric_cols if c not in sel_num]
        self.refresh_lists()

    def _dbl_to_cat(self, event):
        idx = self.list_pool.curselection()
        if not idx: return
        c = self.list_pool.get(idx[0])
        if c not in self.categorical_cols:
            self.categorical_cols.append(c)
        self.refresh_lists()

    def _dbl_remove_cat(self, event):
        idx = self.list_cat.curselection()
        if not idx: return
        c = self.list_cat.get(idx[0])
        self.categorical_cols = [x for x in self.categorical_cols if x != c]
        self.refresh_lists()

    def _dbl_remove_num(self, event):
        idx = self.list_num.curselection()
        if not idx: return
        c = self.list_num.get(idx[0])
        self.numeric_cols = [x for x in self.numeric_cols if x != c]
        self.refresh_lists()

    # ---------- File & load ----------
    def select_csv(self):
        path = filedialog.askopenfilename(title="Seleccionar CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        self.csv_path = path
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, path)

    def load_csv(self):
        path = self.file_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Selecciona un CSV válido.")
            return
        sep = self.sep_var.get()
        if sep == "\\t":
            sep = "\t"
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
        except Exception as e:
            messagebox.showerror("Error al leer CSV", str(e))
            return

        self.df = df

        cols = list(df.columns)
        for combo in (self.combo_east, self.combo_north, self.combo_elev):
            combo["values"] = cols
            combo.set("")

        # Heurística inicial
        self.categorical_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        self.numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        # Scenario combos y targets
        self.scen1["values"] = cols
        self.scen2["values"] = cols
        self.scen3["values"] = cols

        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        self.combo_target["values"] = num_cols
        self.combo_scatter_x["values"] = num_cols if hasattr(self, "combo_scatter_x") else []
        self.combo_scatter_y["values"] = num_cols if hasattr(self, "combo_scatter_y") else []
        self.combo_scatter_hue["values"] = cols if hasattr(self, "combo_scatter_hue") else []

        # Numeric filter column list
        self.combo_filter_col["values"] = num_cols

        self.refresh_lists()
        self.status.set(f"CSV cargado: {os.path.basename(path)}  |  Filas: {len(df)}  Columnas: {len(cols)}")

    def save_selection(self):
        if self.df is None:
            messagebox.showerror("Error", "Carga primero un CSV.")
            return

        e, n, z = self.easting_col.get(), self.northing_col.get(), self.elevation_col.get()
        if not e or not n or not z:
            messagebox.showerror("Error", "Selecciona columnas de coordenadas (X/Y/Z).")
            return

        if not self.categorical_cols:
            messagebox.showerror("Error", "Marca al menos una columna como Categórica.")
            return
        if not self.numeric_cols:
            messagebox.showerror("Error", "Marca al menos una columna como Numérica.")
            return

        levels = (self.categorical_cols + ["", "", ""])[:3]
        self.scenario_level1.set(levels[0])
        self.scenario_level2.set(levels[1])
        self.scenario_level3.set(levels[2])

        self.combo_target["values"] = [c for c in self.numeric_cols]
        self.combo_scatter_x["values"] = self.combo_target["values"]
        self.combo_scatter_y["values"] = self.combo_target["values"]
        self.combo_scatter_hue["values"] = self.categorical_cols

        # default filter column
        if self.numeric_cols:
            self.filter_col.set(self.numeric_cols[0])

        self.status.set("Selección guardada. Ajusta el escenario y las opciones de salida.")

    # ---------- Helpers: filter & grouping ----------
    def _apply_numeric_filter(self, df):
        col = self.filter_col.get().strip()
        op = self.filter_op.get().strip()
        if not col or col not in df.columns:
            return df  # no filter
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            val = float(self.filter_val.get())
        except Exception:
            return df
        if op == ">=":
            mask = series >= val
        elif op == ">":
            mask = series > val
        elif op == "<=":
            mask = series <= val
        elif op == "<":
            mask = series < val
        elif op == "==":
            mask = series == val
        elif op == "!=":
            mask = series != val
        else:
            mask = pd.Series(True, index=df.index)
        return df.loc[mask]

    def _validate_and_get_fields(self):
        if self.df is None:
            messagebox.showerror("Error", "Carga primero un CSV.")
            return None
        e, n, z = self.easting_col.get(), self.northing_col.get(), self.elevation_col.get()
        if not e or not n or not z:
            messagebox.showerror("Error", "Selecciona columnas de coordenadas (X/Y/Z).")
            return None
        lvl1, lvl2, lvl3 = self.scenario_level1.get(), self.scenario_level2.get(), self.scenario_level3.get()
        if not lvl1 or not lvl2:
            messagebox.showerror("Error", "Define al menos Nivel 1 y Nivel 2 del escenario.")
            return None
        num = self.numeric_target.get()
        if not num:
            messagebox.showerror("Error", "Elige una variable numérica objetivo (N).")
            return None
        return dict(e=e, n=n, z=z, lvl1=lvl1, lvl2=lvl2, lvl3=lvl3, num=num)

    def _group_iter(self, df, lvl1, lvl2, lvl3):
        if lvl3:
            grouped = df.groupby([lvl1, lvl2, lvl3], dropna=False)
        else:
            grouped = df.groupby([lvl1, lvl2], dropna=False)
        for keys, sub in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = " | ".join([f"{kname}={kval}" for kname, kval in zip([lvl1, lvl2, lvl3] if lvl3 else [lvl1, lvl2], keys)])
            # Additionally return set key (lvl1,lvl2) for axis_scope='set'
            set_key = keys[:2] if len(keys) >= 2 else keys
            yield label, set_key, sub

    def _groups_with_min_count(self, df, lvl1, lvl2, lvl3, num, min_count):
        out = []
        for label, set_key, sub in self._group_iter(df, lvl1, lvl2, lvl3):
            vals = pd.to_numeric(sub[num], errors="coerce")
            n = vals.notna().sum()
            if n >= min_count:
                out.append((label, set_key, sub))
        return out

    # Axis extents
    def _swath_axis_extents(self, groups, coord_col, num, bins=30, agg="mean", clip_pct=0.0):
        xs_min, xs_max = [], []
        yvals = []
        for _, _, sub in groups:
            c = pd.to_numeric(sub[coord_col], errors="coerce").values
            v = pd.to_numeric(sub[num], errors="coerce").values
            centers, agg_vals, _ = swath_bins(c, v, nbins=bins, agg=agg)
            if centers.size > 0:
                xs_min.append(np.nanmin(centers))
                xs_max.append(np.nanmax(centers))
                yvals.extend(list(agg_vals[~np.isnan(agg_vals)]))
        if not xs_min:
            return None
        xmin, xmax = np.nanmin(xs_min), np.nanmax(xs_max)
        if yvals:
            arr = np.asarray(yvals, dtype=float)
            if clip_pct and clip_pct > 0:
                lo = np.nanpercentile(arr, clip_pct)
                hi = np.nanpercentile(arr, 100 - clip_pct)
            else:
                lo = np.nanmin(arr); hi = np.nanmax(arr)
            if lo == hi:
                lo -= 0.5; hi += 0.5
            return (xmin, xmax, lo, hi)
        return (xmin, xmax, None, None)

    def _ecdf_x_extent(self, groups, num, clip_pct=0.0):
        xs = []
        for _, _, sub in groups:
            v = pd.to_numeric(sub[num], errors="coerce").values
            v = v[~np.isnan(v)]
            xs.extend(list(v))
        if not xs:
            return None
        arr = np.asarray(xs, dtype=float)
        if clip_pct and clip_pct > 0:
            lo = np.nanpercentile(arr, clip_pct)
            hi = np.nanpercentile(arr, 100 - clip_pct)
        else:
            lo = np.nanmin(arr); hi = np.nanmax(arr)
        if lo == hi:
            lo -= 0.5; hi += 0.5
        return (lo, hi)

    # ---------- Actions ----------
    def action_ecdf(self):
        fields = self._validate_and_get_fields()
        if fields is None: return
        lvl1, lvl2, lvl3, num = fields["lvl1"], fields["lvl2"], fields["lvl3"], fields["num"]

        # Apply numeric base filter
        df_use = self._apply_numeric_filter(self.df)

        # Filter by min_count
        groups = self._groups_with_min_count(df_use, lvl1, lvl2, lvl3, num, int(self.min_count.get()))
        if not groups:
            messagebox.showinfo("ECDF", "No hay grupos que cumplan el mínimo de muestras.")
            return

        # Precompute extents (global or per set)
        xlims_global = None
        if self.fix_axes.get() and self.axis_scope.get() == "global":
            xlims_global = self._ecdf_x_extent(groups, num, clip_pct=float(self.y_clip_pct.get()))

        made = 0
        # Group by set_key if scope == set
        from collections import defaultdict
        set_groups = defaultdict(list)
        for label, set_key, sub in groups:
            set_groups[set_key].append((label, set_key, sub))

        for set_key, items in set_groups.items():
            xlims_set = None
            if self.fix_axes.get() and self.axis_scope.get() == "set":
                xlims_set = self._ecdf_x_extent(items, num, clip_pct=float(self.y_clip_pct.get()))

            for label, _, sub in items:
                x, y = ecdf(pd.to_numeric(sub[num], errors="coerce").values)
                if x.size == 0:
                    continue
                plt.figure()
                plt.plot(x, y, linewidth=1.2)
                plt.xlabel(num); plt.ylabel("ECDF")
                plt.title(f"ECDF — {num}\n{label}")
                plt.grid(True, linestyle="--", linewidth=0.5)
                if self.fix_axes.get():
                    xlims = xlims_global if self.axis_scope.get() == "global" else xlims_set
                    if xlims: plt.xlim(*xlims)
                plt.tight_layout()
                made += 1

        plt.show(); plt.close("all")
        self.status.set(f"ECDF: {made} gráfico(s) generados.")

    def action_swath(self):
        fields = self._validate_and_get_fields()
        if fields is None: return
        e, n, z = fields["e"], fields["n"], fields["z"]
        lvl1, lvl2, lvl3, num = fields["lvl1"], fields["lvl2"], fields["lvl3"], fields["num"]
        axis = self.swath_axis.get()
        bins = int(self.swath_bins_var.get())
        agg = self.swath_agg.get()
        coord_col = {"Easting": e, "Northing": n, "Elevation": z}[axis]

        # Apply numeric base filter
        df_use = self._apply_numeric_filter(self.df)

        # Filter by min_count
        groups = self._groups_with_min_count(df_use, lvl1, lvl2, lvl3, num, int(self.min_count.get()))
        if not groups:
            messagebox.showinfo("Swath", "No hay grupos que cumplan el mínimo de muestras.")
            return

        from collections import defaultdict
        set_groups = defaultdict(list)
        for label, set_key, sub in groups:
            set_groups[set_key].append((label, set_key, sub))

        # Precompute axis extents
        ext_global = None
        if self.fix_axes.get() and self.axis_scope.get() == "global":
            ext_global = self._swath_axis_extents(groups, coord_col, num, bins=bins, agg=agg, clip_pct=float(self.y_clip_pct.get()))

        made = 0
        for set_key, items in set_groups.items():
            ext_set = None
            if self.fix_axes.get() and self.axis_scope.get() == "set":
                ext_set = self._swath_axis_extents(items, coord_col, num, bins=bins, agg=agg, clip_pct=float(self.y_clip_pct.get()))
            for label, _, sub in items:
                centers, agg_vals, counts = swath_bins(
                    pd.to_numeric(sub[coord_col], errors="coerce").values,
                    pd.to_numeric(sub[num], errors="coerce").values,
                    nbins=bins, agg=agg
                )
                if centers.size == 0:
                    continue
                plt.figure()
                plt.plot(centers, agg_vals, linewidth=1.2)
                plt.xlabel(coord_col); plt.ylabel(f"{num} ({agg})")
                plt.title(f"Swath ({axis}, {agg}) — {num}\n{label}")
                plt.grid(True, linestyle="--", linewidth=0.5)
                if self.fix_axes.get():
                    ext = ext_global if self.axis_scope.get() == "global" else ext_set
                    if ext:
                        xmin, xmax, ymin, ymax = ext
                        plt.xlim(xmin, xmax)
                        if ymin is not None and ymax is not None:
                            plt.ylim(ymin, ymax)
                plt.tight_layout()
                made += 1

        plt.show(); plt.close("all")
        self.status.set(f"Swath: {made} gráfico(s) generados.")

    def action_stats(self):
        fields = self._validate_and_get_fields()
        if fields is None: return
        lvl1, lvl2, lvl3, num = fields["lvl1"], fields["lvl2"], fields["lvl3"], fields["num"]

        # Apply numeric base filter
        df_use = self._apply_numeric_filter(self.df)

        rows = []
        for label, set_key, sub in self._group_iter(df_use, lvl1, lvl2, lvl3):
            vals = pd.to_numeric(sub[num], errors="coerce").to_numpy()
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            rows.append({
                "group": label,
                "count": int(vals.size),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "mean": float(np.nanmean(vals)),
                "p75": float(np.nanpercentile(vals, 75)),
            })
        if not rows:
            messagebox.showinfo("Estadísticas", "No hay datos válidos para exportar.")
            return
        out_df = pd.DataFrame(rows).sort_values(by="group").reset_index(drop=True)

        base = os.path.splitext(os.path.basename(self.csv_path or self.file_entry.get().strip()))[0] if (self.csv_path or self.file_entry.get().strip()) else "stats"
        out_name = f"{base}_stats_{lvl1}_{lvl2}_{lvl3 or 'NA'}_{num}.csv"
        out_path = os.path.join(os.path.dirname(self.csv_path or "."), out_name)
        try:
            out_df.to_csv(out_path, index=False)
        except Exception as e:
            messagebox.showerror("Error al guardar CSV", str(e)); return
        messagebox.showinfo("Estadísticas", f"Archivo exportado:\n{out_path}")
        self.status.set(f"Estadísticas exportadas: {out_name}")

    def action_scatter(self):
        if self.df is None:
            messagebox.showerror("Error", "Carga primero un CSV."); return
        xcol = self.scatter_x.get(); ycol = self.scatter_y.get(); hue = self.scatter_hue.get().strip()
        if not xcol or not ycol:
            messagebox.showerror("Error", "Selecciona columnas X e Y para el scatter."); return

        # Apply numeric base filter
        df_use = self._apply_numeric_filter(self.df)

        cols = [xcol, ycol] + ([hue] if hue else [])
        df = df_use[cols].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[xcol, ycol])

        plt.figure()
        if hue and hue in df.columns:
            for k, sub in df.groupby(hue, dropna=False):
                plt.scatter(sub[xcol].values, sub[ycol].values, s=10, label=str(k), alpha=0.7)
            plt.legend(title=hue, fontsize=8)
        else:
            plt.scatter(df[xcol].values, df[ycol].values, s=10, alpha=0.7)
        plt.xlabel(xcol); plt.ylabel(ycol)
        plt.title(f"Scatter: {xcol} vs {ycol}" + (f"  |  color={hue}" if hue else ""))
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()

        plt.show(); plt.close("all")
        self.status.set("Scatter generado.")

    def run(self):
        self.mainloop()


if __name__ == "__main__":
    GeoApp().run()
